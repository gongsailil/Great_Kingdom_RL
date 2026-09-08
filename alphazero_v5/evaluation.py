"""Self-contained checkpoint and strategic evaluation for AlphaZero V5."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from great_kingdom_v2 import BLUE, RED, PASS_ACTION, GreatKingdomLogicV2

from .arena import _root_records
from .batched_mcts import BatchedMCTS, BatchedNetworkEvaluator
from .common import visit_count_policy
from .config import V5Config
from .encoder import encode_state
from .network import PolicyValueAuxNetwork, calibrated_scalar
from .openings import strategic_positions
from .tactical import solve_tactical_root
from .training_runner import ARCHITECTURE, FORMAT_VERSION, network_digest


@dataclass
class V5EvaluationCheckpoint:
    path: Path
    cycle: int
    best_version: int
    config: V5Config
    network: PolicyValueAuxNetwork
    calibration_temperature: float
    device: torch.device

    @property
    def iteration(self):
        """Compatibility label for UI records; V5 is cycle/version based."""
        return self.cycle


def load_v5_checkpoint(path, device="auto"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("format_version") != FORMAT_VERSION
        or payload.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("not an AlphaZero V5 strategy checkpoint")
    config = V5Config.from_dict(payload["config"])
    network = PolicyValueAuxNetwork(
        config.channels, config.residual_blocks, config.input_planes
    ).to(device)
    network.load_state_dict(payload["network_state_dict"])
    network.eval()
    temperature = float(payload["calibration_temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid V5 calibration temperature")
    return V5EvaluationCheckpoint(
        path, int(payload["cycle"]), int(payload["best_version"]),
        config, network, temperature, device
    )


def select_v5_with_diagnostics(checkpoint, logic, simulations=256, top_k=5):
    if int(simulations) <= 0:
        raise ValueError("MCTS simulations must be positive")
    tactical = solve_tactical_root(logic)
    if tactical.mode == "IMMEDIATE_WIN":
        action = min(tactical.immediate_win_actions)
        return {
            "action": action, "tactical_mode": tactical.mode,
            "top_actions": [
                {"action": item, "exact_q": 1.0}
                for item in tactical.immediate_win_actions
            ],
            "mcts_executed": False, "safe_defense_actions": [],
            "unsafe_actions_filtered": 0,
        }
    evaluator = BatchedNetworkEvaluator(
        checkpoint.network, encode_state, checkpoint.device,
        lambda logits: calibrated_scalar(
            logits, checkpoint.calibration_temperature
        ),
    )
    root = BatchedMCTS(
        evaluator, checkpoint.config.c_puct,
        checkpoint.config.dirichlet_alpha, 0.0
    ).run(
        logic, simulations=int(simulations),
        root_actions=tactical.allowed_actions
    ).root
    action = int(np.argmax(visit_count_policy(root, temperature=0)))
    return {
        "action": action, "tactical_mode": tactical.mode,
        "top_actions": _root_records(root, logic, top_k),
        "mcts_executed": True,
        "safe_defense_actions": list(tactical.safe_defense_actions),
        "unsafe_actions_filtered": (
            len(tactical.exact_unsafe_actions)
            if tactical.mode == "SAFE_DEFENSE" else 0
        ),
    }


def paired_bootstrap(games, candidate_name, replicates=10_000, seed=20260908):
    """Generic paired-opening statistic retained for checkpoint comparisons."""
    grouped = defaultdict(list)
    for game in games:
        grouped[game["opening_id"]].append(game)
    pair_scores = []
    for opening_id in sorted(grouped):
        pair = grouped[opening_id]
        if len(pair) != 2 or pair[0]["opening_actions"] != pair[1]["opening_actions"]:
            raise ValueError("paired bootstrap requires identical color-swapped openings")
        colors = {
            (game["blue_agent"] == candidate_name,
             game["red_agent"] == candidate_name)
            for game in pair
        }
        if colors != {(True, False), (False, True)}:
            raise ValueError("candidate color swap missing")
        pair_scores.append(np.mean([
            game["winner_agent"] == candidate_name for game in pair
        ]))
    scores = np.asarray(pair_scores, dtype=np.float64)
    if not len(scores):
        raise ValueError("paired bootstrap requires at least one opening pair")
    rng = np.random.default_rng(seed)
    samples = scores[
        rng.integers(len(scores), size=(int(replicates), len(scores)))
    ].mean(axis=1)
    return {
        "pairs": len(scores), "replicates": int(replicates), "seed": int(seed),
        "win_rate": float(scores.mean()),
        "win_rate_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def run_strategic_suite(checkpoint, simulations=256):
    """Run exact tactical checks and diagnostic-only positional scenarios."""
    results = []
    for name, logic, exact in strategic_positions():
        player = logic.turn
        before = logic.territory_counts()
        selection = select_v5_with_diagnostics(
            checkpoint, logic, simulations=simulations, top_k=5
        )
        tactical = solve_tactical_root(logic)
        child = logic.copy()
        result = child.apply_action(selection["action"])
        after = child.territory_counts()
        if exact:
            success = (
                child.game_over and child.winner == player
                if tactical.mode == "IMMEDIATE_WIN"
                else selection["action"] in tactical.safe_defense_actions
            )
        else:
            success = None
        results.append({
            "scenario": name, "exact_assertion": exact, "success": success,
            "selected_action": selection["action"],
            "is_pass": selection["action"] == PASS_ACTION,
            "tactical_mode": selection["tactical_mode"],
            "top_actions": selection["top_actions"], "result": result.name,
            "territory_before": {str(p): before[p] for p in (BLUE, RED)},
            "territory_after": {str(p): after[p] for p in (BLUE, RED)},
        })
    return results


def validate_checkpoint(path, device="auto"):
    """Load twice and verify stable metadata, weights, and initial output."""
    first = load_v5_checkpoint(path, device)
    second = load_v5_checkpoint(path, first.device)
    tensor = torch.from_numpy(
        encode_state(GreatKingdomLogicV2())[None]
    ).to(first.device)
    with torch.no_grad():
        outputs_a = first.network(tensor)
        outputs_b = second.network(tensor)
    return {
        "cycle": first.cycle, "best_version": first.best_version,
        "parameter_count": first.network.parameter_count(),
        "calibration_temperature": first.calibration_temperature,
        "network_digest": network_digest(first.network),
        "reload_output_equal": bool(
            all(torch.equal(a, b) for a, b in zip(outputs_a, outputs_b))
        ),
    }
