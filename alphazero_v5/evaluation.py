"""Fixed BEST checkpoint loading and final V5 acceptance evaluation helpers."""

from dataclasses import dataclass
from collections import defaultdict
import copy
import json
from pathlib import Path

import numpy as np
import torch

from alphazero_v2.encoder import encode_state as encode_v2_state
from alphazero_v2.evaluate import load_evaluation_checkpoint
from alphazero_v2.mcts import visit_count_policy
from alphazero_v3.encoder import encode_state as encode_v3_state
from alphazero_v4.acceptance import generate_acceptance_openings, strategic_positions
from alphazero_v4.evaluation import load_v4_evaluation_checkpoint
from alphazero_v4.network import value_logit_to_scalar
from alphazero_v4.tactical import solve_tactical_root
from great_kingdom_v2 import BLUE, RED, NUM_ACTIONS, PASS_ACTION, GreatKingdomLogicV2

from .arena import ArenaAgent, play_paired_arena, summarize_arena, _root_records
from .batched_mcts import BatchedMCTS, BatchedNetworkEvaluator, SearchRequest
from .config import V5Config
from .encoder import encode_state
from .network import PolicyValueAuxNetwork, calibrated_scalar
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
        """Compatibility label for shared UI logging; V5 is cycle/version based."""
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
    if payload.get("format_version") != FORMAT_VERSION or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("not an AlphaZero V5 strategy checkpoint")
    config = V5Config.from_dict(payload["config"])
    network = PolicyValueAuxNetwork(config.channels, config.residual_blocks, config.input_planes).to(device)
    network.load_state_dict(payload["network_state_dict"])
    network.eval()
    temperature = float(payload["calibration_temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid V5 calibration temperature")
    return V5EvaluationCheckpoint(path, int(payload["cycle"]), int(payload["best_version"]),
                                  config, network, temperature, device)


def select_v5_with_diagnostics(checkpoint, logic, simulations=256, top_k=5):
    tactical = solve_tactical_root(logic)
    if tactical.mode == "IMMEDIATE_WIN":
        action = min(tactical.immediate_win_actions)
        return {"action": action, "tactical_mode": tactical.mode,
                "top_actions": [{"action": a, "exact_q": 1.0} for a in tactical.immediate_win_actions],
                "mcts_executed": False, "safe_defense_actions": [],
                "unsafe_actions_filtered": 0}
    evaluator = BatchedNetworkEvaluator(
        checkpoint.network, encode_state, checkpoint.device,
        lambda logits: calibrated_scalar(logits, checkpoint.calibration_temperature))
    root = BatchedMCTS(evaluator, checkpoint.config.c_puct,
                       checkpoint.config.dirichlet_alpha, 0.0).run(
        logic, simulations=int(simulations), root_actions=tactical.allowed_actions).root
    policy = visit_count_policy(root, temperature=0)
    action = int(np.argmax(policy))
    return {"action": action, "tactical_mode": tactical.mode,
            "top_actions": _root_records(root, logic, top_k), "mcts_executed": True,
            "safe_defense_actions": list(tactical.safe_defense_actions),
            "unsafe_actions_filtered": len(tactical.exact_unsafe_actions) if tactical.mode == "SAFE_DEFENSE" else 0}


def v5_agent(checkpoint, name="v5_best"):
    return ArenaAgent(name, checkpoint.network, encode_state,
                      lambda logits: calibrated_scalar(logits, checkpoint.calibration_temperature),
                      checkpoint.device, 256, 1.5, True)


def historical_agents(v5_device, v4_path, v3_path, v2_path=None):
    agents = {}
    v4 = load_v4_evaluation_checkpoint(v4_path, v5_device, expected_iteration=50)
    agents["v4_iter50"] = ArenaAgent("v4_iter50", v4.network, encode_v3_state,
                                     value_logit_to_scalar, v5_device, 256, 1.5, True)
    v3 = load_evaluation_checkpoint(v3_path, v5_device, expected_iteration=50,
                                    state_encoder=encode_v3_state)
    agents["v3_iter50"] = ArenaAgent("v3_iter50", v3.network, encode_v3_state,
                                     lambda value: value, v5_device, 256, 1.5, False)
    if v2_path is not None and Path(v2_path).is_file():
        v2 = load_evaluation_checkpoint(v2_path, v5_device, expected_iteration=375,
                                        state_encoder=encode_v2_state)
        agents["v2_iter375"] = ArenaAgent("v2_iter375", v2.network, encode_v2_state,
                                          lambda value: value, v5_device, 256, 1.5, False)
    return agents


def paired_bootstrap(games, candidate_name, replicates=10_000, seed=20260908):
    grouped = defaultdict(list)
    for game in games:
        grouped[game["opening_id"]].append(game)
    pair_scores = []
    for opening_id in sorted(grouped):
        pair = grouped[opening_id]
        if len(pair) != 2 or pair[0]["opening_actions"] != pair[1]["opening_actions"]:
            raise ValueError("paired bootstrap requires two color-swapped identical openings")
        colors = {(game["blue_agent"] == candidate_name, game["red_agent"] == candidate_name) for game in pair}
        if colors != {(True, False), (False, True)}:
            raise ValueError("candidate color swap missing")
        pair_scores.append(np.mean([game["winner_agent"] == candidate_name for game in pair]))
    scores = np.asarray(pair_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = scores[rng.integers(len(scores), size=(int(replicates), len(scores)))].mean(axis=1)
    return {"pairs": len(scores), "replicates": int(replicates), "seed": int(seed),
            "win_rate": float(scores.mean()), "win_rate_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "difference_from_50": float(scores.mean() - 0.5),
            "difference_from_50_ci95": (np.quantile(samples, [0.025, 0.975]) - 0.5).tolist()}


def run_final_arenas(v5, opponents, openings, concurrent_games=16):
    before = network_digest(v5.network)
    results, all_games = {}, []
    for name, opponent in opponents.items():
        games, inference = play_paired_arena(v5_agent(v5), opponent, openings, concurrent_games)
        summary = summarize_arena(games, "v5_best")
        summary["paired_bootstrap"] = paired_bootstrap(games, "v5_best")
        summary["inference"] = inference
        results[name] = summary
        for game in games:
            compact = dict(game)
            compact.pop("moves", None)
            all_games.append(compact)
    if network_digest(v5.network) != before:
        raise RuntimeError("final evaluation mutated V5 network")
    return results, all_games


def run_strategic_suite(checkpoint):
    positions = list(strategic_positions())
    from .diagnostics import positional_openings
    positions.extend((name, logic, False) for name, logic in positional_openings())
    results = []
    for name, logic, exact in positions:
        before = logic.territory_counts()
        selected = select_v5_with_diagnostics(checkpoint, logic)
        tactical = solve_tactical_root(logic)
        child = logic.copy(); result = child.apply_action(selected["action"])
        after = child.territory_counts()
        if exact:
            success = (selected["action"] in tactical.immediate_win_actions
                       if tactical.mode == "IMMEDIATE_WIN"
                       else selected["action"] in tactical.safe_defense_actions)
        else:
            success = None
        results.append({"scenario": name, "exact_assertion": exact, "success": success,
                        "selected_action": selected["action"], "is_pass": selected["action"] == PASS_ACTION,
                        "tactical_mode": selected["tactical_mode"], "top_actions": selected["top_actions"],
                        "result": result.name, "territory_before": before, "territory_after": after,
                        "territory_delta": {str(p): after[p] - before[p] for p in (BLUE, RED)}})
    return results


def restore_serialized_logic(payload):
    logic = GreatKingdomLogicV2()
    logic.board = [list(row) for row in payload["board"]]
    logic.turn = int(payload["turn"])
    logic.consecutive_passes = int(payload["consecutive_passes"])
    logic.castles_remaining = {int(key): int(value) for key, value in payload["castles_remaining"].items()}
    return logic


def extract_human_challenge_states(log_dir, maximum=20):
    records = []
    for path in sorted(Path(log_dir).glob("game_*.json")):
        payload = json.loads(path.read_text())
        logic = GreatKingdomLogicV2()
        for move in payload.get("moves", []):
            if move["actor"] == "AI":
                action = int(move["action"])
                edge = action != PASS_ACTION and (action % 9 in (0, 8) or action // 9 in (0, 8))
                priority = 3 if move["tactical_mode"] == "FORCED_LOSS" else 2 if edge else 1
                records.append({"source": path.name, "ply": int(move["ply"]), "priority": priority,
                                "v4_tactical_mode": move["tactical_mode"], "v4_action": action,
                                "board": copy.deepcopy(logic.board), "turn": logic.turn,
                                "consecutive_passes": logic.consecutive_passes,
                                "castles_remaining": dict(logic.castles_remaining)})
            result = logic.apply_action(int(move["action"]))
            if result.name.startswith("IMPOSSIBLE"):
                raise RuntimeError(f"invalid human log {path} at ply {move['ply']}")
    records.sort(key=lambda row: (-row["priority"], row["source"], row["ply"]))
    selected = records[:int(maximum)]
    for index, row in enumerate(selected):
        row["challenge_id"] = index
    return selected


def evaluate_human_challenge(v5, v4, states):
    from alphazero_v4.acceptance import select_v4_with_diagnostics
    output = []
    for state in states:
        logic = restore_serialized_logic(state)
        v5_selection = select_v5_with_diagnostics(v5, logic)
        v4_selection = select_v4_with_diagnostics(v4, logic, 256, top_k=5)
        tactical = solve_tactical_root(logic)
        child = logic.copy(); child.apply_action(v5_selection["action"])
        before, after = logic.territory_counts(), child.territory_counts()
        output.append({**state, "v5": v5_selection,
                       "v4_recomputed": {"action": v4_selection["action"],
                                         "tactical_mode": v4_selection["tactical_mode"],
                                         "top_actions": v4_selection["top_actions"]},
                       "forced_loss_risk": tactical.mode == "FORCED_LOSS",
                       "v5_territory_delta": {str(p): after[p] - before[p] for p in (BLUE, RED)}})
    return output
