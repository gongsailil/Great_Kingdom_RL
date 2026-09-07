"""Fixed tactical and symmetry diagnostics for AlphaZero V5."""

from dataclasses import dataclass

import numpy as np
import torch

from alphazero_v2.evaluate import (
    apply_opening,
    capture_tactical_position,
    defense_tactical_position,
    winning_pass_tactical_position,
)
from alphazero_v2.mcts import visit_count_policy
from alphazero_v4.tactical import solve_tactical_root
from great_kingdom_v2 import NUM_ACTIONS

from .batched_mcts import BatchedMCTS, BatchedNetworkEvaluator, SearchRequest
from .encoder import encode_state
from .network import calibrated_scalar
from .symmetry import inverse_policy, transform_state


@dataclass(frozen=True)
class DiagnosticCheckpoint:
    network: object
    calibration_temperature: float
    device: object


def _select(checkpoint, config, logic):
    tactical = solve_tactical_root(logic)
    if tactical.mode == "IMMEDIATE_WIN":
        return min(tactical.immediate_win_actions), tactical
    evaluator = BatchedNetworkEvaluator(
        checkpoint.network,
        encode_state,
        checkpoint.device,
        lambda logits: calibrated_scalar(logits, checkpoint.calibration_temperature),
    )
    search = BatchedMCTS(evaluator, config.c_puct, config.dirichlet_alpha, 0.0)
    root = search.run(
        logic,
        simulations=config.arena_mcts_simulations,
        root_actions=tactical.allowed_actions,
    ).root
    return int(np.argmax(visit_count_policy(root, temperature=0))), tactical


def run_fixed_tactical_diagnostics(network, temperature, config, device):
    checkpoint = DiagnosticCheckpoint(network, float(temperature), device)
    fixtures = []
    capture_logic = capture_tactical_position()
    capture_action = 1 + 2 * 9
    defense_logic = defense_tactical_position()
    pass_logic = winning_pass_tactical_position()
    for name, logic, exact_action in (
        ("immediate_capture", capture_logic, capture_action),
        ("safe_defense", defense_logic, None),
        ("winning_pass", pass_logic, 81),
    ):
        action, tactical = _select(checkpoint, config, logic)
        if name == "safe_defense":
            success = action in tactical.safe_defense_actions
        else:
            success = action == exact_action
        fixtures.append(
            {
                "fixture": name,
                "selected_action": int(action),
                "expected_action": exact_action,
                "tactical_mode": tactical.mode,
                "safe_defense_actions": list(tactical.safe_defense_actions),
                "success": bool(success),
            }
        )
    return {"fixtures": fixtures, "all_success": all(row["success"] for row in fixtures)}


def symmetry_consistency(network, temperature, device, openings, maximum_states=8):
    """Compare raw policy/value predictions over all transformed fixed states."""
    logics = [apply_opening(item["actions"]) for item in openings[:maximum_states]]
    l1_values, kl_values, value_differences = [], [], []
    was_training = network.training
    network.eval()
    with torch.no_grad():
        for logic in logics:
            state = encode_state(logic)
            base_output = network(torch.from_numpy(state[None]).to(device))
            base_policy = torch.softmax(base_output[0][0], dim=0).cpu().numpy()
            base_value = float(calibrated_scalar(base_output[1], temperature)[0].item())
            for transform in range(1, 8):
                transformed = transform_state(state, transform).astype(np.float32)
                output = network(torch.from_numpy(transformed[None]).to(device))
                policy = inverse_policy(torch.softmax(output[0][0], dim=0).cpu().numpy(), transform)
                value = float(calibrated_scalar(output[1], temperature)[0].item())
                l1_values.append(float(np.abs(base_policy - policy).sum()))
                kl_values.append(float(np.sum(base_policy * np.log((base_policy + 1e-12) / (policy + 1e-12)))))
                value_differences.append(abs(base_value - value))
    if was_training:
        network.train()
    return {
        "states": len(logics),
        "transformed_comparisons": len(l1_values),
        "mean_policy_l1": float(np.mean(l1_values)) if l1_values else 0.0,
        "max_policy_l1": float(np.max(l1_values)) if l1_values else 0.0,
        "mean_policy_kl": float(np.mean(kl_values)) if kl_values else 0.0,
        "mean_value_abs_difference": float(np.mean(value_differences)) if value_differences else 0.0,
        "max_value_abs_difference": float(np.max(value_differences)) if value_differences else 0.0,
    }


def positional_openings():
    """Rules-valid, non-label positional states used only for diagnostics."""
    prefixes = {
        "territory_rich": [2, 80, 11, 79, 20, 78, 18, 77, 19, 76],
        "score_behind": [1, 80, 9, 79],
        "score_ahead": [1, 80, 9],
        "own_territory_choice": [9, 80, 10, 79, 2, 78],
    }
    return [(name, apply_opening(actions)) for name, actions in prefixes.items()]
