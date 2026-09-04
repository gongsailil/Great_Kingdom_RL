"""Fixed tactical and value-oracle monitors for AlphaZero V4."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from alphazero_v2.evaluate import (
    EvaluationCheckpoint,
    capture_tactical_position,
    defense_tactical_position,
    winning_pass_tactical_position,
)
from alphazero_v3.encoder import encode_state
from alphazero_v3.temperature_audit import network_state_digest
from alphazero_v3.value_oracle_audit import (
    audit_networks_on_states,
    select_audit_indices,
)
from great_kingdom_v2 import BOARD_SIZE, PASS_ACTION

from .network import ScalarValueNetworkAdapter
from .self_play import select_root_action
from .tactical import solve_tactical_root


VALUE_ORACLE_ITERATIONS = (10, 30, 50)
VALUE_ORACLE_SEED = 20260902


def _fixture_result(network, config, device, logic, expected_action, iteration):
    tactical = solve_tactical_root(logic)
    selected = select_root_action(
        network,
        logic,
        config,
        device,
        np.random.default_rng(1000 + int(iteration)),
        ply=8,
        add_root_noise=False,
    )
    return {
        "solver_mode": tactical.mode,
        "selected_action": selected.action,
        "expected_action": int(expected_action),
        "success": selected.action == int(expected_action),
        "immediate_win_actions": list(tactical.immediate_win_actions),
        "opponent_threat_actions": list(tactical.opponent_threat_actions),
        "safe_defense_actions": list(tactical.safe_defense_actions),
        "exact_unsafe_actions": list(tactical.exact_unsafe_actions),
    }


def run_fixed_tactical_diagnostics(state, config, device):
    before = network_state_digest(state.network)
    action = 1 + 2 * BOARD_SIZE
    capture = _fixture_result(
        state.network,
        config,
        device,
        capture_tactical_position(),
        action,
        state.iteration,
    )
    defense = _fixture_result(
        state.network,
        config,
        device,
        defense_tactical_position(),
        action,
        state.iteration,
    )
    winning_pass = _fixture_result(
        state.network,
        config,
        device,
        winning_pass_tactical_position(),
        PASS_ACTION,
        state.iteration,
    )
    after = network_state_digest(state.network)
    if before != after:
        raise RuntimeError("tactical diagnostics changed V4 network state")
    return {
        "iteration": int(state.iteration),
        "mcts_simulations": int(config.mcts_simulations),
        "immediate_capture": capture,
        "safe_defense": defense,
        "winning_pass": winning_pass,
        "all_success": bool(
            capture["success"] and defense["success"] and winning_pass["success"]
        ),
        "network_unchanged": True,
    }


def run_value_oracle_monitor(
    state,
    config,
    device,
    replay_path,
    *,
    maximum=None,
    progress_callback=None,
):
    replay_path = Path(replay_path)
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    states = np.asarray(replay["states"], dtype=np.float32)
    limit = config.value_oracle_max_states if maximum is None else int(maximum)
    indices = select_audit_indices(
        len(states), maximum=limit, seed=VALUE_ORACLE_SEED
    )
    adapter = ScalarValueNetworkAdapter(state.network)
    checkpoint = EvaluationCheckpoint(
        path=Path(f"iteration_{state.iteration:06d}.pt"),
        iteration=int(state.iteration),
        config=config.to_dict(),
        network=adapter,
        device=torch.device(device),
        state_encoder=encode_state,
    )
    before = network_state_digest(state.network)
    audit = audit_networks_on_states(
        [checkpoint],
        states,
        indices,
        chunk_size=128,
        progress_callback=progress_callback,
    )
    after = network_state_digest(state.network)
    if before != after:
        raise RuntimeError("value-oracle monitor changed V4 network state")
    network = audit["networks"][str(state.iteration)]
    return {
        "iteration": int(state.iteration),
        "reference_replay_path": str(replay_path),
        "reference_replay_states": len(states),
        "audited_states": len(indices),
        "audited_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "state_class_counts": audit["state_class_counts"],
        "audited_player_counts": audit["audited_player_counts"],
        "exact_loss": network["exact_loss"],
        "exact_loss_by_root_player": network["exact_loss_by_root_player"],
        "exact_loss_by_action_type": network["exact_loss_by_action_type"],
        "defense": network["defense"],
        "immediate_win_alternatives": network["immediate_win_alternatives"],
        "saturation": network["saturation"],
        "network_unchanged": True,
    }
