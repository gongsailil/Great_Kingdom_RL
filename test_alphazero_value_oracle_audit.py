"""Tests for Rules-exact V3 replay value diagnostics."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from alphazero_v2.evaluate import (
    EvaluationCheckpoint,
    capture_tactical_position,
    defense_tactical_position,
    winning_pass_tactical_position,
)
from alphazero_v3.encoder import encode_state
from alphazero_v3.temperature_audit import network_state_digest
from alphazero_v3.value_oracle_audit import (
    EXACT_LOSS,
    audit_networks_on_states,
    classify_oracle_state,
    decode_v3_state,
    immediate_winning_actions,
    predict_values,
    root_q_from_child_value,
    select_audit_indices,
    validate_replay,
)
from great_kingdom_v2 import (
    BLUE,
    NUM_ACTIONS,
    PASS_ACTION,
    GreatKingdomLogicV2,
    MoveResultV2,
)


class CountingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.evaluated_states = 0

    def forward(self, inputs):
        self.evaluated_states += len(inputs)
        logits = torch.zeros(
            (len(inputs), NUM_ACTIONS),
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        values = torch.full(
            (len(inputs),),
            0.25,
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        return logits, values


def fake_checkpoint(network, iteration=10):
    return EvaluationCheckpoint(
        path=Path(f"iteration_{iteration:06d}.pt"),
        iteration=iteration,
        config={"channels": 8, "residual_blocks": 1, "input_planes": 9},
        network=network,
        device=torch.device("cpu"),
        state_encoder=encode_state,
    )


def test_v3_replay_decode_encode_roundtrip_and_perspective():
    logic = GreatKingdomLogicV2()
    assert logic.apply_action(0) == MoveResultV2.NORMAL
    assert logic.apply_action(80) == MoveResultV2.NORMAL
    assert logic.apply_action(PASS_ACTION) == MoveResultV2.PASS
    encoded = encode_state(logic)
    decoded = decode_v3_state(encoded)
    assert decoded.turn == logic.turn
    assert decoded.consecutive_passes == 1
    assert decoded.board == logic.board
    assert decoded.castles_remaining == logic.castles_remaining
    assert np.array_equal(encode_state(decoded), encoded)


def test_child_value_sign_conversion():
    assert root_q_from_child_value(0.75) == -0.75
    assert root_q_from_child_value(-0.25) == 0.25


def test_exact_immediate_win_and_allows_immediate_loss_classes():
    capture = capture_tactical_position()
    oracle = classify_oracle_state(capture)
    assert oracle.immediate_win_actions == [19]
    assert 19 in immediate_winning_actions(capture)

    defense = defense_tactical_position()
    oracle = classify_oracle_state(defense)
    assert oracle.original_opponent_threat_actions == [19]
    assert oracle.defense_opportunity
    assert [child.action for child in oracle.safe_nonterminal_children] == [19]
    assert oracle.exact_loss_children
    unsafe = oracle.exact_loss_children[0]
    assert unsafe.classification == EXACT_LOSS
    child = defense.copy()
    assert child.apply_action(unsafe.action) in (
        MoveResultV2.NORMAL,
        MoveResultV2.PASS,
    )
    assert immediate_winning_actions(child) == unsafe.opponent_winning_actions

    winning_pass = winning_pass_tactical_position()
    oracle = classify_oracle_state(winning_pass)
    assert PASS_ACTION in oracle.immediate_win_actions

    # Blue PASS is nonterminal on the initial board; Red can then PASS and win
    # the exact 0-0 score, so the first PASS is an exact -1 root action.
    initial = classify_oracle_state(GreatKingdomLogicV2())
    pass_child = next(
        child for child in initial.children if child.action == PASS_ACTION
    )
    assert pass_child.classification == EXACT_LOSS
    assert pass_child.opponent_winning_actions == [PASS_ACTION]


def test_terminal_actions_are_not_sent_to_network():
    oracle = classify_oracle_state(capture_tactical_position())
    assert 19 in oracle.immediate_win_actions
    assert all(child.action != 19 for child in oracle.children)
    network = CountingNetwork()
    checkpoint = fake_checkpoint(network)
    child_states = np.stack([child.encoded_child for child in oracle.children])
    values = predict_values(checkpoint, child_states, batch_size=16)
    assert len(values) == len(oracle.children)
    assert network.evaluated_states == len(oracle.children)


def test_same_deterministic_subset_and_contradictory_exact_duplicates():
    first = select_audit_indices(20_000, maximum=10_000, seed=20260902)
    second = select_audit_indices(20_000, maximum=10_000, seed=20260902)
    assert np.array_equal(first, second)
    assert len(first) == 10_000

    state = encode_state(GreatKingdomLogicV2())
    validation = validate_replay(
        np.stack((state, state)),
        np.asarray((-1.0, 1.0), dtype=np.float32),
        np.asarray((BLUE, BLUE), dtype=np.int8),
    )
    assert validation["duplicate_unique_state_count"] == 1
    assert validation["contradictory_z_unique_state_count"] == 1
    assert validation["contradictory_z_sample_count"] == 2


def test_value_inference_does_not_change_network_state():
    network = CountingNetwork()
    checkpoint = fake_checkpoint(network)
    state = encode_state(GreatKingdomLogicV2())
    before = network_state_digest(network)
    values = predict_values(checkpoint, np.stack((state, state)))
    after = network_state_digest(network)
    assert np.allclose(values, 0.25)
    assert before == after


def test_iteration_networks_share_identical_state_action_subset():
    state = encode_state(GreatKingdomLogicV2())
    first = fake_checkpoint(CountingNetwork(), iteration=10)
    second = fake_checkpoint(CountingNetwork(), iteration=50)
    audit = audit_networks_on_states(
        [first, second], np.stack((state,)), np.asarray((0,)), chunk_size=1
    )
    assert set(audit["networks"]) == {"10", "50"}
    assert (
        audit["networks"]["10"]["exact_loss"]["count"]
        == audit["networks"]["50"]["exact_loss"]["count"]
        == audit["state_class_counts"]["exact_loss_actions"]
    )
    assert (
        audit["networks"]["10"]["saturation"]["predicted_root_q"]["count"]
        == audit["networks"]["50"]["saturation"]["predicted_root_q"]["count"]
    )


if __name__ == "__main__":
    test_v3_replay_decode_encode_roundtrip_and_perspective()
    test_child_value_sign_conversion()
    test_exact_immediate_win_and_allows_immediate_loss_classes()
    test_terminal_actions_are_not_sent_to_network()
    test_same_deterministic_subset_and_contradictory_exact_duplicates()
    test_value_inference_does_not_change_network_state()
    test_iteration_networks_share_identical_state_action_subset()
    print("AlphaZero value-oracle audit tests: PASS")
