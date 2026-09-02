"""Lightweight tests for the fixed-network temperature audit."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from alphazero_v2.encoder import encode_state
from alphazero_v2.evaluate import (
    EvaluationCheckpoint,
    capture_tactical_position,
    defense_tactical_position,
    winning_pass_tactical_position,
)
from alphazero_v2.mcts import Node, visit_count_policy
from alphazero_v3.temperature_audit import (
    defense_opportunity,
    immediate_capture_actions,
    immediate_winning_pass,
    network_state_digest,
    paired_game_seeds,
    play_temperature_audit_game,
    select_root_action,
    temperature_for_ply,
)
from great_kingdom_v2 import NUM_ACTIONS, PASS_ACTION


class PassNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        logits = torch.zeros(
            (inputs.shape[0], NUM_ACTIONS),
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        logits[:, PASS_ACTION] = 20.0 + self.anchor * 0.0
        values = torch.zeros(
            inputs.shape[0], dtype=inputs.dtype, device=inputs.device
        ) + self.anchor * 0.0
        return logits, values


def fake_checkpoint():
    return EvaluationCheckpoint(
        path=Path("iteration_000050.pt"),
        iteration=50,
        config={
            "channels": 8,
            "residual_blocks": 1,
            "mcts_simulations": 256,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_fraction": 0.25,
            "max_game_moves": 20,
        },
        network=PassNetwork(),
        device=torch.device("cpu"),
        state_encoder=encode_state,
    )


def synthetic_root():
    root = Node(1.0, 1)
    for action, visits in ((2, 2), (4, 5), (7, 5)):
        child = Node(1.0 / 3.0, 2)
        child.visit_count = visits
        root.children[action] = child
    return root


def test_temperature_schedules_and_boundaries():
    assert temperature_for_ply("all_hot", 0) == 1.0
    assert temperature_for_ply("all_hot", 200) == 1.0
    assert temperature_for_ply("early8", 7) == 1.0
    assert temperature_for_ply("early8", 8) == 0.0
    assert temperature_for_ply("greedy", 0) == 0.0
    assert temperature_for_ply("greedy", 200) == 0.0


def test_all_hot_reproduces_temperature_one_sampling():
    root = synthetic_root()
    expected_policy = visit_count_policy(root, temperature=1.0)
    expected_rng = np.random.default_rng(91)
    expected_action = int(
        expected_rng.choice(NUM_ACTIONS, p=expected_policy.astype(np.float64))
    )
    actual_action, actual_policy = select_root_action(
        root, 1.0, np.random.default_rng(91)
    )
    assert np.array_equal(actual_policy, expected_policy)
    assert actual_action == expected_action


def test_temperature_zero_selects_stable_max_visit_action():
    action, policy = select_root_action(
        synthetic_root(), 0.0, np.random.default_rng(1)
    )
    assert action == 4
    assert policy[4] == 1.0
    assert np.count_nonzero(policy) == 1


def test_rules_v2_tactical_opportunity_detectors():
    capture = capture_tactical_position()
    assert immediate_capture_actions(capture) == [19]

    defense = defense_opportunity(defense_tactical_position())
    assert defense["opponent_threat_actions"] == [19]
    assert defense["safe_defense_actions"] == [19]
    assert defense["is_opportunity"]

    winning_pass = winning_pass_tactical_position()
    assert immediate_winning_pass(winning_pass)


def test_game_has_no_illegal_action_and_does_not_change_network():
    checkpoint = fake_checkpoint()
    before = network_state_digest(checkpoint.network)
    game = play_temperature_audit_game(
        checkpoint,
        "greedy",
        game_index=0,
        game_seed=paired_game_seeds(20260902, 1)[0],
        simulations=4,
        c_puct=1.5,
    )
    after = network_state_digest(checkpoint.network)
    assert game["illegal_violations"] == 0
    assert game["terminal_reason"] == "PASS_SCORE_END"
    assert game["actions"] == [PASS_ACTION, PASS_ACTION]
    assert before == after


if __name__ == "__main__":
    test_temperature_schedules_and_boundaries()
    test_all_hot_reproduces_temperature_one_sampling()
    test_temperature_zero_selects_stable_max_visit_action()
    test_rules_v2_tactical_opportunity_detectors()
    test_game_has_no_illegal_action_and_does_not_change_network()
    print("AlphaZero temperature audit tests: PASS")
