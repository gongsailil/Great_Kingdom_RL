"""Tests for policy/value-separated diagnostic MCTS."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from alphazero_v2.config import AlphaZeroConfig
from alphazero_v2.encoder import encode_state as encode_v2_state
from alphazero_v2.evaluate import EvaluationCheckpoint, winning_pass_tactical_position
from alphazero_v2.mcts import MCTS
from alphazero_v3.search_guidance_audit import (
    SEARCH_GUIDANCE_MODES,
    GuidanceAuditMCTS,
    analyze_guidance_mode,
    policy_statistics,
    uniform_legal_policy,
    value_from_player_to_root,
)
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import BLUE, NUM_ACTIONS, PASS_ACTION, RED


class FixedNetwork(nn.Module):
    def __init__(self, preferred_action=PASS_ACTION, value=0.25):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.preferred_action = int(preferred_action)
        self.fixed_value = float(value)

    def forward(self, inputs):
        logits = torch.zeros(
            (inputs.shape[0], NUM_ACTIONS),
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        logits[:, self.preferred_action] = 20.0 + self.anchor * 0.0
        values = torch.full(
            (inputs.shape[0],),
            self.fixed_value,
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        return logits, values


def fake_checkpoint(network):
    return EvaluationCheckpoint(
        path=Path("iteration_000001.pt"),
        iteration=1,
        config={"channels": 8, "residual_blocks": 1, "c_puct": 1.5},
        network=network,
        device=torch.device("cpu"),
        state_encoder=encode_v2_state,
    )


def test_uniform_policy_normalizes_only_legal_actions():
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[2, 17, PASS_ACTION]] = True
    priors = uniform_legal_policy(mask)
    assert np.isclose(priors.sum(), 1.0)
    assert np.allclose(priors[mask], 1.0 / 3.0)
    assert np.count_nonzero(priors[~mask]) == 0


def test_zero_value_does_not_replace_exact_terminal_winner_value():
    checkpoint = fake_checkpoint(FixedNetwork(PASS_ACTION, value=-0.9))
    zero_value_mode = SEARCH_GUIDANCE_MODES[2]
    diagnostic = analyze_guidance_mode(
        checkpoint,
        winning_pass_tactical_position(),
        PASS_ACTION,
        zero_value_mode,
        simulations=2,
    )
    expected = diagnostic["expected_action_stats"]
    assert diagnostic["success"]
    assert expected["visit_count"] > 0
    assert expected["q_value_root_player"] == 1.0
    assert diagnostic["expected_child_value"][
        "exact_terminal_value_root_player"
    ] == 1.0


def test_learned_learned_diagnostic_matches_production_mcts():
    logic = winning_pass_tactical_position()
    network = FixedNetwork(PASS_ACTION, value=0.25)
    config = AlphaZeroConfig(
        mcts_simulations=8,
        c_puct=1.5,
        dirichlet_fraction=0.0,
        temperature=0.0,
    )
    production = MCTS(network, config, "cpu").run(logic)
    diagnostic = GuidanceAuditMCTS(
        network,
        config,
        "cpu",
        state_encoder=encode_v2_state,
        mode=SEARCH_GUIDANCE_MODES[0],
    ).run(logic)
    assert set(production.children) == set(diagnostic.children)
    for action in production.children:
        left = production.children[action]
        right = diagnostic.children[action]
        assert left.prior == right.prior
        assert left.visit_count == right.visit_count
        assert left.value_sum == right.value_sum


def test_policy_rank_uses_legal_softmax_and_action_tie_break():
    logits = np.zeros(NUM_ACTIONS, dtype=np.float32)
    logits[3] = 4.0
    logits[7] = 2.0
    logits[11] = 2.0
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[3, 7, 11, PASS_ACTION]] = True
    stats = policy_statistics(logits, mask, 11)
    assert stats["expected_action_legal_rank"] == 3
    assert stats["legal_action_count"] == 4
    assert 0.0 < stats["expected_action_legal_probability"] < 1.0
    assert stats["legal_policy_entropy"] > 0.0


def test_value_sign_is_normalized_to_root_perspective():
    assert value_from_player_to_root(0.4, BLUE, BLUE) == 0.4
    assert value_from_player_to_root(0.4, RED, BLUE) == -0.4
    assert value_from_player_to_root(-0.7, BLUE, RED) == 0.7


def test_fixture_mask_still_comes_from_rules_v2():
    logic = winning_pass_tactical_position()
    mask = action_mask_for_logic(logic, logic.turn)
    priors = uniform_legal_policy(mask)
    assert priors[PASS_ACTION] > 0.0
    assert np.all(priors[~mask] == 0.0)


if __name__ == "__main__":
    test_uniform_policy_normalizes_only_legal_actions()
    test_zero_value_does_not_replace_exact_terminal_winner_value()
    test_learned_learned_diagnostic_matches_production_mcts()
    test_policy_rank_uses_legal_softmax_and_action_tie_break()
    test_value_sign_is_normalized_to_root_perspective()
    test_fixture_mask_still_comes_from_rules_v2()
    print("AlphaZero search-guidance audit tests: PASS")
