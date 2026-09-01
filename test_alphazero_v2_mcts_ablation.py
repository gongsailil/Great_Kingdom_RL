"""Lightweight tests for the fixed iteration-375 MCTS budget ablation."""

from pathlib import Path

import torch
from torch import nn

from alphazero_v2.evaluate import (
    EvaluationCheckpoint,
    analyze_safe_defense_actions,
    analyze_search_root,
    capture_tactical_position,
    defense_tactical_position,
    play_search_budget_game,
    winning_pass_tactical_position,
)
from great_kingdom_v2 import BLUE, NUM_ACTIONS, PASS_ACTION, MoveResultV2
from run_alphazero_v2_mcts_ablation import (
    DEFAULT_BUDGETS,
    interpret_results,
    parse_args,
)


class BiasedNetwork(nn.Module):
    def __init__(self, preferred_action):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.preferred_action = int(preferred_action)

    def forward(self, inputs):
        logits = torch.zeros(
            (inputs.shape[0], NUM_ACTIONS),
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        logits[:, self.preferred_action] = 20.0 + self.anchor * 0.0
        values = torch.zeros(
            inputs.shape[0], dtype=inputs.dtype, device=inputs.device
        ) + self.anchor * 0.0
        return logits, values


def fake_checkpoint(preferred_action):
    return EvaluationCheckpoint(
        path=Path("iteration_000375.pt"),
        iteration=375,
        config={"channels": 8, "residual_blocks": 1, "c_puct": 1.5},
        network=BiasedNetwork(preferred_action),
        device=torch.device("cpu"),
    )


def child_stats(diagnostic, action):
    return next(item for item in diagnostic["children"] if item["action"] == action)


def test_root_diagnostic_uses_root_player_q_perspective():
    capture_action = 1 + 2 * 9
    diagnostic = analyze_search_root(
        fake_checkpoint(capture_action),
        capture_tactical_position(),
        simulations=4,
    )
    stats = child_stats(diagnostic, capture_action)
    assert diagnostic["selected_action"] == capture_action
    assert stats["visit_count"] > 0
    assert stats["immediate_result"] == "CAPTURE_WIN"
    assert stats["q_value_root_player"] == 1.0


def test_safe_defense_actions_are_computed_by_opponent_capture_scan():
    analysis = analyze_safe_defense_actions(defense_tactical_position())
    assert analysis["player"] == BLUE
    assert analysis["original_immediate_threat_actions"] == [19]
    assert analysis["safe_defense_actions"] == [19]
    assert analysis["opponent_immediate_captures_after"]["19"] == []
    assert analysis["opponent_immediate_captures_after"][str(PASS_ACTION)] == [19]


def test_winning_pass_fixture_is_immediate_terminal_current_player_win():
    logic = winning_pass_tactical_position()
    original_player = logic.turn
    result = logic.apply_action(PASS_ACTION)
    assert result == MoveResultV2.PASS_SCORE_END
    assert logic.game_over
    assert logic.winner == original_player == BLUE


def test_same_checkpoint_different_budget_agent_identity():
    checkpoint = fake_checkpoint(PASS_ACTION)
    opening = {"opening_id": 0, "actions": [0, 80], "resulting_turn": BLUE}
    game = play_search_budget_game(checkpoint, 4, 8, opening)
    assert game["blue_iteration"] == game["red_iteration"] == 375
    assert game["blue_agent"] == "iter375_mcts4"
    assert game["red_agent"] == "iter375_mcts8"
    assert game["blue_mcts_simulations"] == 4
    assert game["red_mcts_simulations"] == 8
    assert game["winner_agent"] == game["red_agent"]
    assert game["terminal_reason"] == "PASS_SCORE_END"


def test_fixed_budget_parser():
    defaults = parse_args([])
    assert tuple(defaults.budgets) == DEFAULT_BUDGETS == (64, 128, 256, 512)
    explicit = parse_args(["--budgets", "64", "128", "256", "512"])
    assert tuple(explicit.budgets) == DEFAULT_BUDGETS


def test_unvisited_winning_pass_is_not_reported_as_backup_bug():
    tactical = {
        "results": [
            {
                "simulations": 512,
                "defense": {"success": True},
                "winning_pass": {
                    "success": False,
                    "pass_action_stats": {
                        "visit_count": 0,
                        "q_value_root_player": 0.0,
                    },
                },
            }
        ]
    }
    arena = [{"higher_budget_wins": 2, "baseline_wins": 1}]
    interpretation = interpret_results(tactical, arena)
    assert not interpretation["potential_mcts_bug"]
    assert "never visited" in interpretation["mcts_bug_diagnostic"]
    assert "not evaluated" in interpretation["mcts_bug_diagnostic"]


if __name__ == "__main__":
    test_root_diagnostic_uses_root_player_q_perspective()
    test_safe_defense_actions_are_computed_by_opponent_capture_scan()
    test_winning_pass_fixture_is_immediate_terminal_current_player_win()
    test_same_checkpoint_different_budget_agent_identity()
    test_fixed_budget_parser()
    test_unvisited_winning_pass_is_not_reported_as_backup_bug()
    print("AlphaZero V2 MCTS ablation tests: PASS")
