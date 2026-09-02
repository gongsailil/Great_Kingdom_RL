"""Tests for the fixed-network c_puct ablation."""

from pathlib import Path

import torch
from torch import nn

from alphazero_v2.encoder import encode_state
from alphazero_v2.evaluate import (
    EvaluationCheckpoint,
    winning_pass_tactical_position,
)
from alphazero_v3.puct_ablation import (
    DEFAULT_C_PUCT_VALUES,
    aggregate_puct_matchup,
    checkpoint_with_c_puct,
    classify_puct_sweep,
    play_puct_arena_game,
    puct_agent_id,
    select_arena_candidates,
)
from alphazero_v3.search_guidance_audit import (
    SEARCH_GUIDANCE_MODES,
    analyze_guidance_mode,
)
from great_kingdom_v2 import NUM_ACTIONS, PASS_ACTION
from run_alphazero_puct_ablation import parse_args


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
        config={"channels": 8, "residual_blocks": 1, "c_puct": 1.5},
        network=PassNetwork(),
        device=torch.device("cpu"),
        state_encoder=encode_state,
    )


def synthetic_tactical(success_rows):
    records = []
    fixtures = (
        "immediate_capture",
        "immediate_capture_threat_defense",
        "immediate_winning_pass",
    )
    for c_puct, successes in zip(DEFAULT_C_PUCT_VALUES, success_rows):
        for fixture, success in zip(fixtures, successes):
            records.append(
                {"c_puct": c_puct, "fixture": fixture, "success": success}
            )
    return {
        "c_puct_values": list(DEFAULT_C_PUCT_VALUES),
        "records": records,
    }


def test_fixed_parser_and_c_puct_checkpoint_copy():
    args = parse_args([])
    assert tuple(args.c_puct) == DEFAULT_C_PUCT_VALUES
    assert args.simulations == 256
    original = fake_checkpoint()
    changed = checkpoint_with_c_puct(original, 6.0)
    assert original.config["c_puct"] == 1.5
    assert changed.config["c_puct"] == 6.0
    assert changed.network is original.network


def test_candidate_selection_is_improvement_only_and_at_most_two():
    tactical = synthetic_tactical(
        [
            (False, False, False),
            (True, False, True),
            (True, False, True),
            (True, True, True),
        ]
    )
    assert select_arena_candidates(tactical) == [3.0, 12.0]
    assert classify_puct_sweep(tactical) == "CASE_PUCT"


def test_terminal_discovery_is_a_meaningful_fallback_candidate():
    tactical = synthetic_tactical(
        [
            (False, False, False),
            (False, False, False),
            (False, False, False),
            (False, False, False),
        ]
    )
    for record in tactical["records"]:
        visits = 0
        if record["c_puct"] == 12.0 and record["fixture"] in (
            "immediate_capture",
            "immediate_winning_pass",
        ):
            visits = 1
        record["expected_action_stats"] = {"visit_count": visits}
    assert select_arena_candidates(tactical) == [12.0]


def test_requested_case_classifications():
    partial = synthetic_tactical(
        [
            (False, False, False),
            (True, False, True),
            (True, False, True),
            (True, False, True),
        ]
    )
    value = synthetic_tactical(
        [
            (False, False, False),
            (False, False, False),
            (True, False, False),
            (True, False, False),
        ]
    )
    overexplore = synthetic_tactical(
        [
            (False, False, False),
            (True, True, True),
            (True, False, True),
            (True, False, True),
        ]
    )
    assert classify_puct_sweep(partial) == "CASE_PARTIAL"
    assert classify_puct_sweep(value) == "CASE_VALUE"
    assert classify_puct_sweep(overexplore) == "CASE_OVEREXPLORE"


def test_c_puct_override_records_root_coverage():
    diagnostic = analyze_guidance_mode(
        fake_checkpoint(),
        winning_pass_tactical_position(),
        PASS_ACTION,
        SEARCH_GUIDANCE_MODES[0],
        simulations=4,
        c_puct=6.0,
    )
    assert diagnostic["c_puct"] == 6.0
    assert diagnostic["legal_child_count"] > 0
    assert 0 < diagnostic["visited_child_count"] <= 4
    assert diagnostic["visited_child_fraction"] == (
        diagnostic["visited_child_count"] / diagnostic["legal_child_count"]
    )
    assert diagnostic["visit_entropy"] >= 0.0


def test_paired_arena_identity_and_aggregation():
    checkpoint = fake_checkpoint()
    opening = {"opening_id": 0, "actions": [0, 80], "resulting_turn": 1}
    first = play_puct_arena_game(checkpoint, 1.5, 3.0, opening, simulations=2)
    second = play_puct_arena_game(checkpoint, 3.0, 1.5, opening, simulations=2)
    assert first["blue_agent"] == puct_agent_id(checkpoint, 1.5)
    assert first["red_agent"] == puct_agent_id(checkpoint, 3.0)
    assert first["blue_c_puct"] == 1.5
    assert first["red_c_puct"] == 3.0
    result = aggregate_puct_matchup([first, second], 50, 1.5, 3.0)
    assert result["games"] == 2
    assert result["baseline_wins"] + result["candidate_wins"] == 2
    assert result["capture_endings"] == 0
    assert result["pass_score_endings"] == 2


if __name__ == "__main__":
    test_fixed_parser_and_c_puct_checkpoint_copy()
    test_candidate_selection_is_improvement_only_and_at_most_two()
    test_terminal_discovery_is_a_meaningful_fallback_candidate()
    test_requested_case_classifications()
    test_c_puct_override_records_root_coverage()
    test_paired_arena_identity_and_aggregation()
    print("AlphaZero c_puct ablation tests: PASS")
