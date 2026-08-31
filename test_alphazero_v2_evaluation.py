"""Lightweight tests for AlphaZero V2 arena evaluation and human UI."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
from torch import nn

from alphazero_v2.evaluate import (
    EvaluationCheckpoint,
    apply_opening,
    generate_opening_suite,
    load_evaluation_checkpoint,
    play_paired_opening,
    select_evaluation_action,
    validate_opening_suite,
)
from alphazero_v2.network import PolicyValueNetwork
from game_ui import GreatKingdomRenderer
from great_kingdom_v2 import (
    BLUE,
    NUM_ACTIONS,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
)
from play_vs_alphazero_v2 import (
    HumanVsAlphaZeroV2Controller,
    HumanVsAlphaZeroV2UI,
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


def fake_checkpoint(iteration, preferred_action=PASS_ACTION):
    return EvaluationCheckpoint(
        path=Path(f"iteration_{iteration:06d}.pt"),
        iteration=iteration,
        config={
            "channels": 8,
            "residual_blocks": 1,
            "c_puct": 1.5,
        },
        network=BiasedNetwork(preferred_action),
        device=torch.device("cpu"),
    )


def normal_opening():
    return {"opening_id": 0, "actions": [0, 80], "resulting_turn": BLUE}


def test_evaluation_checkpoint_loads_architecture_and_iteration():
    network = PolicyValueNetwork(channels=8, residual_blocks=1)
    payload = {
        "config": {"channels": 8, "residual_blocks": 1, "c_puct": 1.5},
        "network_state_dict": network.state_dict(),
        "iteration": 7,
        "optimizer_state_dict": {"not": "needed"},
    }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        path = Path(temp_dir_name) / "iteration_000007.pt"
        torch.save(payload, path)
        loaded = load_evaluation_checkpoint(path, "cpu", expected_iteration=7)
        assert loaded.iteration == 7
        assert loaded.network.channels == 8
        assert loaded.network.residual_blocks == 1
        assert not loaded.network.training


def test_opening_suite_is_deterministic_and_legal():
    first = generate_opening_suite(count=10, seed=20260901)
    second = generate_opening_suite(count=10, seed=20260901)
    assert first == second
    assert validate_opening_suite(first)
    assert len(first["openings"]) == 10
    for opening in first["openings"]:
        logic = apply_opening(opening["actions"])
        assert not logic.game_over
        assert logic.turn == opening["resulting_turn"]
        assert 2 <= len(opening["actions"]) <= 4
        assert PASS_ACTION not in opening["actions"]


def test_deterministic_action_and_illegal_action_exclusion():
    pass_checkpoint = fake_checkpoint(1, PASS_ACTION)
    logic = apply_opening(normal_opening()["actions"])
    first = select_evaluation_action(pass_checkpoint, logic.copy(), simulations=4)
    second = select_evaluation_action(pass_checkpoint, logic.copy(), simulations=4)
    assert first == second == PASS_ACTION

    occupied_preference = fake_checkpoint(2, 4 + 4 * 9)
    selected = select_evaluation_action(
        occupied_preference,
        GreatKingdomLogicV2(),
        simulations=4,
    )
    assert selected != 4 + 4 * 9


def test_paired_color_swap_and_same_openings_reused():
    opening = normal_opening()
    checkpoint_a = fake_checkpoint(10)
    checkpoint_b = fake_checkpoint(20)
    checkpoint_c = fake_checkpoint(30)
    first_matchup = play_paired_opening(
        checkpoint_a, checkpoint_b, opening, simulations=4
    )
    second_matchup = play_paired_opening(
        checkpoint_a, checkpoint_c, opening, simulations=4
    )
    assert first_matchup[0]["blue_iteration"] == 10
    assert first_matchup[0]["red_iteration"] == 20
    assert first_matchup[1]["blue_iteration"] == 20
    assert first_matchup[1]["red_iteration"] == 10
    assert all(game["opening_actions"] == opening["actions"] for game in first_matchup)
    assert all(game["opening_actions"] == opening["actions"] for game in second_matchup)
    assert all(game["terminal_reason"] == "PASS_SCORE_END" for game in first_matchup)


def scripted_pass(checkpoint, logic, simulations):
    return PASS_ACTION


def test_human_blue_red_initialization_ai_first_and_pass():
    checkpoint = SimpleNamespace(iteration=375)
    human_blue = HumanVsAlphaZeroV2Controller(
        "blue", checkpoint, action_selector=scripted_pass
    )
    assert human_blue.human_player == BLUE
    assert human_blue.is_human_turn

    human_red = HumanVsAlphaZeroV2Controller(
        "red", checkpoint, action_selector=scripted_pass
    )
    assert human_red.human_player == RED
    assert human_red.is_ai_turn
    action, result = human_red.play_ai_move()
    assert action == PASS_ACTION
    assert result == MoveResultV2.PASS
    assert human_red.logic.consecutive_passes == 1
    assert human_red.is_human_turn


def test_ai_pass_ui_message_and_dummy_draw_shutdown():
    checkpoint = SimpleNamespace(iteration=375)
    renderer = GreatKingdomRenderer("AlphaZero V2 UI smoke")
    ui = HumanVsAlphaZeroV2UI(
        "red",
        checkpoint,
        mcts_simulations=64,
        renderer=renderer,
        action_selector=scripted_pass,
    )
    ui.run(max_frames=1)
    assert ui.controller.last_ai_action == PASS_ACTION
    assert "AI last: PASS" in ui.info_message


if __name__ == "__main__":
    test_evaluation_checkpoint_loads_architecture_and_iteration()
    test_opening_suite_is_deterministic_and_legal()
    test_deterministic_action_and_illegal_action_exclusion()
    test_paired_color_swap_and_same_openings_reused()
    test_human_blue_red_initialization_ai_first_and_pass()
    test_ai_pass_ui_message_and_dummy_draw_shutdown()
    print("AlphaZero V2 evaluation/UI tests: PASS")
