import os
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

import play_human
import play_vs_ai
from game_ui import GreatKingdomRenderer
from great_kingdom import GreatKingdomLogic, MoveResult


class ScriptedModel:
    def __init__(self, action):
        self.action = action
        self.calls = []

    def predict(self, observation, action_masks, deterministic):
        self.calls.append(
            {
                "observation": np.array(observation, copy=True),
                "action_masks": np.array(action_masks, copy=True),
                "deterministic": deterministic,
            }
        )
        return np.asarray(self.action), None


def expect_exception(exception_type, function):
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_ui_imports():
    assert play_human.HumanVsHumanUI
    assert play_vs_ai.HumanVsAIUI


def test_stable_default_checkpoints():
    assert str(play_vs_ai.default_model_for_human("blue")).endswith(
        "red14_ft_vs_blue13_plus10k.zip"
    )
    assert str(play_vs_ai.default_model_for_human("red")).endswith(
        "blue13_ft_vs_red13_plus10k.zip"
    )
    assert "blue14" not in str(play_vs_ai.DEFAULT_BLUE_MODEL).lower()


def test_blue_ai_observation_perspective():
    controller = play_vs_ai.HumanVsAIController("red", ScriptedModel(0))
    controller.logic.board[0][0] = 1
    controller.logic.board[0][1] = 2
    observation = controller.ai_observation()
    assert observation[:, 0, 0].tolist() == [1, 0, 0]
    assert observation[:, 0, 1].tolist() == [0, 1, 0]
    assert observation[:, 4, 4].tolist() == [0, 0, 1]


def test_red_ai_observation_perspective():
    controller = play_vs_ai.HumanVsAIController("blue", ScriptedModel(1))
    controller.logic.board[0][0] = 1
    controller.logic.board[0][1] = 2
    observation = controller.ai_observation()
    assert observation[:, 0, 0].tolist() == [0, 1, 0]
    assert observation[:, 0, 1].tolist() == [1, 0, 0]
    assert observation[:, 4, 4].tolist() == [0, 0, 1]


def test_action_mask_is_passed_to_deterministic_prediction():
    model = ScriptedModel(1)
    controller = play_vs_ai.HumanVsAIController("blue", model)
    assert controller.play_human_move(0, 0) == MoveResult.NORMAL
    controller.play_ai_move()

    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["deterministic"] is True
    assert call["action_masks"].dtype == bool
    assert not call["action_masks"][0]
    assert not call["action_masks"][4 + 4 * 9]
    assert call["action_masks"][1]


def test_masked_ai_action_raises_without_fallback():
    controller = play_vs_ai.HumanVsAIController("red", ScriptedModel(4 + 4 * 9))
    expect_exception(RuntimeError, controller.play_ai_move)
    assert sum(cell == 1 for row in controller.logic.board for cell in row) == 0


def test_nonexistent_model_fails_fast():
    with tempfile.TemporaryDirectory() as temp_dir_name:
        missing = Path(temp_dir_name) / "missing.zip"
        expect_exception(FileNotFoundError, lambda: play_vs_ai.load_ai_model(missing))


def test_blue_ai_moves_first():
    model = ScriptedModel(0)
    controller = play_vs_ai.HumanVsAIController("red", model)
    assert controller.ai_player == 1
    assert controller.is_ai_turn
    x, y, result = controller.play_ai_move()
    assert (x, y, result) == (0, 0, MoveResult.NORMAL)
    assert controller.logic.board[0][0] == 1
    assert controller.is_human_turn


def test_red_ai_waits_for_human_blue():
    controller = play_vs_ai.HumanVsAIController("blue", ScriptedModel(1))
    assert controller.ai_player == 2
    assert controller.is_human_turn
    expect_exception(RuntimeError, controller.play_ai_move)
    assert controller.play_human_move(0, 0) == MoveResult.NORMAL
    assert controller.is_ai_turn


def test_human_impossible_move_does_not_advance_turn():
    game = object.__new__(play_human.HumanVsHumanUI)
    game.renderer = None
    game.logic = GreatKingdomLogic()
    game.info_message = ""
    assert game.play_at(4, 4) == MoveResult.IMPOSSIBLE_OCCUPIED
    assert game.logic.turn == 1
    assert "occupied" in game.info_message


def test_human_territory_move_does_not_advance_turn():
    game = object.__new__(play_human.HumanVsHumanUI)
    game.renderer = None
    game.logic = GreatKingdomLogic()
    game.logic.board = [[3] * 9 for _ in range(9)]
    game.logic.board[0][0] = 2
    game.logic.board[4][4] = 0
    game.logic.board[3][4] = 1
    game.logic.board[4][3] = 1
    game.logic.board[4][5] = 1
    game.logic.board[5][4] = 1
    game.info_message = ""

    assert game.logic.get_territory_owner(4, 4) == 1
    assert game.play_at(4, 4) == MoveResult.IMPOSSIBLE_TERRITORY
    assert game.logic.turn == 1
    assert "territory" in game.info_message


def test_dummy_video_draw_and_shutdown():
    renderer = GreatKingdomRenderer("Great Kingdom UI smoke")
    logic = GreatKingdomLogic()
    renderer.draw_frame(
        logic,
        "Smoke test",
        ghost_player=1,
        mouse_position=(45, 45),
    )
    renderer.close()


if __name__ == "__main__":
    test_ui_imports()
    test_stable_default_checkpoints()
    test_blue_ai_observation_perspective()
    test_red_ai_observation_perspective()
    test_action_mask_is_passed_to_deterministic_prediction()
    test_masked_ai_action_raises_without_fallback()
    test_nonexistent_model_fails_fast()
    test_blue_ai_moves_first()
    test_red_ai_waits_for_human_blue()
    test_human_impossible_move_does_not_advance_turn()
    test_human_territory_move_does_not_advance_turn()
    test_dummy_video_draw_and_shutdown()
    print("interactive UI tests: PASS")
