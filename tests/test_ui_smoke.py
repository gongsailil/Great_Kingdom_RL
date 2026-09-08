import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import PASS_ACTION
from play_vs_alphazero_v5 import (
    HumanVsAlphaZeroV5Controller, HumanVsAlphaZeroV5UI,
)


class FakeCheckpoint:
    path = Path(__file__)
    cycle = 31
    best_version = 20
    calibration_temperature = 1.0


def first_legal(checkpoint, logic, simulations, top_k=5):
    action = int(np.flatnonzero(action_mask_for_logic(logic, logic.turn))[0])
    return {
        "action": action, "tactical_mode": "NORMAL", "top_actions": [],
        "mcts_executed": True, "safe_defense_actions": [],
        "unsafe_actions_filtered": 0,
    }


def choose_pass(checkpoint, logic, simulations, top_k=5):
    return {
        "action": PASS_ACTION, "tactical_mode": "NORMAL", "top_actions": [],
        "mcts_executed": True, "safe_defense_actions": [],
        "unsafe_actions_filtered": 0,
    }


class TestUISmoke(unittest.TestCase):
    def test_human_blue_draw_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            ui = HumanVsAlphaZeroV5UI(
                "blue", FakeCheckpoint(), 2, directory, selector=first_legal
            )
            self.assertTrue(ui.controller.is_human_turn)
            ui.run(max_frames=1)

    def test_human_red_ai_first(self):
        with tempfile.TemporaryDirectory() as directory:
            ui = HumanVsAlphaZeroV5UI(
                "red", FakeCheckpoint(), 2, directory, selector=first_legal
            )
            self.assertTrue(ui.controller.is_ai_turn)
            ui.run(max_frames=1)
            self.assertEqual(ui.controller.last_ai_action, 0)
            self.assertTrue(ui.controller.is_human_turn)

    def test_pass_scoring_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HumanVsAlphaZeroV5Controller(
                "blue", FakeCheckpoint(), 2, directory, choose_pass
            )
            controller.play_human_pass()
            self.assertTrue(controller.is_ai_turn)
            action, result = controller.play_ai_move()
            self.assertEqual(action, PASS_ACTION)
            self.assertTrue(controller.logic.game_over)
            self.assertEqual(result.name, "PASS_SCORE_END")
            payload = json.loads(controller.log_path.read_text())
            self.assertEqual(payload["terminal_reason"], "PASS_SCORE_END")
            self.assertEqual(len(payload["moves"]), 2)


if __name__ == "__main__":
    unittest.main()
