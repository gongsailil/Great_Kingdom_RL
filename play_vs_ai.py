"""Play historical 81-action Rules V1 against a trained MaskablePPO policy.

Rules V1 checkpoints are not compatible with the 82-action Rules V2 engine.
"""

import argparse
from pathlib import Path

import numpy as np
import pygame
from sb3_contrib import MaskablePPO

from game_ui import (
    COLOR_BLUE_TOP,
    COLOR_RED_TOP,
    GreatKingdomRenderer,
    apply_move,
    impossible_move_message,
    new_game,
    player_name,
)
from gk_env import action_mask_for_logic, observation_for_player
from great_kingdom import BOARD_SIZE, MoveResult


DEFAULT_RED_MODEL = Path(
    "models/MaskablePPO_CNN/red14_ft_vs_blue13_plus10k.zip"
)
DEFAULT_BLUE_MODEL = Path(
    "models/MaskablePPO_CNN/blue13_ft_vs_red13_plus10k.zip"
)


def player_number(player):
    if player in ("blue", 1):
        return 1
    if player in ("red", 2):
        return 2
    raise ValueError("human player must be 'blue' or 'red'")


def default_model_for_human(human_player):
    if player_number(human_player) == 1:
        return DEFAULT_RED_MODEL
    return DEFAULT_BLUE_MODEL


def load_ai_model(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"AI model does not exist: {model_path}")
    print(f"Loading AI model: {model_path}", flush=True)
    return MaskablePPO.load(model_path, device="cpu")


def predict_masked_action(model, observation, action_mask):
    action, _ = model.predict(
        observation,
        action_masks=action_mask,
        deterministic=True,
    )
    action_array = np.asarray(action)
    if action_array.size != 1:
        raise RuntimeError(
            f"AI returned a non-scalar action with shape {action_array.shape}"
        )
    action_value = int(action_array.item())
    if (
        action_value < 0
        or action_value >= action_mask.size
        or not bool(action_mask[action_value])
    ):
        raise RuntimeError(f"AI selected masked action {action_value}")
    return action_value


class HumanVsAIController:
    """Testable turn controller; all rule decisions remain in GreatKingdomLogic."""

    def __init__(self, human_player, model):
        self.human_player = player_number(human_player)
        self.ai_player = 3 - self.human_player
        self.model = model
        self.logic = new_game()

    @property
    def is_human_turn(self):
        return not self.logic.game_over and self.logic.turn == self.human_player

    @property
    def is_ai_turn(self):
        return not self.logic.game_over and self.logic.turn == self.ai_player

    def restart(self):
        self.logic = new_game()

    def ai_observation(self):
        return observation_for_player(self.logic, self.ai_player)

    def action_mask(self):
        return action_mask_for_logic(self.logic)

    def play_human_move(self, x, y):
        if not self.is_human_turn:
            raise RuntimeError("human move requested when it is not the human turn")
        return apply_move(self.logic, x, y)

    def play_ai_move(self):
        if not self.is_ai_turn:
            raise RuntimeError("AI move requested when it is not the AI turn")

        mask = self.action_mask()
        if not np.any(mask):
            self.logic.check_game_end_simple()
            return None

        action = predict_masked_action(self.model, self.ai_observation(), mask)
        x = action % BOARD_SIZE
        y = action // BOARD_SIZE
        print(f"AI selected: ({x}, {y}) [action={action}]", flush=True)
        result = apply_move(self.logic, x, y)
        if result in (
            MoveResult.IMPOSSIBLE_OCCUPIED,
            MoveResult.IMPOSSIBLE_TERRITORY,
        ):
            raise RuntimeError(
                f"masked AI action became impossible at ({x}, {y}): {result.name}"
            )
        return x, y, result


class HumanVsAIUI:
    def __init__(self, human_player, model, renderer=None):
        self.controller = HumanVsAIController(human_player, model)
        self.renderer = renderer or GreatKingdomRenderer(
            "Great Kingdom - Human vs MaskablePPO"
        )
        self.info_message = self._start_message()

    def _start_message(self):
        human = player_name(self.controller.human_player)
        ai = player_name(self.controller.ai_player)
        if self.controller.ai_player == 1:
            return f"Human: {human}, AI: {ai}. AI opens."
        return f"Human: {human}, AI: {ai}. Human opens."

    def restart(self):
        self.controller.restart()
        self.info_message = self._start_message()

    def play_human_at(self, x, y):
        result = self.controller.play_human_move(x, y)
        if result in (
            MoveResult.IMPOSSIBLE_OCCUPIED,
            MoveResult.IMPOSSIBLE_TERRITORY,
        ):
            self.info_message = impossible_move_message(result)
        elif result == MoveResult.NORMAL and not self.controller.logic.game_over:
            self.info_message = "AI thinking..."
        return result

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.restart()
        elif (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.controller.is_human_turn
        ):
            coordinate = self.renderer.board_coordinate(event.pos)
            if coordinate is not None:
                self.play_human_at(*coordinate)
        return True

    def draw(self):
        logic = self.controller.logic
        if self.controller.is_ai_turn:
            headline = "AI thinking..."
            headline_color = (
                COLOR_BLUE_TOP if self.controller.ai_player == 1 else COLOR_RED_TOP
            )
            ghost_player = None
        elif self.controller.is_human_turn:
            headline = f"Your turn ({player_name(self.controller.human_player)})"
            headline_color = (
                COLOR_BLUE_TOP
                if self.controller.human_player == 1
                else COLOR_RED_TOP
            )
            ghost_player = self.controller.human_player
        else:
            headline = None
            headline_color = None
            ghost_player = None

        self.renderer.draw_frame(
            logic,
            self.info_message,
            headline=headline,
            headline_color=headline_color,
            ghost_player=ghost_player,
        )

    def run(self, max_frames=None):
        running = True
        frame_count = 0
        try:
            while running and (max_frames is None or frame_count < max_frames):
                self.renderer.clock.tick(30)
                for event in pygame.event.get():
                    running = self.handle_event(event)
                    if not running:
                        break

                # Present the thinking state before deterministic inference.
                self.draw()
                if running and self.controller.is_ai_turn:
                    pygame.event.pump()
                    move = self.controller.play_ai_move()
                    if move is None:
                        self.info_message = "No playable move; scored game over."
                    else:
                        x, y, result = move
                        if result == MoveResult.NORMAL:
                            self.info_message = f"AI played ({x}, {y}). Your turn."
                frame_count += 1
        finally:
            self.renderer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Play Great Kingdom against a stable MaskablePPO checkpoint."
    )
    parser.add_argument(
        "--human-player",
        choices=("blue", "red"),
        default="blue",
        help="human color (Blue moves first)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "MaskablePPO model path; defaults to Red14 for a Blue human "
            "or stable Blue13 for a Red human"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_path = args.model or default_model_for_human(args.human_player)
    model = load_ai_model(model_path)
    HumanVsAIUI(args.human_player, model).run()


if __name__ == "__main__":
    main()
