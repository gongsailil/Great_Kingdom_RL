"""Play Great Kingdom Rules V2 against an AlphaZero V2 checkpoint."""

import argparse
from pathlib import Path

import pygame

from alphazero_v2.evaluate import (
    load_evaluation_checkpoint,
    select_evaluation_action,
)
from game_ui import COLOR_BLUE_TOP, COLOR_RED_TOP, GreatKingdomRenderer
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
    player_name,
)
from play_human_v2 import IMPOSSIBLE_MESSAGES


DEFAULT_CHECKPOINT = Path("runs/alphazero_v2/main_20260830/latest.pt")


def player_number(player):
    if player in ("blue", BLUE):
        return BLUE
    if player in ("red", RED):
        return RED
    raise ValueError("human player must be 'blue' or 'red'")


class HumanVsAlphaZeroV2Controller:
    def __init__(
        self,
        human_player,
        checkpoint,
        mcts_simulations=64,
        action_selector=select_evaluation_action,
    ):
        if int(mcts_simulations) <= 0:
            raise ValueError("MCTS simulations must be positive")
        self.human_player = player_number(human_player)
        self.ai_player = 3 - self.human_player
        self.checkpoint = checkpoint
        self.mcts_simulations = int(mcts_simulations)
        self.action_selector = action_selector
        self.logic = GreatKingdomLogicV2()
        self.last_ai_action = None

    @property
    def is_human_turn(self):
        return not self.logic.game_over and self.logic.turn == self.human_player

    @property
    def is_ai_turn(self):
        return not self.logic.game_over and self.logic.turn == self.ai_player

    def restart(self):
        self.logic = GreatKingdomLogicV2()
        self.last_ai_action = None

    def play_human_move(self, x, y):
        if not self.is_human_turn:
            raise RuntimeError("human move requested outside the human turn")
        return self.logic.place_stone_detailed(x, y)

    def play_human_pass(self):
        if not self.is_human_turn:
            raise RuntimeError("human PASS requested outside the human turn")
        return self.logic.pass_turn()

    def play_ai_move(self):
        if not self.is_ai_turn:
            raise RuntimeError("AI move requested outside the AI turn")
        action = int(
            self.action_selector(
                self.checkpoint,
                self.logic,
                self.mcts_simulations,
            )
        )
        legal_mask = action_mask_for_logic(self.logic, self.ai_player)
        if not 0 <= action < legal_mask.size or not bool(legal_mask[action]):
            raise RuntimeError(f"AlphaZero selected illegal action {action}")
        result = self.logic.apply_action(action)
        if result not in (
            MoveResultV2.NORMAL,
            MoveResultV2.CAPTURE_WIN,
            MoveResultV2.PASS,
            MoveResultV2.PASS_SCORE_END,
        ):
            raise RuntimeError(f"AlphaZero action became illegal: {result.name}")
        self.last_ai_action = action
        if action == PASS_ACTION:
            print("AlphaZero selected: PASS [action=81]", flush=True)
        else:
            x, y = action % BOARD_SIZE, action // BOARD_SIZE
            print(f"AlphaZero selected: ({x}, {y}) [action={action}]", flush=True)
        return action, result


class HumanVsAlphaZeroV2UI:
    def __init__(
        self,
        human_player,
        checkpoint,
        mcts_simulations=64,
        renderer=None,
        action_selector=select_evaluation_action,
    ):
        self.controller = HumanVsAlphaZeroV2Controller(
            human_player,
            checkpoint,
            mcts_simulations,
            action_selector,
        )
        self.renderer = renderer or GreatKingdomRenderer(
            "Great Kingdom - Human vs AlphaZero V2"
        )
        self.info_message = self._start_message()

    def _identity_text(self):
        controller = self.controller
        return (
            f"Human {player_name(controller.human_player)} | "
            f"AI iter {controller.checkpoint.iteration} | "
            f"MCTS {controller.mcts_simulations}"
        )

    def _start_message(self):
        opener = "AI opens." if self.controller.ai_player == BLUE else "Human opens."
        return f"{self._identity_text()} | {opener} P = PASS"

    def _terminal_message(self, result):
        logic = self.controller.logic
        score = ""
        if logic.score_blue is not None:
            score = f" | territory B {logic.score_blue} R {logic.score_red}"
        return (
            f"Winner {player_name(logic.winner)} | {result.name}{score}"
        )

    def restart(self):
        self.controller.restart()
        self.info_message = self._start_message()

    def play_human_at(self, x, y):
        result = self.controller.play_human_move(x, y)
        if result in IMPOSSIBLE_MESSAGES:
            self.info_message = IMPOSSIBLE_MESSAGES[result]
        elif self.controller.logic.game_over:
            self.info_message = self._terminal_message(result)
        else:
            self.info_message = f"{self._identity_text()} | AI thinking..."
        return result

    def play_human_pass(self):
        result = self.controller.play_human_pass()
        if self.controller.logic.game_over:
            self.info_message = self._terminal_message(result)
        else:
            self.info_message = f"{self._identity_text()} | Human passed; AI thinking..."
        return result

    def _set_ai_result_message(self, action, result):
        if self.controller.logic.game_over:
            self.info_message = self._terminal_message(result)
        elif action == PASS_ACTION:
            self.info_message = f"{self._identity_text()} | AI last: PASS | Your turn"
        else:
            x, y = action % BOARD_SIZE, action // BOARD_SIZE
            self.info_message = (
                f"{self._identity_text()} | AI last: ({x}, {y}) | Your turn"
            )

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.restart()
            elif event.key == pygame.K_p and self.controller.is_human_turn:
                self.play_human_pass()
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
        if logic.game_over:
            headline = None
            headline_color = None
            ghost_player = None
        else:
            role = "AI" if self.controller.is_ai_turn else "Human"
            headline = (
                f"{role} {player_name(logic.turn)} turn | "
                f"B {logic.castles_remaining[BLUE]} "
                f"R {logic.castles_remaining[RED]} | "
                f"passes {logic.consecutive_passes}"
            )
            headline_color = COLOR_BLUE_TOP if logic.turn == BLUE else COLOR_RED_TOP
            ghost_player = self.controller.human_player if self.controller.is_human_turn else None
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
                self.draw()
                if running and self.controller.is_ai_turn:
                    pygame.event.pump()
                    action, result = self.controller.play_ai_move()
                    self._set_ai_result_message(action, result)
                frame_count += 1
        finally:
            self.renderer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--human-player",
        choices=("blue", "red"),
        default="blue",
        help="human color (Blue moves first)",
    )
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.mcts_simulations <= 0:
        parser.error("--mcts-simulations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = load_evaluation_checkpoint(args.checkpoint, args.device)
    HumanVsAlphaZeroV2UI(
        args.human_player,
        checkpoint,
        args.mcts_simulations,
    ).run()


if __name__ == "__main__":
    main()
