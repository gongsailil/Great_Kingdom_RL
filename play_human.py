"""Interactive human-vs-human Great Kingdom game using the current rules."""

import pygame

from game_ui import (
    GreatKingdomRenderer,
    apply_move,
    impossible_move_message,
    new_game,
    player_name,
)
from great_kingdom import MoveResult


class HumanVsHumanUI:
    def __init__(self, renderer=None):
        self.renderer = renderer or GreatKingdomRenderer(
            "Great Kingdom - Human vs Human"
        )
        self.logic = new_game()
        self.info_message = "Game start: Blue plays first."

    def restart(self):
        self.logic = new_game()
        self.info_message = "Game restarted: Blue plays first."

    def play_at(self, x, y):
        moving_player = self.logic.turn
        result = apply_move(self.logic, x, y)
        if result in (
            MoveResult.IMPOSSIBLE_OCCUPIED,
            MoveResult.IMPOSSIBLE_TERRITORY,
        ):
            self.info_message = impossible_move_message(result)
        elif result == MoveResult.NORMAL and not self.logic.game_over:
            self.info_message = (
                f"{player_name(moving_player)} played ({x}, {y}). "
                f"{player_name(self.logic.turn)} to move."
            )
        return result

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.restart()
        elif (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and not self.logic.game_over
        ):
            coordinate = self.renderer.board_coordinate(event.pos)
            if coordinate is not None:
                self.play_at(*coordinate)
        return True

    def draw(self):
        ghost_player = None if self.logic.game_over else self.logic.turn
        self.renderer.draw_frame(
            self.logic,
            self.info_message,
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
                frame_count += 1
        finally:
            self.renderer.close()


def main():
    HumanVsHumanUI().run()


if __name__ == "__main__":
    main()
