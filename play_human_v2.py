"""Human-vs-human UI for Great Kingdom Rules V2."""

import pygame

from game_ui import (
    COLOR_BLUE_TOP,
    COLOR_RED_TOP,
    GreatKingdomRenderer,
)
from great_kingdom_v2 import (
    BLUE,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
    player_name,
)


IMPOSSIBLE_MESSAGES = {
    MoveResultV2.IMPOSSIBLE_OCCUPIED: "Cannot play: occupied square.",
    MoveResultV2.IMPOSSIBLE_OPPONENT_TERRITORY: (
        "Cannot play inside the opponent's territory."
    ),
    MoveResultV2.IMPOSSIBLE_SUICIDE: "Cannot play: pure suicide is illegal.",
    MoveResultV2.IMPOSSIBLE_NO_CASTLES: "No castles remain; press P to pass.",
    MoveResultV2.IMPOSSIBLE_OUT_OF_BOUNDS: "Cannot play outside the board.",
}


class HumanVsHumanV2UI:
    def __init__(self, renderer=None):
        self.renderer = renderer or GreatKingdomRenderer(
            "Great Kingdom - Human vs Human - Rules V2"
        )
        self.logic = GreatKingdomLogicV2()
        self.info_message = "Blue opens. Click to place; P = PASS."

    def restart(self):
        self.logic = GreatKingdomLogicV2()
        self.info_message = "Game restarted. Blue opens; P = PASS."

    def play_at(self, x, y):
        player = self.logic.turn
        result = self.logic.place_stone_detailed(x, y)
        if result in IMPOSSIBLE_MESSAGES:
            self.info_message = IMPOSSIBLE_MESSAGES[result]
        elif result == MoveResultV2.NORMAL:
            self.info_message = f"{player_name(player)} placed ({x}, {y}). P = PASS."
        return result

    def pass_current_turn(self):
        player = self.logic.turn
        result = self.logic.pass_turn()
        if result == MoveResultV2.PASS:
            self.info_message = f"{player_name(player)} passed. P = PASS."
        return result

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.restart()
            elif event.key == pygame.K_p and not self.logic.game_over:
                self.pass_current_turn()
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
        if self.logic.game_over:
            headline = None
            color = None
            ghost_player = None
        else:
            headline = (
                f"{player_name(self.logic.turn)} turn | "
                f"Blue {self.logic.castles_remaining[BLUE]} | "
                f"Red {self.logic.castles_remaining[RED]} | "
                f"passes {self.logic.consecutive_passes}"
            )
            color = COLOR_BLUE_TOP if self.logic.turn == BLUE else COLOR_RED_TOP
            ghost_player = self.logic.turn
        self.renderer.draw_frame(
            self.logic,
            self.info_message,
            headline=headline,
            headline_color=color,
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
    HumanVsHumanV2UI().run()


if __name__ == "__main__":
    main()
