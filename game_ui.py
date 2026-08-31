"""Small shared Pygame renderer for the interactive Great Kingdom games."""

import pygame

from great_kingdom import (
    BOARD_SIZE,
    GRID_SIZE,
    MARGIN,
    GreatKingdomLogic,
    MoveResult,
)


SCREEN_WIDTH = 720
SCREEN_HEIGHT = 780

COLOR_BOARD_BASE = (245, 245, 245)
COLOR_BOARD_INSET = (255, 255, 255)
COLOR_GRID_SHADOW = (210, 210, 210)
COLOR_BLUE_BASE = (0, 100, 230, 180)
COLOR_BLUE_TOP = (50, 150, 255, 220)
COLOR_RED_BASE = (220, 30, 30, 180)
COLOR_RED_TOP = (255, 80, 80, 220)
COLOR_NEUTRAL_BASE = (220, 220, 220, 255)
COLOR_NEUTRAL_TOP = (255, 255, 255, 255)
COLOR_TEXT = (50, 50, 50)
COLOR_HIGHLIGHT = (220, 50, 50)


def player_name(player):
    if player == 1:
        return "Blue"
    if player == 2:
        return "Red"
    return "Draw"


def apply_move(logic, x, y):
    """Apply one move through the current rule API and run score termination."""
    result = logic.place_stone_detailed(x, y)
    if result == MoveResult.NORMAL:
        logic.check_game_end_simple()
    return result


def impossible_move_message(result):
    if result == MoveResult.IMPOSSIBLE_OCCUPIED:
        return "Cannot play: that square is occupied."
    if result == MoveResult.IMPOSSIBLE_TERRITORY:
        return "Cannot play: that square is established territory."
    raise ValueError(f"not an impossible move result: {result}")


class GreatKingdomRenderer:
    """Legacy-style board renderer with no ownership of game rules."""

    def __init__(self, caption):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("malgungothic", 18, True)
            self.large_font = pygame.font.SysFont("malgungothic", 26, True)
        except Exception:
            self.font = pygame.font.SysFont("arial", 18, True)
            self.large_font = pygame.font.SysFont("arial", 26, True)

    @staticmethod
    def board_coordinate(position):
        mouse_x, mouse_y = position
        board_end_x = MARGIN + BOARD_SIZE * GRID_SIZE
        board_end_y = MARGIN + BOARD_SIZE * GRID_SIZE
        if not (MARGIN <= mouse_x < board_end_x):
            return None
        if not (MARGIN <= mouse_y < board_end_y):
            return None
        return (
            (mouse_x - MARGIN) // GRID_SIZE,
            (mouse_y - MARGIN) // GRID_SIZE,
        )

    def draw_board(self, logic):
        self.screen.fill(COLOR_BOARD_BASE)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cell_x = MARGIN + x * GRID_SIZE
                cell_y = MARGIN + y * GRID_SIZE
                pygame.draw.rect(
                    self.screen,
                    COLOR_GRID_SHADOW,
                    (cell_x + 2, cell_y + 2, GRID_SIZE - 4, GRID_SIZE - 4),
                )
                pygame.draw.rect(
                    self.screen,
                    COLOR_BOARD_INSET,
                    (cell_x + 5, cell_y + 5, GRID_SIZE - 10, GRID_SIZE - 10),
                )
                state = logic.board[y][x]
                if state:
                    self.draw_castle(x, y, state)

    def draw_castle(self, x, y, state, ghost=False):
        if state == 1:
            base_color, top_color = COLOR_BLUE_BASE, COLOR_BLUE_TOP
        elif state == 2:
            base_color, top_color = COLOR_RED_BASE, COLOR_RED_TOP
        elif state == 3:
            base_color, top_color = COLOR_NEUTRAL_BASE, COLOR_NEUTRAL_TOP
        else:
            return

        if ghost:
            base_color = (*base_color[:3], 100)
            top_color = (*top_color[:3], 120)

        castle = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
        base_size = GRID_SIZE * 0.8
        base_offset = (GRID_SIZE - base_size) / 2
        pygame.draw.rect(
            castle,
            base_color,
            pygame.Rect(base_offset, base_offset, base_size, base_size),
            border_radius=4,
        )

        top_size = base_size * 0.6
        top_offset_x = (GRID_SIZE - top_size) / 2
        top_offset_y = (GRID_SIZE - top_size) / 2 - 4
        pygame.draw.rect(
            castle,
            top_color,
            pygame.Rect(top_offset_x, top_offset_y, top_size, top_size),
            border_radius=2,
        )
        self.screen.blit(castle, (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE))

    def draw_status(self, logic, message, headline=None, headline_color=None):
        ui_y = SCREEN_HEIGHT - 100
        pygame.draw.rect(
            self.screen,
            COLOR_BOARD_BASE,
            pygame.Rect(0, ui_y, SCREEN_WIDTH, 100),
        )
        pygame.draw.line(
            self.screen,
            COLOR_GRID_SHADOW,
            (0, ui_y),
            (SCREEN_WIDTH, ui_y),
            2,
        )

        if logic.game_over:
            reason = logic.win_reason or "Game Over"
            reason_surface = self.large_font.render(reason, True, COLOR_HIGHLIGHT)
            self.screen.blit(reason_surface, (30, ui_y + 8))
            if message:
                message_surface = self.font.render(message, True, COLOR_TEXT)
                self.screen.blit(message_surface, (30, ui_y + 48))
            restart_surface = self.font.render("Restart: press R", True, COLOR_TEXT)
            self.screen.blit(restart_surface, (30, ui_y + 73))
            return

        if headline is None:
            headline = f"{player_name(logic.turn)} turn"
        if headline_color is None:
            headline_color = COLOR_BLUE_TOP if logic.turn == 1 else COLOR_RED_TOP
        headline_surface = self.large_font.render(
            headline,
            True,
            headline_color[:3],
        )
        self.screen.blit(headline_surface, (30, ui_y + 20))
        message_surface = self.font.render(message, True, COLOR_TEXT)
        self.screen.blit(message_surface, (30, ui_y + 60))

    def draw_frame(
        self,
        logic,
        message,
        *,
        headline=None,
        headline_color=None,
        ghost_player=None,
        mouse_position=None,
    ):
        self.draw_board(logic)
        self.draw_status(logic, message, headline, headline_color)

        if not logic.game_over and ghost_player is not None:
            if mouse_position is None:
                mouse_position = pygame.mouse.get_pos()
            coordinate = self.board_coordinate(mouse_position)
            if coordinate is not None:
                x, y = coordinate
                if not logic.is_impossible_action(x, y):
                    self.draw_castle(x, y, ghost_player, ghost=True)

        pygame.display.flip()

    def close(self):
        pygame.quit()


def new_game():
    """One named construction point used by both restart flows."""
    return GreatKingdomLogic()
