import pygame

# --- Setting Constants ---
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 850
BOARD_SIZE = 9
GRID_SIZE = 70
MARGIN = 45

# --- Color Definition ---
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

# --- Game Logic ---
class GreatKingdomLogic:
    def __init__(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn = 1
        self.game_over = False
        self.winner = None
        self.komi = 3.0
        self.win_reason = "" # Initialization
        self.board[4][4] = 3

    def copy(self):
        new_game = GreatKingdomLogic()
        new_game.board = [row[:] for row in self.board]
        new_game.turn = self.turn
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.win_reason = self.win_reason # Copy
        return new_game

    def is_on_board(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def get_empty_spots(self):
        spots = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:
                    spots.append((x, y))
        return spots

    def count_liberties(self, start_x, start_y, target_color):
        if self.board[start_y][start_x] != target_color: return 0
        group = set(); liberties = set(); stack = [(start_x, start_y)]
        group.add((start_x, start_y))
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if self.is_on_board(nx, ny):
                    state = self.board[ny][nx]
                    if state == 0: liberties.add((nx, ny))
                    elif state == target_color and (nx, ny) not in group:
                        group.add((nx, ny)); stack.append((nx, ny))
        return len(liberties)

    def get_territory_owner(self, x, y):
        if self.board[y][x] != 0: return None
        has_blue = any(1 in row for row in self.board)
        has_red = any(2 in row for row in self.board)
        visited = set(); stack = [(x, y)]; visited.add((x, y))
        touched_colors = set()
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if self.is_on_board(nx, ny):
                    state = self.board[ny][nx]
                    if state == 0:
                        if (nx, ny) not in visited: visited.add((nx, ny)); stack.append((nx, ny))
                    elif state == 1 or state == 2: touched_colors.add(state)
        is_blue = 1 in touched_colors
        is_red = 2 in touched_colors
        if is_blue and not is_red: return 1 if has_red else 0
        if is_red and not is_blue: return 2 if has_blue else 0
        return 0

    def place_stone(self, x, y):
        if self.game_over: return False
        if self.get_territory_owner(x, y) != 0: return False

        curr = self.turn; opp = 3 - self.turn
        self.board[y][x] = curr

        captured = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny) and self.board[ny][nx] == opp:
                if self.count_liberties(nx, ny, opp) == 0:
                    captured = True; break
        if captured:
            self.game_over = True; self.winner = curr
            w_str = "Blue" if curr == 1 else "Red"
            self.win_reason = f"{w_str} Wins (Capture)"
            return True

        if self.count_liberties(x, y, curr) == 0:
            self.board[y][x] = 0; return False

        self.turn = 3 - self.turn
        return True

    def check_game_end_simple(self):
        if self.game_over: return
        empty_spots = self.get_empty_spots()
        if not empty_spots: self.game_over = True; self.calculate_score(); return
        has_move = False
        for px, py in empty_spots:
            if self.get_territory_owner(px, py) == 0: has_move = True; break
        if not has_move: self.game_over = True; self.calculate_score()

    def calculate_score(self):
        b = 0; r = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:
                    o = self.get_territory_owner(x, y)
                    if o == 1: b += 1
                    elif o == 2: r += 1
        final_r = r + self.komi
        if b > final_r: self.winner = 1
        elif final_r > b: self.winner = 2
        else: self.winner = 0
        w_str = "Blue" if self.winner == 1 else ("Red" if self.winner == 2 else "Draw")
        self.win_reason = f"{w_str} ({b} vs {final_r})"

# --- [UI] ---
class GameUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Great Kingdom")
        self.clock = pygame.time.Clock()
        self.logic = GreatKingdomLogic()
        try:
            self.font = pygame.font.SysFont("malgungothic", 20, True)
            self.l_font = pygame.font.SysFont("malgungothic", 30, True)
        except:
            self.font = pygame.font.SysFont("arial", 20, True)
            self.l_font = pygame.font.SysFont("arial", 30, True)
        self.info_msg = "Game Start"
        self.is_ai_thinking = False

    def draw_board(self):
        self.screen.fill(COLOR_BOARD_BASE)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cx, cy = MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE
                pygame.draw.rect(self.screen, COLOR_GRID_SHADOW, (cx+2, cy+2, GRID_SIZE-4, GRID_SIZE-4))
                pygame.draw.rect(self.screen, COLOR_BOARD_INSET, (cx+5, cy+5, GRID_SIZE-10, GRID_SIZE-10))
                if self.logic.board[y][x] != 0: self.draw_piece(cx, cy, self.logic.board[y][x])

    def draw_piece(self, cx, cy, st, ghost=False):
        if st==1: b,t = COLOR_BLUE_BASE, COLOR_BLUE_TOP
        elif st==2: b,t = COLOR_RED_BASE, COLOR_RED_TOP
        elif st==3: b,t = COLOR_NEUTRAL_BASE, COLOR_NEUTRAL_TOP
        else: return
        if ghost: b=(b[0],b[1],b[2],100); t=(t[0],t[1],t[2],120)
        s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, b, (10,10,50,50), border_radius=4)
        pygame.draw.rect(s, t, (20,16,30,30), border_radius=2)
        self.screen.blit(s, (cx, cy))

    def draw_ui(self):
        ui_y = SCREEN_HEIGHT - 100
        pygame.draw.rect(self.screen, COLOR_BOARD_BASE, (0, ui_y, SCREEN_WIDTH, 100))
        pygame.draw.line(self.screen, COLOR_GRID_SHADOW, (0, ui_y), (SCREEN_WIDTH, ui_y), 2)
        if self.logic.game_over:
            reason = self.logic.win_reason if self.logic.win_reason else "Game Over"
            t = self.l_font.render(reason, True, COLOR_HIGHLIGHT)
            self.screen.blit(t, (30, ui_y + 15))
        else:
            m = self.font.render(self.info_msg, True, COLOR_TEXT)
            self.screen.blit(m, (30, ui_y + 60))

    def run(self): pass
