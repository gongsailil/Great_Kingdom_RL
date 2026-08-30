import pygame
import sys

# --- 설정 상수 ---
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 780
BOARD_SIZE = 9
GRID_SIZE = 70
MARGIN = 45

# --- 색상 정의 ---
COLOR_BOARD_BASE = (245, 245, 245)
COLOR_BOARD_INSET = (255, 255, 255)
COLOR_GRID_SHADOW = (210, 210, 210)

# 말 색상 (R, G, B, A)
COLOR_BLUE_BASE = (0, 100, 230, 180)
COLOR_BLUE_TOP = (50, 150, 255, 220)

COLOR_RED_BASE = (220, 30, 30, 180)
COLOR_RED_TOP = (255, 80, 80, 220)

COLOR_NEUTRAL_BASE = (220, 220, 220, 255)
COLOR_NEUTRAL_TOP = (255, 255, 255, 255)

COLOR_TEXT = (50, 50, 50)
COLOR_HIGHLIGHT = (220, 50, 50)


# --- 게임 로직 클래스 ---
class GreatKingdomGame:
    def __init__(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        # 0:빈곳, 1:파랑, 2:빨강, 3:중립
        self.turn = 1  # 1: 파랑(선공), 2: 빨강
        self.game_over = False
        self.winner = None
        self.win_reason = ""
        self.info_message = "게임 시작: 파랑(Blue)부터 두세요."

        # 중립 성 배치
        self.board[4][4] = 3

        # 점수 정보
        self.komi = 3.0  # 후공 덤 3집
        self.score_blue = 0
        self.score_red = 0

    def is_on_board(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def switch_turn(self):
        self.turn = 3 - self.turn

        # 활로 계산

    def count_liberties(self, start_x, start_y, target_color):
        if self.board[start_y][start_x] != target_color: return 0
        group = set()
        liberties = set()
        stack = [(start_x, start_y)]
        group.add((start_x, start_y))

        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if self.is_on_board(nx, ny):
                    state = self.board[ny][nx]
                    if state == 0:
                        liberties.add((nx, ny))
                    elif state == target_color and (nx, ny) not in group:
                        group.add((nx, ny))
                        stack.append((nx, ny))
        return len(liberties)

    # [수정됨] 해당 좌표가 누구의 '완성된 영토'인지 판별
    def get_territory_owner(self, x, y):
        if self.board[y][x] != 0: return None

        # 1. 현재 보드에 각 돌이 하나라도 있는지 확인 (버그 수정 핵심)
        has_blue_stone = any(1 in row for row in self.board)
        has_red_stone = any(2 in row for row in self.board)

        # Flood Fill로 빈 공간 그룹 탐색
        visited = set()
        stack = [(x, y)]
        visited.add((x, y))

        touched_colors = set()

        while stack:
            cx, cy = stack.pop()

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if self.is_on_board(nx, ny):
                    state = self.board[ny][nx]
                    if state == 0:
                        if (nx, ny) not in visited:
                            visited.add((nx, ny))
                            stack.append((nx, ny))
                    elif state == 1 or state == 2:
                        touched_colors.add(state)

        has_blue_contact = 1 in touched_colors
        has_red_contact = 2 in touched_colors

        # [중요 수정] 상대방 돌이 아예 없는 경우, 내 돌과 닿아있어도 '영토'로 인정하지 않음 (그냥 공터)
        if has_blue_contact and not has_red_contact:
            if has_red_stone:
                return 1  # 빨강이 존재할 때만 파랑 영토로 인정
            else:
                return 0  # 빨강이 없으면 그냥 빈 땅

        if has_red_contact and not has_blue_contact:
            if has_blue_stone:
                return 2
            else:
                return 0

        return 0  # 둘 다 닿아있거나 아무것도 아니면 공배

    def place_stone(self, x, y):
        if self.game_over or self.board[y][x] != 0: return

        current_color = self.turn
        opponent_color = 3 - self.turn

        # [규칙 1] 이미 완성된 영토에는 착수 불가
        territory_owner = self.get_territory_owner(x, y)
        if territory_owner == 1:
            self.info_message = "착수 금지: 이미 파랑(Blue)의 확정된 집입니다."
            print(self.info_message)
            return
        elif territory_owner == 2:
            self.info_message = "착수 금지: 이미 빨강(Red)의 확정된 집입니다."
            print(self.info_message)
            return

        # 가착수
        self.board[y][x] = current_color

        # 1. 상대방 돌 잡기 (서든데스)
        opponent_captured = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny) and self.board[ny][nx] == opponent_color:
                if self.count_liberties(nx, ny, opponent_color) == 0:
                    opponent_captured = True
                    break

        if opponent_captured:
            self.game_over = True
            self.winner = current_color
            w_text = '파랑(Blue)' if current_color == 1 else '빨강(Red)'
            self.win_reason = f"{w_text} 승리! (상대말 포획)"
            self.info_message = "게임 종료!"
            return

        # 2. 자살수 금지
        if self.count_liberties(x, y, current_color) == 0:
            self.board[y][x] = 0  # 원상복구
            self.info_message = "착수 금지: 자살수입니다."
            print(self.info_message)
            return

        # 정상 착수
        self.switch_turn()
        self.info_message = f"{'파랑' if self.turn == 1 else '빨강'}의 차례입니다."

        # 게임 종료 여부 확인
        self.check_if_game_ends()

    def check_if_game_ends(self):
        """남은 모든 빈 공간이 '영토'로 확정되었는지 확인"""
        has_valid_move = False

        # 빨강, 파랑이 최소 하나씩은 있어야 영토전이 성립하므로,
        # 초반에는 게임 종료 체크를 느슨하게 함
        has_blue = any(1 in row for row in self.board)
        has_red = any(2 in row for row in self.board)

        if not has_blue or not has_red:
            return  # 아직 싸움이 시작 안 됐으므로 게임 계속

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:
                    owner = self.get_territory_owner(x, y)
                    if owner == 0:  # 아직 누구 땅도 아닌 곳(공배)이 있으면 게임 계속
                        has_valid_move = True
                        break
            if has_valid_move: break

        if not has_valid_move:
            self.calculate_final_score()

    def calculate_final_score(self):
        self.game_over = True

        blue_territory = 0
        red_territory = 0

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:
                    owner = self.get_territory_owner(x, y)
                    if owner == 1:
                        blue_territory += 1
                    elif owner == 2:
                        red_territory += 1

        self.score_blue = blue_territory
        self.score_red = red_territory + self.komi

        result_msg = f"파랑: {self.score_blue}집 vs 빨강: {self.score_red}집"
        if self.score_blue > self.score_red:
            self.winner = 1
            self.win_reason = f"파랑 승리! ({result_msg})"
        elif self.score_red > self.score_blue:
            self.winner = 2
            self.win_reason = f"빨강 승리! ({result_msg})"
        else:
            self.winner = 0
            self.win_reason = f"무승부! ({result_msg})"

        self.info_message = "더 이상 둘 곳이 없어 게임 종료."


# --- 그리기 함수들 ---
def draw_plastic_board(screen):
    screen.fill(COLOR_BOARD_BASE)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            cx = MARGIN + x * GRID_SIZE
            cy = MARGIN + y * GRID_SIZE
            pygame.draw.rect(screen, COLOR_GRID_SHADOW, (cx + 2, cy + 2, GRID_SIZE - 4, GRID_SIZE - 4))
            inset_margin = 5
            pygame.draw.rect(screen, COLOR_BOARD_INSET,
                             (cx + inset_margin, cy + inset_margin, GRID_SIZE - inset_margin * 2,
                              GRID_SIZE - inset_margin * 2))


def draw_castle_piece(screen, x, y, state, is_ghost=False):
    cx = MARGIN + x * GRID_SIZE
    cy = MARGIN + y * GRID_SIZE

    if state == 1:
        base_c, top_c = COLOR_BLUE_BASE, COLOR_BLUE_TOP
    elif state == 2:
        base_c, top_c = COLOR_RED_BASE, COLOR_RED_TOP
    elif state == 3:
        base_c, top_c = COLOR_NEUTRAL_BASE, COLOR_NEUTRAL_TOP
    else:
        return

    if is_ghost:
        base_c = (base_c[0], base_c[1], base_c[2], 100)
        top_c = (top_c[0], top_c[1], top_c[2], 120)

    castle_surf = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    base_size = GRID_SIZE * 0.8
    base_offset = (GRID_SIZE - base_size) / 2
    base_rect = pygame.Rect(base_offset, base_offset, base_size, base_size)
    pygame.draw.rect(castle_surf, base_c, base_rect, border_radius=4)

    top_size = base_size * 0.6
    top_offset_x = (GRID_SIZE - top_size) / 2
    top_offset_y = (GRID_SIZE - top_size) / 2 - 4
    top_rect = pygame.Rect(top_offset_x, top_offset_y, top_size, top_size)
    pygame.draw.rect(castle_surf, top_c, top_rect, border_radius=2)
    screen.blit(castle_surf, (cx, cy))


def draw_ui(screen, game, font, large_font):
    ui_y = SCREEN_HEIGHT - 100
    ui_rect = pygame.Rect(0, ui_y, SCREEN_WIDTH, 100)
    pygame.draw.rect(screen, COLOR_BOARD_BASE, ui_rect)
    pygame.draw.line(screen, COLOR_GRID_SHADOW, (0, ui_y), (SCREEN_WIDTH, ui_y), 2)

    if not game.game_over:
        turn_text = "파랑(Blue) 차례" if game.turn == 1 else "빨강(Red) 차례"
        turn_color = COLOR_BLUE_TOP if game.turn == 1 else COLOR_RED_TOP
        turn_surf = large_font.render(turn_text, True, turn_color[:3])
        screen.blit(turn_surf, (30, ui_y + 20))
        msg_surf = font.render(game.info_message, True, COLOR_TEXT)
        screen.blit(msg_surf, (30, ui_y + 60))
    else:
        win_surf = large_font.render(game.win_reason, True, COLOR_HIGHLIGHT)
        screen.blit(win_surf, (30, ui_y + 15))
        restart_surf = font.render("다시 시작: 'R' 키", True, COLOR_TEXT)
        screen.blit(restart_surf, (30, ui_y + 60))


# --- 메인 루프 ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Great Kingdom - Final Fixed")

    try:
        font = pygame.font.SysFont("malgungothic", 18, True)
        large_font = pygame.font.SysFont("malgungothic", 26, True)
    except:
        font = pygame.font.SysFont("arial", 18, True)
        large_font = pygame.font.SysFont("arial", 26, True)

    game = GreatKingdomGame()
    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                if event.button == 1:
                    mouse_x, mouse_y = event.pos
                    grid_x = int((mouse_x - MARGIN) / GRID_SIZE)
                    grid_y = int((mouse_y - MARGIN) / GRID_SIZE)

                    if 0 <= grid_x < BOARD_SIZE and 0 <= grid_y < BOARD_SIZE:
                        cx = MARGIN + grid_x * GRID_SIZE
                        cy = MARGIN + grid_y * GRID_SIZE
                        if cx < mouse_x < cx + GRID_SIZE and cy < mouse_y < cy + GRID_SIZE:
                            game.place_stone(grid_x, grid_y)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = GreatKingdomGame()

        draw_plastic_board(screen)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if game.board[y][x] != 0:
                    draw_castle_piece(screen, x, y, game.board[y][x])

        draw_ui(screen, game, font, large_font)

        if not game.game_over:
            mx, my = pygame.mouse.get_pos()
            gx = int((mx - MARGIN) / GRID_SIZE)
            gy = int((my - MARGIN) / GRID_SIZE)
            if 0 <= gx < BOARD_SIZE and 0 <= gy < BOARD_SIZE:
                if game.board[gy][gx] == 0:
                    owner = game.get_territory_owner(gx, gy)
                    if owner == 0:
                        draw_castle_piece(screen, gx, gy, game.turn, is_ghost=True)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()