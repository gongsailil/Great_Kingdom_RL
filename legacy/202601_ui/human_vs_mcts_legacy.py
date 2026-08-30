import pygame
import sys
import random
import math
import copy
import time

# --- 설정 상수 ---
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 850
BOARD_SIZE = 9
GRID_SIZE = 70
MARGIN = 45

# --- 색상 정의 ---
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

# --- [전략] 위치 가치 테이블 (Heuristic Map) ---
# 중앙(4,4) 중립 성 주변을 가장 높게 평가하여 AI가 중앙 싸움을 하도록 유도
POSITION_WEIGHTS = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 3, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 5, 5, 5, 3, 2, 1],
    [1, 2, 3, 5, 0, 5, 3, 2, 1],  # 정중앙은 이미 성이 있음(0)
    [1, 2, 3, 5, 5, 5, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 3, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]


# --- [Logic] 게임 로직 클래스 ---
class GreatKingdomLogic:
    def __init__(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn = 1
        self.game_over = False
        self.winner = None
        self.komi = 3.0
        self.win_reason = ""
        self.board[4][4] = 3

    def copy(self):
        new_game = GreatKingdomLogic()
        new_game.board = [row[:] for row in self.board]
        new_game.turn = self.turn
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.win_reason = self.win_reason
        return new_game

    def is_on_board(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def get_empty_spots(self):
        # 중앙부터 탐색하도록 정렬하면 킬러 무브 발견 확률이 높아짐
        spots = [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if self.board[y][x] == 0]
        # 중앙에 가까운 순서로 정렬 (Heuristic Sort)
        spots.sort(key=lambda p: POSITION_WEIGHTS[p[1]][p[0]], reverse=True)
        return spots

    def count_liberties(self, start_x, start_y, target_color):
        if self.board[start_y][start_x] != target_color: return 0

        group = set();
        liberties = set();
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
                        group.add((nx, ny));
                        stack.append((nx, ny))
        return len(liberties)

    def get_territory_owner(self, x, y):
        if self.board[y][x] != 0: return None

        has_blue = any(1 in row for row in self.board)
        has_red = any(2 in row for row in self.board)

        visited = set();
        stack = [(x, y)];
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
                            visited.add((nx, ny));
                            stack.append((nx, ny))
                    elif state == 1 or state == 2:
                        touched_colors.add(state)

        is_blue_wall = 1 in touched_colors
        is_red_wall = 2 in touched_colors

        if is_blue_wall and not is_red_wall: return 1 if has_red else 0
        if is_red_wall and not is_blue_wall: return 2 if has_blue else 0
        return 0

    def place_stone(self, x, y, simulate=False):
        if self.game_over: return False

        # 영토 체크 (시뮬레이션 속도를 위해 simulate=True일 땐 가끔 생략 가능하지만 정확도 위해 유지)
        if self.get_territory_owner(x, y) != 0: return False

        curr = self.turn
        opp = 3 - self.turn
        self.board[y][x] = curr

        # 포획 체크
        captured = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny) and self.board[ny][nx] == opp:
                if self.count_liberties(nx, ny, opp) == 0:
                    captured = True;
                    break

        if captured:
            self.game_over = True
            self.winner = curr
            if not simulate:
                w_name = "Blue" if curr == 1 else "Red"
                self.win_reason = f"{w_name} Wins! (Captured Enemy)"
            return True

        # 자살수 금지
        if self.count_liberties(x, y, curr) == 0:
            self.board[y][x] = 0
            return False

        self.turn = 3 - self.turn
        return True

    def check_game_end_simple(self):
        if self.game_over: return

        # 최적화: 빈칸 리스트를 매번 뽑지 않고, 보드 스캔
        has_move = False
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:
                    if self.get_territory_owner(x, y) == 0:
                        has_move = True;
                        break
            if has_move: break

        if not has_move:
            self.game_over = True
            self.calculate_score()

    def calculate_score(self):
        b_score = 0;
        r_score = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:
                    owner = self.get_territory_owner(x, y)
                    if owner == 1:
                        b_score += 1
                    elif owner == 2:
                        r_score += 1

        final_r = r_score + self.komi
        self.final_score_diff = final_r - b_score  # 양수면 레드 유리

        if b_score > final_r:
            self.winner = 1
        elif final_r > b_score:
            self.winner = 2
        else:
            self.winner = 0

        if self.winner == 1:
            self.win_reason = f"Blue Wins ({b_score} vs {final_r:.1f})"
        elif self.winner == 2:
            self.win_reason = f"Red Wins ({b_score} vs {final_r:.1f})"
        else:
            self.win_reason = "Draw"


# --- [AI] Genius MCTS Agent (전략 강화) ---
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.possible_moves = game_state.get_empty_spots()
        self.player_just_moved = 3 - game_state.turn

    def ucb1_select_child(self):
        # Exploration(탐험) 비중을 1.2로 설정하여 밸런스 유지
        s = sorted(self.children, key=lambda c: c.wins / (c.visits + 1e-4) + 1.2 * math.sqrt(
            2 * math.log(self.visits + 1) / (c.visits + 1e-4)))
        return s[-1]

    def add_child(self, move, state):
        child = MCTSNode(state, parent=self, move=move)
        if move in self.possible_moves:
            self.possible_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        reward = 0
        if self.player_just_moved == result:
            reward = 1.0  # 승리
        elif result == 0:
            reward = 0.5  # 무승부
        else:
            reward = 0.0  # 패배
        self.wins += reward


class GeniusAgent:
    def __init__(self, iterations=300):  # 생각하는 깊이
        self.iterations = iterations

    def get_smart_simulation_move(self, state):
        """
        [전략 시뮬레이션]
        랜덤으로 두지 않고, '좋은 수'를 확률적으로 선택하는 똑똑한 Rollout
        """
        candidates = []
        empty_spots = state.get_empty_spots()
        if not empty_spots: return None

        opp = 3 - state.turn

        for x, y in empty_spots:
            weight = 1.0

            # 1. 위치 가중치 (중앙 선호)
            weight += POSITION_WEIGHTS[y][x] * 2.0

            # 2. 공격적 수 (상대 돌 근처에 두어 압박)
            # 상하좌우에 상대 돌이 있으면 가중치 대폭 증가
            near_enemy = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if state.is_on_board(nx, ny) and state.board[ny][nx] == opp:
                    near_enemy = True
                    break
            if near_enemy:
                weight += 5.0  # 전투 유도

            candidates.append(((x, y), weight))

        # 가중치 기반 랜덤 선택 (Weighted Random Choice)
        total_weight = sum(w for _, w in candidates)
        r = random.uniform(0, total_weight)
        upto = 0
        for move, w in candidates:
            if upto + w >= r:
                return move
            upto += w
        return candidates[-1][0]

    def get_best_move(self, root_game_state):
        # --- [1단계: 킬각 & 방어 본능 (Absolute Priority)] ---
        valid_moves = []
        empty_spots = root_game_state.get_empty_spots()
        opponent_turn = 3 - root_game_state.turn

        my_winning_moves = []
        must_block_moves = []

        for m in empty_spots:
            # 1. 내가 둬서 이기는 수 (Kill)
            test_state = root_game_state.copy()
            if test_state.place_stone(m[0], m[1], simulate=True):
                if test_state.game_over and test_state.winner == root_game_state.turn:
                    my_winning_moves.append(m)

                # 그 수가 유효하다면, 반대로 상대가 뒀을 때 내가 죽는지 확인 (Defense)
                # 상대 턴으로 가정
                defense_state = root_game_state.copy()
                defense_state.turn = opponent_turn
                if defense_state.place_stone(m[0], m[1], simulate=True):
                    if defense_state.game_over and defense_state.winner == opponent_turn:
                        must_block_moves.append(m)

        if my_winning_moves:
            print(f"Genius AI: 킬각 발견! {my_winning_moves[0]}로 끝내겠습니다.")
            return my_winning_moves[0]

        if must_block_moves:
            print(f"Genius AI: 위기 감지! {must_block_moves[0]}를 막아야 합니다.")
            return must_block_moves[0]

        # --- [2단계: MCTS (Deep Thought)] ---
        root_node = MCTSNode(root_game_state.copy())
        end_time = time.time() + 2.5  # 최대 2.5초 생각

        for i in range(self.iterations):
            if i % 20 == 0: pygame.event.pump()
            if time.time() > end_time: break

            node = root_node
            state = root_node.game_state.copy()

            # Selection
            while not node.possible_moves and node.children:
                node = node.ucb1_select_child()
                state.place_stone(node.move[0], node.move[1], simulate=True)

            # Expansion
            if node.possible_moves:
                # [전략] 가능한 수 중 중앙/전투 지역부터 먼저 확장 (Heuristic Expansion)
                # possible_moves는 이미 heuristic sort 되어 있음
                m = node.possible_moves.pop(0)

                temp_state = state.copy()
                if temp_state.place_stone(m[0], m[1], simulate=True):
                    node = node.add_child(m, temp_state)
                    state = temp_state

            # Simulation (Smart Rollout)
            depth = 0
            while not state.game_over and depth < 30:
                # 똑똑한 랜덤 무브 (가중치 기반)
                m = self.get_smart_simulation_move(state)
                if not m:
                    state.check_game_end_simple()
                    break

                if not state.place_stone(m[0], m[1], simulate=True):
                    # 실패 시(자살수 등) 그냥 다음 턴으로
                    pass
                depth += 1

            if not state.game_over: state.check_game_end_simple()

            # Backpropagation
            while node != None:
                node.update(state.winner)
                node = node.parent

        if not root_node.children: return None

        best_child = sorted(root_node.children, key=lambda c: c.visits)[-1]
        print(f"Genius AI 선택: {best_child.move} (승률: {best_child.wins / best_child.visits:.2f})")
        return best_child.move


# --- [UI] 메인 게임 클래스 ---
class GameUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Great Kingdom: Master AI Challenge")
        self.clock = pygame.time.Clock()

        self.logic = GreatKingdomLogic()
        self.ai = GeniusAgent(iterations=400)  # 고성능 설정

        try:
            self.font = pygame.font.SysFont("malgungothic", 20, True)
            self.l_font = pygame.font.SysFont("malgungothic", 30, True)
        except:
            self.font = pygame.font.SysFont("arial", 20, True)
            self.l_font = pygame.font.SysFont("arial", 30, True)

        self.info_msg = "도전자여, 덤벼보세요. (파랑: 유저 vs 빨강: AI)"
        self.is_ai_thinking = False

    def draw_board(self):
        self.screen.fill(COLOR_BOARD_BASE)
        # 보드판 및 좌표 표시
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cx, cy = MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE
                pygame.draw.rect(self.screen, COLOR_GRID_SHADOW, (cx + 2, cy + 2, GRID_SIZE - 4, GRID_SIZE - 4))
                pygame.draw.rect(self.screen, COLOR_BOARD_INSET, (cx + 5, cy + 5, GRID_SIZE - 10, GRID_SIZE - 10))

                # 중앙 성 강조 (바닥 색 다르게)
                if x == 4 and y == 4:
                    pygame.draw.rect(self.screen, (230, 230, 255), (cx + 5, cy + 5, GRID_SIZE - 10, GRID_SIZE - 10))

                state = self.logic.board[y][x]
                if state != 0:
                    self.draw_piece(cx, cy, state)

    def draw_piece(self, cx, cy, state, ghost=False):
        if state == 1:
            b, t = COLOR_BLUE_BASE, COLOR_BLUE_TOP
        elif state == 2:
            b, t = COLOR_RED_BASE, COLOR_RED_TOP
        elif state == 3:
            b, t = COLOR_NEUTRAL_BASE, COLOR_NEUTRAL_TOP
        else:
            return

        if ghost:
            b = (b[0], b[1], b[2], 100)
            t = (t[0], t[1], t[2], 120)

        s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, b, (10, 10, 50, 50), border_radius=4)
        pygame.draw.rect(s, t, (20, 16, 30, 30), border_radius=2)
        self.screen.blit(s, (cx, cy))

    def draw_ui(self):
        ui_y = SCREEN_HEIGHT - 100
        pygame.draw.rect(self.screen, COLOR_BOARD_BASE, (0, ui_y, SCREEN_WIDTH, 100))
        pygame.draw.line(self.screen, COLOR_GRID_SHADOW, (0, ui_y), (SCREEN_WIDTH, ui_y), 2)

        if self.logic.game_over:
            reason = self.logic.win_reason if self.logic.win_reason else "Game Over"
            t = self.l_font.render(reason, True, COLOR_HIGHLIGHT)
            self.screen.blit(t, (30, ui_y + 15))
            r = self.font.render("재도전 하려면 'R' 키를 누르세요", True, COLOR_TEXT)
            self.screen.blit(r, (30, ui_y + 60))
        else:
            if self.is_ai_thinking:
                turn_t = "AI가 묘수를 계산 중입니다..."
                c = COLOR_RED_TOP
            else:
                turn_t = "당신의 차례입니다"
                c = COLOR_BLUE_TOP

            s = self.l_font.render(turn_t, True, c[:3])
            self.screen.blit(s, (30, ui_y + 20))
            m = self.font.render(self.info_msg, True, COLOR_TEXT)
            self.screen.blit(m, (30, ui_y + 60))

    def run(self):
        while True:
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()

                # 사용자 입력
                if not self.logic.game_over and self.logic.turn == 1 and not self.is_ai_thinking:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        gx = int((mx - MARGIN) / GRID_SIZE)
                        gy = int((my - MARGIN) / GRID_SIZE)
                        if self.logic.is_on_board(gx, gy):
                            if self.logic.place_stone(gx, gy):
                                self.info_msg = "AI Turn..."
                                self.is_ai_thinking = True
                            else:
                                self.info_msg = "착수 불가 (영토 또는 자살수)"

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.logic = GreatKingdomLogic()
                        self.info_msg = "새로운 도전!"
                        self.is_ai_thinking = False

            self.draw_board()
            self.draw_ui()

            if not self.logic.game_over and self.logic.turn == 1 and not self.is_ai_thinking:
                mx, my = pygame.mouse.get_pos()
                gx, gy = int((mx - MARGIN) / GRID_SIZE), int((my - MARGIN) / GRID_SIZE)
                if self.logic.is_on_board(gx, gy) and self.logic.board[gy][gx] == 0:
                    if self.logic.get_territory_owner(gx, gy) == 0:
                        self.draw_piece(MARGIN + gx * GRID_SIZE, MARGIN + gy * GRID_SIZE, 1, ghost=True)

            pygame.display.flip()

            if not self.logic.game_over and self.logic.turn == 2 and self.is_ai_thinking:
                pygame.display.update()

                best_move = self.ai.get_best_move(self.logic)

                if best_move:
                    self.logic.place_stone(best_move[0], best_move[1])
                    self.info_msg = "AI가 강력한 수를 두었습니다."
                else:
                    self.logic.check_game_end_simple()
                    self.info_msg = "AI가 더 이상 둘 곳이 없습니다."

                self.is_ai_thinking = False


if __name__ == "__main__":
    game = GameUI()
    game.run()