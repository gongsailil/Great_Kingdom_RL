"""Rules V2 engine for the documented Great Kingdom board game.

The former V1 engine and its 81-action checkpoints are preserved in the
``ppo-v1-final`` Git tag as a historical baseline.
"""

from enum import Enum, auto


BOARD_SIZE = 9
BLUE = 1
RED = 2
NEUTRAL = 3
CASTLES_PER_PLAYER = 40
PASS_ACTION = BOARD_SIZE * BOARD_SIZE
NUM_ACTIONS = PASS_ACTION + 1
BLUE_REQUIRED_TERRITORY_LEAD = 2


class MoveResultV2(Enum):
    IMPOSSIBLE_GAME_OVER = auto()
    IMPOSSIBLE_ACTION = auto()
    IMPOSSIBLE_OUT_OF_BOUNDS = auto()
    IMPOSSIBLE_OCCUPIED = auto()
    IMPOSSIBLE_OPPONENT_TERRITORY = auto()
    IMPOSSIBLE_SUICIDE = auto()
    IMPOSSIBLE_NO_CASTLES = auto()
    NORMAL = auto()
    CAPTURE_WIN = auto()
    PASS = auto()
    PASS_SCORE_END = auto()


LEGAL_PLACEMENT_RESULTS = frozenset(
    (MoveResultV2.NORMAL, MoveResultV2.CAPTURE_WIN)
)


def player_name(player):
    if player == BLUE:
        return "Blue"
    if player == RED:
        return "Red"
    raise ValueError("player must be Blue (1) or Red (2)")


def determine_scoring_winner(blue_territory, red_territory):
    """Apply the documented first-player threshold; the rule has no draw."""
    if blue_territory >= red_territory + BLUE_REQUIRED_TERRITORY_LEAD:
        return BLUE
    return RED


class GreatKingdomLogicV2:
    """Rules-only Great Kingdom state with PASS and finite inventories."""

    def __init__(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board[BOARD_SIZE // 2][BOARD_SIZE // 2] = NEUTRAL
        self.turn = BLUE
        self.game_over = False
        self.winner = None
        self.win_reason = ""
        self.last_move_result = None
        self.consecutive_passes = 0
        self.castles_remaining = {
            BLUE: CASTLES_PER_PLAYER,
            RED: CASTLES_PER_PLAYER,
        }
        self.score_blue = None
        self.score_red = None

    def copy(self):
        copied = GreatKingdomLogicV2()
        copied.board = [row[:] for row in self.board]
        copied.turn = self.turn
        copied.game_over = self.game_over
        copied.winner = self.winner
        copied.win_reason = self.win_reason
        copied.last_move_result = self.last_move_result
        copied.consecutive_passes = self.consecutive_passes
        copied.castles_remaining = dict(self.castles_remaining)
        copied.score_blue = self.score_blue
        copied.score_red = self.score_red
        return copied

    @staticmethod
    def _validate_player(player):
        if player not in (BLUE, RED):
            raise ValueError("player must be Blue (1) or Red (2)")

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def _empty_region_and_touched_players(self, start_x, start_y):
        region = {(start_x, start_y)}
        touched_players = set()
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if not self.is_on_board(nx, ny):
                    continue
                state = self.board[ny][nx]
                if state == 0 and (nx, ny) not in region:
                    region.add((nx, ny))
                    stack.append((nx, ny))
                elif state in (BLUE, RED):
                    touched_players.add(state)
                # The neutral castle is a wall, never an owner.
        return region, touched_players

    def _territory_owner_from_touched_players(self, touched_players):
        # Prevent the opening board from turning into one enormous territory
        # merely because only one player has placed a castle so far.
        has_blue = any(BLUE in row for row in self.board)
        has_red = any(RED in row for row in self.board)
        if not has_blue or not has_red:
            return 0
        if touched_players == {BLUE}:
            return BLUE
        if touched_players == {RED}:
            return RED
        return 0

    def get_territory_owner(self, x, y):
        if not self.is_on_board(x, y) or self.board[y][x] != 0:
            return None
        _, touched_players = self._empty_region_and_touched_players(x, y)
        return self._territory_owner_from_touched_players(touched_players)

    def territory_counts(self):
        counts = {BLUE: 0, RED: 0}
        visited = set()
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] != 0 or (x, y) in visited:
                    continue
                region, touched_players = self._empty_region_and_touched_players(x, y)
                visited.update(region)
                owner = self._territory_owner_from_touched_players(touched_players)
                if owner in (BLUE, RED):
                    counts[owner] += len(region)
        return counts

    def count_territory(self, player):
        self._validate_player(player)
        return self.territory_counts()[player]

    def get_group(self, start_x, start_y):
        if not self.is_on_board(start_x, start_y):
            return set()
        target = self.board[start_y][start_x]
        if target not in (BLUE, RED):
            return set()
        group = {(start_x, start_y)}
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if (
                    self.is_on_board(nx, ny)
                    and self.board[ny][nx] == target
                    and (nx, ny) not in group
                ):
                    group.add((nx, ny))
                    stack.append((nx, ny))
        return group

    def count_liberties(self, start_x, start_y, target_color=None):
        if not self.is_on_board(start_x, start_y):
            return 0
        if target_color is not None and self.board[start_y][start_x] != target_color:
            return 0
        group = self.get_group(start_x, start_y)
        liberties = set()
        for x, y in group:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if self.is_on_board(nx, ny) and self.board[ny][nx] == 0:
                    liberties.add((nx, ny))
        return len(liberties)

    def _placement_captures_opponent(self, player, x, y):
        opponent = 3 - player
        checked_groups = set()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if not self.is_on_board(nx, ny) or self.board[ny][nx] != opponent:
                continue
            group = self.get_group(nx, ny)
            group_key = frozenset(group)
            if group_key in checked_groups:
                continue
            checked_groups.add(group_key)
            if self.count_liberties(nx, ny, opponent) == 0:
                return True
        return False

    def classify_placement(self, player, x, y):
        """Classify without mutating state; capture takes priority over suicide."""
        self._validate_player(player)
        if self.game_over:
            return MoveResultV2.IMPOSSIBLE_GAME_OVER
        if not self.is_on_board(x, y):
            return MoveResultV2.IMPOSSIBLE_OUT_OF_BOUNDS
        if self.board[y][x] != 0:
            return MoveResultV2.IMPOSSIBLE_OCCUPIED
        if self.get_territory_owner(x, y) == 3 - player:
            return MoveResultV2.IMPOSSIBLE_OPPONENT_TERRITORY
        if self.castles_remaining[player] <= 0:
            return MoveResultV2.IMPOSSIBLE_NO_CASTLES

        self.board[y][x] = player
        try:
            if self._placement_captures_opponent(player, x, y):
                return MoveResultV2.CAPTURE_WIN
            if self.count_liberties(x, y, player) == 0:
                return MoveResultV2.IMPOSSIBLE_SUICIDE
            return MoveResultV2.NORMAL
        finally:
            self.board[y][x] = 0

    def is_impossible_action(self, x, y, player=None):
        if player is None:
            player = self.turn
        return self.classify_placement(player, x, y) not in LEGAL_PLACEMENT_RESULTS

    def get_playable_spots(self, player=None):
        if player is None:
            player = self.turn
        return [
            (x, y)
            for y in range(BOARD_SIZE)
            for x in range(BOARD_SIZE)
            if self.classify_placement(player, x, y) in LEGAL_PLACEMENT_RESULTS
        ]

    def place_stone_detailed(self, x, y):
        player = self.turn
        result = self.classify_placement(player, x, y)
        self.last_move_result = result
        if result not in LEGAL_PLACEMENT_RESULTS:
            return result

        self.board[y][x] = player
        self.castles_remaining[player] -= 1
        self.consecutive_passes = 0
        if result == MoveResultV2.CAPTURE_WIN:
            self.game_over = True
            self.winner = player
            self.win_reason = f"Winner = {player_name(player)} (capture)"
            return result

        self.turn = 3 - player
        return result

    def pass_turn(self):
        if self.game_over:
            self.last_move_result = MoveResultV2.IMPOSSIBLE_GAME_OVER
            return self.last_move_result

        self.consecutive_passes += 1
        self.turn = 3 - self.turn
        if self.consecutive_passes >= 2:
            self.calculate_score()
            self.last_move_result = MoveResultV2.PASS_SCORE_END
        else:
            self.last_move_result = MoveResultV2.PASS
        return self.last_move_result

    def apply_action(self, action):
        try:
            action = int(action)
        except (TypeError, ValueError):
            self.last_move_result = MoveResultV2.IMPOSSIBLE_ACTION
            return self.last_move_result
        if action == PASS_ACTION:
            return self.pass_turn()
        if 0 <= action < PASS_ACTION:
            return self.place_stone_detailed(
                action % BOARD_SIZE,
                action // BOARD_SIZE,
            )
        self.last_move_result = MoveResultV2.IMPOSSIBLE_ACTION
        return self.last_move_result

    def calculate_score(self):
        counts = self.territory_counts()
        self.score_blue = counts[BLUE]
        self.score_red = counts[RED]
        self.winner = determine_scoring_winner(self.score_blue, self.score_red)
        self.game_over = True
        self.win_reason = (
            f"Blue territory = {self.score_blue}; "
            f"Red territory = {self.score_red}; "
            f"Winner = {player_name(self.winner)}"
        )
        return self.winner
