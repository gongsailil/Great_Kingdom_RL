"""Rules-exact one-ply root filtering used by AlphaZero V5."""

from dataclasses import dataclass

import numpy as np

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import BOARD_SIZE, PASS_ACTION, MoveResultV2


LEGAL_RESULTS = (
    MoveResultV2.NORMAL,
    MoveResultV2.CAPTURE_WIN,
    MoveResultV2.PASS,
    MoveResultV2.PASS_SCORE_END,
)


def immediate_winning_actions(logic, player=None):
    """Return actions proven by Rules V2 to win in exactly one move."""
    if logic.game_over:
        return []
    player = logic.turn if player is None else int(player)
    wins = []
    for action in sorted(_capture_candidates(logic, player)):
        if logic.classify_placement(
            player, action % BOARD_SIZE, action // BOARD_SIZE
        ) == MoveResultV2.CAPTURE_WIN:
            wins.append(action)
    pass_state = logic.copy()
    pass_state.turn = player
    result = pass_state.apply_action(PASS_ACTION)
    if result == MoveResultV2.PASS_SCORE_END and pass_state.winner == player:
        wins.append(PASS_ACTION)
    return wins


def _capture_candidates(logic, player):
    """Find last-liberty candidates before final Rules V2 classification."""
    opponent = 3 - int(player)
    visited, candidates = set(), set()
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if logic.board[y][x] != opponent or (x, y) in visited:
                continue
            group = logic.get_group(x, y)
            visited.update(group)
            liberties = set()
            for group_x, group_y in group:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_x, next_y = group_x + dx, group_y + dy
                    if (
                        logic.is_on_board(next_x, next_y)
                        and logic.board[next_y][next_x] == 0
                    ):
                        liberties.add((next_x, next_y))
            if len(liberties) == 1:
                liberty_x, liberty_y = next(iter(liberties))
                candidates.add(liberty_x + liberty_y * BOARD_SIZE)
    return candidates


@dataclass(frozen=True)
class TacticalRoot:
    mode: str
    legal_actions: tuple
    allowed_actions: tuple
    immediate_win_actions: tuple
    opponent_threat_actions: tuple
    safe_defense_actions: tuple
    exact_unsafe_actions: tuple

    @property
    def forced_loss(self):
        return bool(self.opponent_threat_actions and not self.safe_defense_actions)


def solve_tactical_root(logic):
    """Restrict only actions with a Rules-proven one-ply terminal outcome."""
    if logic.game_over:
        raise ValueError("tactical root requires an active Rules V2 state")
    player = logic.turn
    opponent = 3 - player
    legal_actions = tuple(
        int(action)
        for action in np.flatnonzero(action_mask_for_logic(logic, player))
    )
    immediate = tuple(immediate_winning_actions(logic, player))
    if immediate:
        return TacticalRoot(
            "IMMEDIATE_WIN", legal_actions, immediate, immediate, (), (), ()
        )
    threats = tuple(immediate_winning_actions(logic, opponent))
    if not threats:
        return TacticalRoot(
            "NORMAL", legal_actions, legal_actions, (), (), (), ()
        )
    safe, unsafe = [], []
    for action in legal_actions:
        child = logic.copy()
        result = child.apply_action(action)
        if result not in LEGAL_RESULTS:
            raise RuntimeError(f"legal tactical action became {result.name}")
        if child.game_over:
            (safe if child.winner == player else unsafe).append(action)
        elif immediate_winning_actions(child, opponent):
            unsafe.append(action)
        else:
            safe.append(action)
    if safe:
        return TacticalRoot(
            "SAFE_DEFENSE", legal_actions, tuple(safe), (), threats,
            tuple(safe), tuple(unsafe)
        )
    return TacticalRoot(
        "FORCED_LOSS", legal_actions, legal_actions, (), threats, (), tuple(unsafe)
    )
