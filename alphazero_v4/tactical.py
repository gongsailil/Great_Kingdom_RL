"""Rules-exact one-ply root solver; no heuristic or reward modification."""

from dataclasses import dataclass

import numpy as np

from alphazero_v3.value_oracle_audit import immediate_winning_actions
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import MoveResultV2


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
        return bool(
            self.opponent_threat_actions and not self.safe_defense_actions
        )


def solve_tactical_root(logic):
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
            mode="IMMEDIATE_WIN",
            legal_actions=legal_actions,
            allowed_actions=immediate,
            immediate_win_actions=immediate,
            opponent_threat_actions=(),
            safe_defense_actions=(),
            exact_unsafe_actions=(),
        )

    threats = tuple(immediate_winning_actions(logic, opponent))
    if not threats:
        return TacticalRoot(
            mode="NORMAL",
            legal_actions=legal_actions,
            allowed_actions=legal_actions,
            immediate_win_actions=(),
            opponent_threat_actions=(),
            safe_defense_actions=(),
            exact_unsafe_actions=(),
        )

    safe = []
    unsafe = []
    for action in legal_actions:
        child = logic.copy()
        result = child.apply_action(action)
        if result not in (
            MoveResultV2.NORMAL,
            MoveResultV2.CAPTURE_WIN,
            MoveResultV2.PASS,
            MoveResultV2.PASS_SCORE_END,
        ):
            raise RuntimeError(f"legal tactical action became {result.name}")
        if child.game_over:
            if child.winner == player:
                safe.append(action)
            else:
                unsafe.append(action)
        elif immediate_winning_actions(child, opponent):
            unsafe.append(action)
        else:
            safe.append(action)
    if safe:
        return TacticalRoot(
            mode="SAFE_DEFENSE",
            legal_actions=legal_actions,
            allowed_actions=tuple(safe),
            immediate_win_actions=(),
            opponent_threat_actions=threats,
            safe_defense_actions=tuple(safe),
            exact_unsafe_actions=tuple(unsafe),
        )
    return TacticalRoot(
        mode="FORCED_LOSS",
        legal_actions=legal_actions,
        allowed_actions=legal_actions,
        immediate_win_actions=(),
        opponent_threat_actions=threats,
        safe_defense_actions=(),
        exact_unsafe_actions=tuple(unsafe),
    )
