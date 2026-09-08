"""Decode the nine-plane historical replay state used to seed V5 boards.

Only board state is imported. Historical policy and result labels are never used.
"""

import numpy as np

from great_kingdom_v2 import (
    BLUE, RED, BOARD_SIZE, CASTLES_PER_PLAYER, NEUTRAL, GreatKingdomLogicV2,
)


LEGACY_SHAPE = (9, BOARD_SIZE, BOARD_SIZE)


def _constant(plane, name, atol):
    value = float(plane[0, 0])
    if not np.allclose(plane, value, rtol=0, atol=atol):
        raise ValueError(f"{name} plane is not constant")
    return value


def encode_legacy_state(logic):
    current, opponent = logic.turn, 3 - logic.turn
    board = np.asarray(logic.board)
    encoded = np.zeros(LEGACY_SHAPE, dtype=np.float32)
    encoded[0] = board == current
    encoded[1] = board == opponent
    encoded[2] = board == NEUTRAL
    encoded[3].fill(logic.consecutive_passes / 2.0)
    encoded[4].fill(logic.castles_remaining[current] / CASTLES_PER_PLAYER)
    encoded[5].fill(logic.castles_remaining[opponent] / CASTLES_PER_PLAYER)
    encoded[6].fill(1.0 if current == BLUE else 0.0)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if logic.board[y][x] != 0:
                continue
            owner = logic.get_territory_owner(x, y)
            if owner == current:
                encoded[7, y, x] = 1.0
            elif owner == opponent:
                encoded[8, y, x] = 1.0
    return encoded


def decode_legacy_state(state, atol=1e-6):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != LEGACY_SHAPE or not np.all(np.isfinite(state)):
        raise ValueError("legacy replay state must be a finite (9,9,9) tensor")
    absolute = _constant(state[6], "absolute-color", atol)
    if np.isclose(absolute, 1.0, rtol=0, atol=atol):
        current = BLUE
    elif np.isclose(absolute, 0.0, rtol=0, atol=atol):
        current = RED
    else:
        raise ValueError("absolute-color plane is invalid")
    opponent = 3 - current
    occupancy = np.rint(state[:3])
    if not np.allclose(state[:3], occupancy, rtol=0, atol=atol):
        raise ValueError("occupancy planes are not binary")
    if np.any(occupancy.sum(axis=0) > 1):
        raise ValueError("occupancy planes overlap")
    passes_value = _constant(state[3], "consecutive-passes", atol)
    passes = int(round(passes_value * 2))
    if passes not in (0, 1) or not np.isclose(passes_value, passes / 2, atol=atol):
        raise ValueError("consecutive-passes plane is invalid")

    def inventory(plane, name):
        value = _constant(plane, name, atol)
        remaining = int(round(value * CASTLES_PER_PLAYER))
        if not 0 <= remaining <= CASTLES_PER_PLAYER or not np.isclose(
            value, remaining / CASTLES_PER_PLAYER, atol=atol
        ):
            raise ValueError(f"{name} plane is invalid")
        return remaining

    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if occupancy[0, y, x]:
                logic.board[y][x] = current
            elif occupancy[1, y, x]:
                logic.board[y][x] = opponent
            elif occupancy[2, y, x]:
                logic.board[y][x] = NEUTRAL
    logic.turn = current
    logic.consecutive_passes = passes
    logic.castles_remaining = {
        current: inventory(state[4], "current-inventory"),
        opponent: inventory(state[5], "opponent-inventory"),
    }
    if not np.allclose(encode_legacy_state(logic), state, rtol=0, atol=atol):
        raise ValueError("legacy replay decode/encode roundtrip mismatch")
    return logic
