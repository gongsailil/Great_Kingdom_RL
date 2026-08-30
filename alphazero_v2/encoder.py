"""Canonical current-player state encoding for Rules V2."""

import numpy as np

from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    CASTLES_PER_PLAYER,
    NEUTRAL,
)


NUM_PLANES = 7
ENCODED_SHAPE = (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)


def encode_state(logic):
    """Encode all Markov state needed by the minimal shared network.

    Planes are current castles, opponent castles, neutral castle,
    consecutive_passes/2, current inventory/40, opponent inventory/40, and
    absolute color (1 for Blue-to-play, 0 for Red-to-play).
    """
    current = logic.turn
    opponent = 3 - current
    board = np.asarray(logic.board)
    encoded = np.empty(ENCODED_SHAPE, dtype=np.float32)
    encoded[0] = board == current
    encoded[1] = board == opponent
    encoded[2] = board == NEUTRAL
    encoded[3].fill(logic.consecutive_passes / 2.0)
    encoded[4].fill(
        logic.castles_remaining[current] / float(CASTLES_PER_PLAYER)
    )
    encoded[5].fill(
        logic.castles_remaining[opponent] / float(CASTLES_PER_PLAYER)
    )
    encoded[6].fill(1.0 if current == BLUE else 0.0)
    return encoded
