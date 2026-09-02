"""Nine-plane canonical encoder with Rules V2 territory ownership."""

import numpy as np

from alphazero_v2.encoder import encode_state as encode_v2_state
from great_kingdom_v2 import BOARD_SIZE


NUM_PLANES = 9
ENCODED_SHAPE = (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)


def encode_state(logic):
    """Append current/opponent territory masks to the seven V2 planes."""
    encoded = np.zeros(ENCODED_SHAPE, dtype=np.float32)
    encoded[:7] = encode_v2_state(logic)
    current = logic.turn
    opponent = 3 - current
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
