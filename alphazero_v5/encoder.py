"""Twelve-plane canonical V5 encoder derived exclusively from Rules V2."""

import numpy as np

from alphazero_v3.encoder import encode_state as encode_v3_state
from great_kingdom_v2 import BLUE, RED, BOARD_SIZE


NUM_PLANES = 12
ENCODED_SHAPE = (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
MAX_ENCODED_LIBERTIES = 8
BLUE_REQUIRED_LEAD = 2.0


def scoring_margin(logic):
    """Current-player territory margin after the asymmetric Blue +2 threshold."""
    counts = logic.territory_counts()
    blue_margin = counts[BLUE] - counts[RED] - BLUE_REQUIRED_LEAD
    current_margin = blue_margin if logic.turn == BLUE else -blue_margin
    return float(np.clip(current_margin / (BOARD_SIZE * BOARD_SIZE), -1.0, 1.0))


def _liberty_planes(logic):
    current, opponent = logic.turn, 3 - logic.turn
    planes = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    visited = set()
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            player = logic.board[y][x]
            if player not in (BLUE, RED) or (x, y) in visited:
                continue
            group = logic.get_group(x, y)
            visited.update(group)
            normalized = min(logic.count_liberties(x, y, player), MAX_ENCODED_LIBERTIES) / MAX_ENCODED_LIBERTIES
            plane = 0 if player == current else 1
            for gx, gy in group:
                planes[plane, gy, gx] = normalized
    return planes


def encode_state(logic):
    if logic.game_over:
        raise ValueError("V5 encoder expects an active Rules V2 state")
    encoded = np.zeros(ENCODED_SHAPE, dtype=np.float32)
    encoded[:9] = encode_v3_state(logic)
    encoded[9:11] = _liberty_planes(logic)
    encoded[11].fill(scoring_margin(logic))
    return encoded


def decode_state(state, atol=5e-4):
    """Diagnostic/start-pool decoder; derived V5 planes only validate state."""
    from alphazero_v3.value_oracle_audit import decode_v3_state
    state = np.asarray(state, dtype=np.float32)
    if state.shape != ENCODED_SHAPE:
        raise ValueError("V5 state must have shape (12,9,9)")
    logic = decode_v3_state(state[:9], atol=atol)
    rebuilt = encode_state(logic)
    if not np.allclose(rebuilt, state, rtol=0, atol=atol):
        raise ValueError("V5 decode/encode roundtrip mismatch")
    return logic
