"""Valid territory-rich starts sourced only from historical board states."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from alphazero_v3.value_oracle_audit import decode_v3_state
from alphazero_v3.encoder import encode_state as encode_v3_state
from great_kingdom_v2 import BLUE, RED, GreatKingdomLogicV2
from alphazero_v4.tactical import solve_tactical_root


@dataclass(frozen=True)
class StoredLogic:
    board: tuple
    turn: int
    consecutive_passes: int
    blue_remaining: int
    red_remaining: int


def store_logic(logic):
    return StoredLogic(tuple(tuple(int(cell) for cell in row) for row in logic.board),
                       logic.turn, logic.consecutive_passes,
                       logic.castles_remaining[BLUE], logic.castles_remaining[RED])


def restore_logic(stored):
    logic = GreatKingdomLogicV2()
    logic.board = [list(row) for row in stored.board]
    logic.turn = stored.turn
    logic.consecutive_passes = stored.consecutive_passes
    logic.castles_remaining = {BLUE: stored.blue_remaining, RED: stored.red_remaining}
    logic.game_over = False; logic.winner = None; logic.win_reason = ""
    logic.last_move_result = None; logic.score_blue = None; logic.score_red = None
    return logic


def validate_stored_logic(stored):
    logic = restore_logic(stored)
    for player in (BLUE, RED):
        stones = sum(cell == player for row in logic.board for cell in row)
        if stones + logic.castles_remaining[player] != 40:
            raise ValueError("stored midgame inventory mismatch")
        if stones == 0:
            raise ValueError("stored midgame lacks both colors")
    if logic.game_over or sum(logic.territory_counts().values()) == 0:
        raise ValueError("stored midgame is terminal or lacks territory")
    if solve_tactical_root(logic).mode == "IMMEDIATE_WIN":
        raise ValueError("stored midgame has an immediate terminal win")
    return logic


def load_v4_territory_pool(replay_path, maximum=4096, seed=20260830):
    replay_path = Path(replay_path)
    payload = torch.load(replay_path, map_location="cpu", weights_only=False)
    states = np.asarray(payload["states"], dtype=np.float32)
    order = np.random.default_rng(seed).permutation(len(states))
    pool, seen = [], set()
    for index in order:
        logic = decode_v3_state(states[index])
        if not np.allclose(encode_v3_state(logic), states[index], rtol=0, atol=1e-6):
            raise RuntimeError("historical V4 state failed roundtrip")
        if not all(any(cell == p for row in logic.board for cell in row) for p in (BLUE, RED)):
            continue
        if sum(logic.territory_counts().values()) == 0:
            continue
        if solve_tactical_root(logic).mode == "IMMEDIATE_WIN":
            continue
        stored = store_logic(logic)
        if stored in seen:
            continue
        validate_stored_logic(stored)
        seen.add(stored); pool.append(stored)
        if len(pool) >= int(maximum):
            break
    if not pool:
        raise RuntimeError("no valid territory-rich states found")
    return pool
