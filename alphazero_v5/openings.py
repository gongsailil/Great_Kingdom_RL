"""Deterministic legal openings and fixed Rules V2 diagnostic positions."""

import numpy as np

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE, RED, BOARD_SIZE, CASTLES_PER_PLAYER, NEUTRAL, PASS_ACTION,
    GreatKingdomLogicV2, MoveResultV2,
)


def apply_opening(actions):
    logic = GreatKingdomLogicV2()
    for ply, raw_action in enumerate(actions):
        action = int(raw_action)
        if not 0 <= action < PASS_ACTION:
            raise ValueError(f"opening ply {ply} is not a placement: {action}")
        if not action_mask_for_logic(logic, logic.turn)[action]:
            raise ValueError(f"opening ply {ply} is illegal: {action}")
        result = logic.apply_action(action)
        if result != MoveResultV2.NORMAL:
            raise ValueError(
                f"opening ply {ply} must be non-terminal NORMAL, got {result.name}"
            )
    colors = {cell for row in logic.board for cell in row if cell in (BLUE, RED)}
    if colors != {BLUE, RED}:
        raise ValueError("opening must contain a placement by both players")
    return logic


def generate_openings(count=32, seed=20260830):
    if int(count) <= 0:
        raise ValueError("opening count must be positive")
    rng = np.random.default_rng(seed)
    openings, seen = [], set()
    for _ in range(int(count) * 100):
        if len(openings) == int(count):
            break
        logic, actions = GreatKingdomLogicV2(), []
        for _ in range(int(rng.integers(2, 7))):
            candidates = [
                int(action)
                for action in np.flatnonzero(
                    action_mask_for_logic(logic, logic.turn)[:PASS_ACTION]
                )
                if logic.classify_placement(
                    logic.turn, int(action) % BOARD_SIZE, int(action) // BOARD_SIZE
                ) == MoveResultV2.NORMAL
            ]
            if not candidates:
                raise RuntimeError("opening has no nonterminal placement")
            action = int(rng.choice(candidates))
            if logic.apply_action(action) != MoveResultV2.NORMAL:
                raise RuntimeError("opening unexpectedly terminal")
            actions.append(action)
        key = tuple(tuple(row) for row in logic.board)
        if key not in seen:
            seen.add(key)
            openings.append({
                "opening_id": len(openings), "actions": actions,
                "resulting_turn": logic.turn,
            })
    if len(openings) != int(count):
        raise RuntimeError("could not create unique openings")
    return {"seed": int(seed), "count": int(count), "openings": openings}


def _empty_tactical_position():
    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    return logic


def _sync_inventory(logic):
    for player in (BLUE, RED):
        used = sum(cell == player for row in logic.board for cell in row)
        logic.castles_remaining[player] = CASTLES_PER_PLAYER - used
    return logic


def capture_tactical_position():
    logic = _empty_tactical_position()
    logic.board[1][1] = RED
    for x, y in ((0, 1), (1, 0), (2, 1)):
        logic.board[y][x] = BLUE
    return _sync_inventory(logic)


def defense_tactical_position():
    logic = _empty_tactical_position()
    logic.board[1][1] = BLUE
    for x, y in ((0, 1), (1, 0), (2, 1)):
        logic.board[y][x] = RED
    return _sync_inventory(logic)


def winning_pass_tactical_position():
    logic = _empty_tactical_position()
    for x, y in ((0, 1), (1, 1), (2, 0)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED
    logic.consecutive_passes = 1
    return _sync_inventory(logic)


def strategic_positions():
    positions = []
    for name, builder in (
        ("capture", capture_tactical_position),
        ("defense", defense_tactical_position),
    ):
        for player in (BLUE, RED):
            logic = builder()
            if player == RED:
                logic.board = [
                    [3 - cell if cell in (BLUE, RED) else cell for cell in row]
                    for row in logic.board
                ]
                logic.castles_remaining = {
                    BLUE: logic.castles_remaining[RED],
                    RED: logic.castles_remaining[BLUE],
                }
                logic.turn = RED
            positions.append((f"exact_{name}_{player}", logic, True))
    positions.append(("exact_winning_pass_blue", winning_pass_tactical_position(), True))
    red_pass = apply_opening([1, 80, 9])
    red_pass.consecutive_passes = 1
    positions.append(("exact_winning_pass_red", red_pass, True))
    for name, prefix in (
        ("established_territory", [2, 80, 11, 79, 20, 78, 18, 77, 19, 76]),
        ("blue_needs_more_territory", [1, 80, 9, 79]),
        ("red_scoring_advantage", [1, 80, 9]),
        ("own_territory_cost", [9, 80, 10, 79, 2, 78]),
    ):
        positions.append((name, apply_opening(prefix), False))
    return positions
