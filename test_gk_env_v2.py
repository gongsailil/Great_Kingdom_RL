import numpy as np

import gk_env_v2
from gk_env_v2 import GreatKingdomEnvV2, action_mask_for_logic
from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    NEUTRAL,
    NUM_ACTIONS,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
)


def action_at(x, y):
    return x + y * BOARD_SIZE


def set_edge_blue_territory(logic):
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    for x, y in ((2, 0), (2, 1), (2, 2), (0, 2), (1, 2)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED


def set_pure_suicide(logic):
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    for x, y in ((1, 0), (0, 1), (2, 1), (0, 2), (2, 2), (1, 3)):
        logic.board[y][x] = RED
    logic.board[2][1] = BLUE
    logic.turn = BLUE


def set_capture_priority(logic):
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    logic.board[0][1] = RED
    for x, y in ((0, 1), (1, 1), (2, 0)):
        logic.board[y][x] = BLUE
    for x, y in ((0, 2), (2, 1), (1, 2)):
        logic.board[y][x] = RED
    logic.turn = BLUE


def expect_exception(exception_type, function):
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_action_space_and_pass_index():
    env = GreatKingdomEnvV2()
    assert PASS_ACTION == 81
    assert NUM_ACTIONS == 82
    assert env.action_space.n == 82


def test_reset_observation_is_absolute_and_complete():
    env = GreatKingdomEnvV2()
    observation, info = env.reset(seed=20260830)
    assert info == {}
    assert env.observation_space.contains(observation)
    assert observation["board"].shape == (9, 9)
    assert observation["board"].dtype == np.uint8
    assert observation["board"][4, 4] == NEUTRAL
    assert observation["turn"] == BLUE
    assert observation["consecutive_passes"] == 0
    assert observation["castles_remaining"].tolist() == [40, 40]


def test_initial_mask_has_80_placements_plus_pass():
    env = GreatKingdomEnvV2()
    mask = env.action_masks()
    assert mask.shape == (82,)
    assert mask.dtype == bool
    assert not mask[action_at(4, 4)]
    assert mask[PASS_ACTION]
    assert int(mask[:PASS_ACTION].sum()) == 80


def test_player_dependent_opponent_and_own_territory_mask():
    logic = GreatKingdomLogicV2()
    set_edge_blue_territory(logic)
    point = action_at(0, 0)
    blue_mask = action_mask_for_logic(logic, BLUE)
    red_mask = action_mask_for_logic(logic, RED)
    assert blue_mask[point]
    assert not red_mask[point]
    assert blue_mask[PASS_ACTION] and red_mask[PASS_ACTION]


def test_pure_suicide_masked_but_capture_priority_unmasked():
    logic = GreatKingdomLogicV2()
    set_pure_suicide(logic)
    assert not action_mask_for_logic(logic, BLUE)[action_at(1, 1)]

    set_capture_priority(logic)
    assert action_mask_for_logic(logic, BLUE)[action_at(0, 0)]


def test_one_pass_step_and_placement_reset():
    env = GreatKingdomEnvV2()
    observation, reward, terminated, truncated, info = env.step(PASS_ACTION)
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["acting_player"] == BLUE
    assert info["move_result"] == "PASS"
    assert observation["turn"] == RED
    assert observation["consecutive_passes"] == 1
    assert observation["castles_remaining"].tolist() == [40, 40]

    observation, reward, terminated, truncated, info = env.step(action_at(0, 0))
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["acting_player"] == RED
    assert info["move_result"] == "NORMAL"
    assert observation["turn"] == BLUE
    assert observation["consecutive_passes"] == 0
    assert observation["castles_remaining"].tolist() == [40, 39]


def test_two_pass_steps_score_and_terminate():
    env = GreatKingdomEnvV2()
    env.step(PASS_ACTION)
    observation, reward, terminated, truncated, info = env.step(PASS_ACTION)
    assert terminated and not truncated
    assert reward == 0.0  # Rules-only env intentionally defines no RL reward.
    assert info["acting_player"] == RED
    assert info["move_result"] == "PASS_SCORE_END"
    assert info["winner"] == RED
    assert observation["consecutive_passes"] == 2
    assert not env.action_masks().any()


def test_zero_inventory_leaves_pass_as_only_action():
    env = GreatKingdomEnvV2()
    env.logic.castles_remaining[BLUE] = 0
    mask = env.action_masks(BLUE)
    assert not mask[:PASS_ACTION].any()
    assert mask[PASS_ACTION]

    _, _, terminated, _, _ = env.step(PASS_ACTION)
    assert not terminated
    env.logic.castles_remaining[RED] = 0
    red_mask = env.action_masks(RED)
    assert not red_mask[:PASS_ACTION].any()
    assert red_mask[PASS_ACTION]


def test_no_placement_does_not_auto_terminate():
    env = GreatKingdomEnvV2()
    env.logic.board = [[BLUE] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    env.logic.board[4][4] = NEUTRAL
    mask = env.action_masks(BLUE)
    assert not mask[:PASS_ACTION].any()
    assert mask[PASS_ACTION]
    _, _, terminated, _, info = env.step(PASS_ACTION)
    assert not terminated
    assert info["move_result"] == "PASS"


def test_illegal_or_out_of_range_step_fails_without_fallback():
    env = GreatKingdomEnvV2()
    expect_exception(ValueError, lambda: env.step(action_at(4, 4)))
    expect_exception(ValueError, lambda: env.step(NUM_ACTIONS))
    assert env.logic.turn == BLUE
    assert env.logic.castles_remaining[BLUE] == 40


def test_v2_env_has_no_v1_model_or_opponent_integration():
    assert not hasattr(gk_env_v2, "MaskablePPO")
    env = GreatKingdomEnvV2()
    assert not hasattr(env, "opponent_model")
    assert not hasattr(env, "agent_player")


if __name__ == "__main__":
    test_action_space_and_pass_index()
    test_reset_observation_is_absolute_and_complete()
    test_initial_mask_has_80_placements_plus_pass()
    test_player_dependent_opponent_and_own_territory_mask()
    test_pure_suicide_masked_but_capture_priority_unmasked()
    test_one_pass_step_and_placement_reset()
    test_two_pass_steps_score_and_terminate()
    test_zero_inventory_leaves_pass_as_only_action()
    test_no_placement_does_not_auto_terminate()
    test_illegal_or_out_of_range_step_fails_without_fallback()
    test_v2_env_has_no_v1_model_or_opponent_integration()
    print("GreatKingdomEnvV2 tests: PASS")
