import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from great_kingdom_v2 import (
    BLUE,
    BLUE_REQUIRED_TERRITORY_LEAD,
    BOARD_SIZE,
    CASTLES_PER_PLAYER,
    NEUTRAL,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
    determine_scoring_winner,
)


def empty_position(logic):
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    logic.turn = BLUE
    logic.game_over = False
    logic.winner = None
    logic.win_reason = ""
    logic.last_move_result = None
    logic.consecutive_passes = 0
    logic.castles_remaining = {
        BLUE: CASTLES_PER_PLAYER,
        RED: CASTLES_PER_PLAYER,
    }
    logic.score_blue = None
    logic.score_red = None


def edge_blue_territory_position():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    for x, y in ((2, 0), (2, 1), (2, 2), (0, 2), (1, 2)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED
    return logic


def pure_suicide_position():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    for x, y in ((1, 0), (0, 1), (2, 1), (0, 2), (2, 2), (1, 3)):
        logic.board[y][x] = RED
    logic.board[2][1] = BLUE
    logic.turn = BLUE
    return logic


def mutual_last_liberty_position():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    # X=(0,0) is the last liberty of both the Red castle at (1,0) and
    # the connected Blue group at (0,1)/(1,1). Capture must win first.
    logic.board[0][1] = RED
    for x, y in ((0, 1), (1, 1), (2, 0)):
        logic.board[y][x] = BLUE
    for x, y in ((0, 2), (2, 1), (1, 2)):
        logic.board[y][x] = RED
    logic.turn = BLUE
    return logic


def two_point_blue_territory_position():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    for x, y in ((0, 1), (1, 1), (2, 0)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED
    return logic


def one_point_blue_territory_position():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    logic.board[0][1] = BLUE
    logic.board[1][0] = BLUE
    logic.board[8][8] = RED
    return logic


def test_initial_state_and_copy():
    logic = GreatKingdomLogicV2()
    assert len(logic.board) == len(logic.board[0]) == BOARD_SIZE == 9
    assert logic.board[4][4] == NEUTRAL
    assert logic.turn == BLUE
    assert logic.consecutive_passes == 0
    assert logic.castles_remaining == {BLUE: 40, RED: 40}
    assert not hasattr(logic, "komi")

    logic.consecutive_passes = 1
    logic.castles_remaining[BLUE] = 39
    logic.score_blue = 2
    logic.score_red = 0
    copied = logic.copy()
    assert copied.consecutive_passes == 1
    assert copied.castles_remaining == {BLUE: 39, RED: 40}
    assert copied.score_blue == 2 and copied.score_red == 0
    copied.board[0][0] = BLUE
    copied.castles_remaining[BLUE] -= 1
    copied.consecutive_passes = 2
    assert logic.board[0][0] == 0
    assert logic.castles_remaining[BLUE] == 39
    assert logic.consecutive_passes == 1


def test_board_edge_bounds_territory():
    logic = edge_blue_territory_position()
    expected_region = {(0, 0), (1, 0), (0, 1), (1, 1)}
    for x, y in expected_region:
        assert logic.get_territory_owner(x, y) == BLUE
    assert logic.count_territory(BLUE) == 4


def test_neutral_castle_can_bound_territory():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    for x, y in ((3, 3), (5, 3), (4, 2)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED
    assert logic.board[4][4] == NEUTRAL
    assert logic.get_territory_owner(4, 3) == BLUE


def test_opponent_inside_region_prevents_territory():
    logic = edge_blue_territory_position()
    logic.board[1][1] = RED
    assert logic.get_territory_owner(0, 0) == 0


def test_both_player_boundaries_make_region_unclaimed():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    logic.board[0][3] = BLUE
    logic.board[0][5] = RED
    logic.board[1][4] = BLUE
    assert logic.get_territory_owner(4, 0) == 0


def test_opening_board_is_not_giant_territory():
    logic = GreatKingdomLogicV2()
    logic.board[0][0] = BLUE
    assert not any(RED in row for row in logic.board)
    assert logic.get_territory_owner(1, 0) == 0
    assert logic.count_territory(BLUE) == 0


def test_opponent_territory_blocked_and_own_territory_allowed():
    logic = edge_blue_territory_position()
    logic.turn = RED
    logic.consecutive_passes = 1
    before_board = [row[:] for row in logic.board]
    before_inventory = dict(logic.castles_remaining)
    assert logic.classify_placement(RED, 0, 0) == (
        MoveResultV2.IMPOSSIBLE_OPPONENT_TERRITORY
    )
    assert logic.place_stone_detailed(0, 0) == (
        MoveResultV2.IMPOSSIBLE_OPPONENT_TERRITORY
    )
    assert logic.board == before_board
    assert logic.turn == RED
    assert logic.castles_remaining == before_inventory
    assert logic.consecutive_passes == 1
    assert logic.classify_placement(BLUE, 0, 0) == MoveResultV2.NORMAL


def test_own_territory_placement_reduces_empty_territory_count():
    logic = edge_blue_territory_position()
    before = logic.count_territory(BLUE)
    before_inventory = logic.castles_remaining[BLUE]

    result = logic.place_stone_detailed(0, 0)

    assert result == MoveResultV2.NORMAL
    assert logic.board[0][0] == BLUE
    assert logic.count_territory(BLUE) == before - 1
    assert logic.castles_remaining[BLUE] == before_inventory - 1


def test_pure_suicide_is_illegal_and_rolls_back_everything():
    logic = pure_suicide_position()
    before_board = [row[:] for row in logic.board]
    before_inventory = dict(logic.castles_remaining)
    before_turn = logic.turn
    logic.consecutive_passes = 1

    assert logic.get_territory_owner(1, 1) == 0
    assert logic.classify_placement(BLUE, 1, 1) == (
        MoveResultV2.IMPOSSIBLE_SUICIDE
    )
    result = logic.place_stone_detailed(1, 1)

    assert result == MoveResultV2.IMPOSSIBLE_SUICIDE
    assert logic.board == before_board
    assert logic.castles_remaining == before_inventory
    assert logic.turn == before_turn
    assert logic.consecutive_passes == 1
    assert not logic.game_over


def test_blue_capture_is_immediate_win():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    logic.board[1][1] = RED
    for x, y in ((0, 1), (1, 0), (2, 1)):
        logic.board[y][x] = BLUE
    before_inventory = logic.castles_remaining[BLUE]

    result = logic.place_stone_detailed(1, 2)

    assert result == MoveResultV2.CAPTURE_WIN
    assert logic.game_over and logic.winner == BLUE
    assert logic.turn == BLUE
    assert logic.castles_remaining[BLUE] == before_inventory - 1
    assert logic.score_blue is logic.score_red is None
    assert "capture" in logic.win_reason


def test_red_capture_is_immediate_win():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    logic.turn = RED
    logic.board[1][1] = BLUE
    for x, y in ((0, 1), (1, 0), (2, 1)):
        logic.board[y][x] = RED

    result = logic.place_stone_detailed(1, 2)

    assert result == MoveResultV2.CAPTURE_WIN
    assert logic.game_over and logic.winner == RED
    assert logic.turn == RED


def test_connected_group_capture_is_terminal():
    logic = GreatKingdomLogicV2()
    empty_position(logic)
    logic.board[1][1] = RED
    logic.board[1][2] = RED
    for x, y in ((1, 0), (2, 0), (0, 1), (3, 1), (1, 2)):
        logic.board[y][x] = BLUE

    assert logic.count_liberties(1, 1, RED) == 1
    result = logic.place_stone_detailed(2, 2)

    assert result == MoveResultV2.CAPTURE_WIN
    assert logic.game_over and logic.winner == BLUE


def test_capture_priority_over_apparent_zero_self_liberty():
    logic = mutual_last_liberty_position()
    assert logic.get_territory_owner(0, 0) == 0
    assert logic.classify_placement(BLUE, 0, 0) == MoveResultV2.CAPTURE_WIN

    result = logic.place_stone_detailed(0, 0)

    assert result == MoveResultV2.CAPTURE_WIN
    assert logic.game_over and logic.winner == BLUE
    assert logic.count_liberties(0, 0, BLUE) == 0


def test_capture_terminal_allows_no_followup_pass_or_score():
    logic = mutual_last_liberty_position()
    logic.place_stone_detailed(0, 0)
    original_reason = logic.win_reason

    assert logic.pass_turn() == MoveResultV2.IMPOSSIBLE_GAME_OVER
    assert logic.winner == BLUE
    assert logic.win_reason == original_reason
    assert logic.score_blue is logic.score_red is None


def test_one_pass_switches_turn_without_inventory_cost():
    logic = GreatKingdomLogicV2()
    before = dict(logic.castles_remaining)

    result = logic.pass_turn()

    assert result == MoveResultV2.PASS
    assert logic.turn == RED
    assert logic.consecutive_passes == 1
    assert logic.castles_remaining == before
    assert not logic.game_over


def test_placement_resets_pass_count():
    logic = GreatKingdomLogicV2()
    assert logic.pass_turn() == MoveResultV2.PASS
    assert logic.turn == RED

    assert logic.place_stone_detailed(0, 0) == MoveResultV2.NORMAL
    assert logic.consecutive_passes == 0
    assert logic.turn == BLUE


def test_two_consecutive_passes_end_by_score():
    logic = GreatKingdomLogicV2()
    assert logic.pass_turn() == MoveResultV2.PASS
    assert logic.pass_turn() == MoveResultV2.PASS_SCORE_END
    assert logic.game_over
    assert logic.consecutive_passes == 2
    assert logic.winner == RED
    assert logic.score_blue == logic.score_red == 0
    assert logic.win_reason == "Blue territory = 0; Red territory = 0; Winner = Red"


def test_inventory_start_decrement_pass_and_illegal_semantics():
    logic = GreatKingdomLogicV2()
    assert logic.castles_remaining == {BLUE: 40, RED: 40}
    assert logic.place_stone_detailed(0, 0) == MoveResultV2.NORMAL
    assert logic.castles_remaining == {BLUE: 39, RED: 40}
    assert logic.pass_turn() == MoveResultV2.PASS
    assert logic.castles_remaining == {BLUE: 39, RED: 40}

    logic = pure_suicide_position()
    before = dict(logic.castles_remaining)
    assert logic.place_stone_detailed(1, 1) == MoveResultV2.IMPOSSIBLE_SUICIDE
    assert logic.castles_remaining == before


def test_zero_inventory_disables_placement_but_not_pass():
    logic = GreatKingdomLogicV2()
    logic.castles_remaining[BLUE] = 0
    assert logic.classify_placement(BLUE, 0, 0) == (
        MoveResultV2.IMPOSSIBLE_NO_CASTLES
    )
    assert logic.pass_turn() == MoveResultV2.PASS


def test_documented_scoring_threshold_has_no_draw():
    assert BLUE_REQUIRED_TERRITORY_LEAD == 2
    assert determine_scoring_winner(2, 0) == BLUE
    assert determine_scoring_winner(3, 1) == BLUE
    assert determine_scoring_winner(1, 0) == RED
    assert determine_scoring_winner(5, 5) == RED


def test_scoring_integration_at_and_below_threshold():
    blue_wins = two_point_blue_territory_position()
    assert blue_wins.territory_counts() == {BLUE: 2, RED: 0}
    blue_wins.pass_turn()
    blue_wins.pass_turn()
    assert blue_wins.winner == BLUE
    assert blue_wins.win_reason == (
        "Blue territory = 2; Red territory = 0; Winner = Blue"
    )

    red_wins = one_point_blue_territory_position()
    assert red_wins.territory_counts() == {BLUE: 1, RED: 0}
    red_wins.pass_turn()
    red_wins.pass_turn()
    assert red_wins.winner == RED
    assert red_wins.win_reason == (
        "Blue territory = 1; Red territory = 0; Winner = Red"
    )


def test_no_ko_or_repetition_state_exists():
    logic = GreatKingdomLogicV2()
    forbidden_attributes = (
        "ko",
        "ko_point",
        "previous_board",
        "board_history",
        "position_history",
        "superko",
    )
    for attribute in forbidden_attributes:
        assert not hasattr(logic, attribute)
        assert not hasattr(logic.copy(), attribute)


def test_human_v2_dummy_draw_shutdown_smoke():
    from play_human_v2 import HumanVsHumanV2UI

    ui = HumanVsHumanV2UI()
    assert ui.logic.turn == BLUE
    assert ui.logic.castles_remaining == {BLUE: 40, RED: 40}
    ui.run(max_frames=2)


def test_human_v2_illegal_move_messages():
    from play_human_v2 import HumanVsHumanV2UI

    ui = object.__new__(HumanVsHumanV2UI)
    ui.renderer = None
    ui.logic = edge_blue_territory_position()
    ui.logic.turn = RED
    ui.info_message = ""
    assert ui.play_at(0, 0) == MoveResultV2.IMPOSSIBLE_OPPONENT_TERRITORY
    assert "opponent's territory" in ui.info_message

    ui.logic = pure_suicide_position()
    ui.info_message = ""
    assert ui.play_at(1, 1) == MoveResultV2.IMPOSSIBLE_SUICIDE
    assert "pure suicide" in ui.info_message


if __name__ == "__main__":
    test_initial_state_and_copy()
    test_board_edge_bounds_territory()
    test_neutral_castle_can_bound_territory()
    test_opponent_inside_region_prevents_territory()
    test_both_player_boundaries_make_region_unclaimed()
    test_opening_board_is_not_giant_territory()
    test_opponent_territory_blocked_and_own_territory_allowed()
    test_own_territory_placement_reduces_empty_territory_count()
    test_pure_suicide_is_illegal_and_rolls_back_everything()
    test_blue_capture_is_immediate_win()
    test_red_capture_is_immediate_win()
    test_connected_group_capture_is_terminal()
    test_capture_priority_over_apparent_zero_self_liberty()
    test_capture_terminal_allows_no_followup_pass_or_score()
    test_one_pass_switches_turn_without_inventory_cost()
    test_placement_resets_pass_count()
    test_two_consecutive_passes_end_by_score()
    test_inventory_start_decrement_pass_and_illegal_semantics()
    test_zero_inventory_disables_placement_but_not_pass()
    test_documented_scoring_threshold_has_no_draw()
    test_scoring_integration_at_and_below_threshold()
    test_no_ko_or_repetition_state_exists()
    test_human_v2_dummy_draw_shutdown_smoke()
    test_human_v2_illegal_move_messages()
    print("Rules V2 tests: PASS")
