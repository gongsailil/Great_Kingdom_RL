from great_kingdom import GreatKingdomLogic, MoveResult


def test_occupied_is_impossible():
    g = GreatKingdomLogic()
    assert g.is_impossible_action(4, 4)
    assert g.place_stone_detailed(4, 4) == MoveResult.IMPOSSIBLE_OCCUPIED
    assert not g.game_over


def test_suicide_is_selectable_and_loses():
    g = GreatKingdomLogic()
    # Blue to move at center of four Red stones.
    g.board = [[0] * 9 for _ in range(9)]
    g.board[3][4] = 2
    g.board[4][3] = 2
    g.board[4][5] = 2
    g.board[5][4] = 2
    g.turn = 1

    assert not g.is_impossible_action(4, 4)
    result = g.place_stone_detailed(4, 4)
    assert result == MoveResult.SUICIDE_LOSS
    assert g.game_over
    assert g.winner == 2


def test_capture_wins():
    g = GreatKingdomLogic()
    g.board = [[0] * 9 for _ in range(9)]
    # Red stone at (4,4) has one liberty at (4,5). Blue closes it.
    g.board[4][4] = 2
    g.board[3][4] = 1
    g.board[4][3] = 1
    g.board[4][5] = 1
    g.turn = 1

    assert not g.is_impossible_action(4, 5)
    result = g.place_stone_detailed(4, 5)
    assert result == MoveResult.CAPTURE_WIN
    assert g.game_over
    assert g.winner == 1


if __name__ == "__main__":
    test_occupied_is_impossible()
    test_suicide_is_selectable_and_loses()
    test_capture_wins()
    print("logic tests: PASS")
