import numpy as np

from gk_env import GreatKingdomEnv
from great_kingdom import GreatKingdomLogic


def action_at(env, x, y):
    return x + y * env.board_size


def reset_position(env, board, turn):
    env.logic = GreatKingdomLogic()
    env.logic.board = board
    env.logic.turn = turn
    env.logic.game_over = False
    env.logic.winner = None
    env.logic.win_reason = ""
    env.logic.last_move_result = None
    env.agent_moves = 0
    env.first_agent_action = None


def test_red_observation_semantics():
    env = GreatKingdomEnv(agent_player=2)
    env.logic.board = [[0] * 9 for _ in range(9)]
    env.logic.board[0][0] = 1
    env.logic.board[0][1] = 2
    env.logic.board[4][4] = 3

    obs = env._get_obs()
    assert obs.dtype == np.uint8
    assert obs[0, 0, 1] == 1  # Red/agent
    assert obs[1, 0, 0] == 1  # Blue/opponent
    assert obs[2, 1, 1] == 1  # Empty
    assert obs[2, 4, 4] == 1  # Neutral
    assert obs[:, 0, 0].tolist() == [0, 1, 0]
    assert obs[:, 0, 1].tolist() == [1, 0, 0]


def test_blue_observation_semantics():
    env = GreatKingdomEnv(agent_player=1)
    env.logic.board = [[0] * 9 for _ in range(9)]
    env.logic.board[0][0] = 1
    env.logic.board[0][1] = 2
    env.logic.board[4][4] = 3

    obs = env._get_obs()
    assert obs.dtype == np.uint8
    assert obs[0, 0, 0] == 1  # Blue/agent
    assert obs[1, 0, 1] == 1  # Red/opponent
    assert obs[2, 1, 1] == 1  # Empty
    assert obs[2, 4, 4] == 1  # Neutral
    assert obs[:, 0, 0].tolist() == [1, 0, 0]
    assert obs[:, 0, 1].tolist() == [0, 1, 0]


def test_red_reset_has_blue_opening():
    env = GreatKingdomEnv(agent_player=2)
    obs, info = env.reset(seed=1234)
    board = np.asarray(env.logic.board)

    assert info == {}
    assert np.count_nonzero(board == 1) == 1
    assert np.count_nonzero(board == 2) == 0
    assert env.logic.turn == 2
    assert int(obs[1].sum()) == 1


def test_blue_reset_waits_for_agent_opening():
    env = GreatKingdomEnv(agent_player=1)
    obs, info = env.reset(seed=1234)
    board = np.asarray(env.logic.board)

    assert info == {}
    assert np.count_nonzero(board == 1) == 0
    assert np.count_nonzero(board == 2) == 0
    assert board[4, 4] == 3
    assert env.logic.turn == 1
    assert int(obs[0].sum()) == 0
    assert int(obs[1].sum()) == 0
    assert obs[2, 4, 4] == 1


def test_blue_occupied_and_territory_masks():
    env = GreatKingdomEnv(agent_player=1)
    env.reset(seed=1)
    assert not env.action_masks()[action_at(env, 4, 4)]

    board = [[3] * 9 for _ in range(9)]
    board[4][4] = 0
    board[3][4] = 2
    board[4][3] = 2
    board[4][5] = 2
    board[5][4] = 2
    board[0][0] = 1
    reset_position(env, board, turn=1)

    assert env.logic.get_territory_owner(4, 4) == 2
    assert not env.action_masks()[action_at(env, 4, 4)]


def test_blue_suicide_is_selectable_and_immediate_loss():
    env = GreatKingdomEnv(agent_player=1)
    board = [[0] * 9 for _ in range(9)]
    board[3][4] = 2
    board[4][3] = 2
    board[4][5] = 2
    board[5][4] = 2
    reset_position(env, board, turn=1)
    action = action_at(env, 4, 4)

    assert env.action_masks()[action]
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    assert reward == -1.0
    assert info["outcome"] == "agent_suicide"
    assert info["winner"] == 2


def test_blue_capture_is_immediate_win():
    env = GreatKingdomEnv(agent_player=1)
    board = [[0] * 9 for _ in range(9)]
    board[4][4] = 2
    board[3][4] = 1
    board[4][3] = 1
    board[4][5] = 1
    reset_position(env, board, turn=1)
    action = action_at(env, 4, 5)

    assert env.action_masks()[action]
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    assert reward == 1.0
    assert info["outcome"] == "agent_capture_win"
    assert info["winner"] == 1


def test_existing_red_action_mask_semantics():
    env = GreatKingdomEnv()
    assert env.agent_player == 2
    env.reset(seed=1)
    assert not env.action_masks()[action_at(env, 4, 4)]

    board = [[3] * 9 for _ in range(9)]
    board[4][4] = 0
    board[3][4] = 1
    board[4][3] = 1
    board[4][5] = 1
    board[5][4] = 1
    board[0][0] = 2
    reset_position(env, board, turn=2)
    assert env.logic.get_territory_owner(4, 4) == 1
    assert not env.action_masks()[action_at(env, 4, 4)]

    board = [[0] * 9 for _ in range(9)]
    board[3][4] = 1
    board[4][3] = 1
    board[4][5] = 1
    board[5][4] = 1
    reset_position(env, board, turn=2)
    assert env.action_masks()[action_at(env, 4, 4)]


def test_invalid_agent_player_is_rejected():
    for player in (0, 3):
        try:
            GreatKingdomEnv(agent_player=player)
        except ValueError:
            pass
        else:
            raise AssertionError(f"agent_player={player} should be rejected")


if __name__ == "__main__":
    test_red_observation_semantics()
    test_blue_observation_semantics()
    test_red_reset_has_blue_opening()
    test_blue_reset_waits_for_agent_opening()
    test_blue_occupied_and_territory_masks()
    test_blue_suicide_is_selectable_and_immediate_loss()
    test_blue_capture_is_immediate_win()
    test_existing_red_action_mask_semantics()
    test_invalid_agent_player_is_rejected()
    print("both-player env tests: PASS")
