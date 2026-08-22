import numpy as np

from frozen_policy_env import FrozenPolicyOpponentEnv
from great_kingdom import GreatKingdomLogic, MoveResult


BLUE_MODEL = "models/MaskablePPO_CNN/blue_masked_ppo_10000.zip"
RED1_MODEL = "models/MaskablePPO_CNN/red10k_ft_vs_blue10k_plus10k.zip"


class ScriptedModel:
    def __init__(self, action):
        self.action = action
        self.mask_seen = False
        self.last_obs = None

    def predict(self, obs, action_masks, deterministic):
        assert obs.shape == (3, 9, 9)
        assert bool(action_masks[self.action])
        self.mask_seen = True
        self.last_obs = np.array(obs, copy=True)
        return np.asarray(self.action), None


def set_position(env, board, turn):
    env.logic = GreatKingdomLogic()
    env.logic.board = board
    env.logic.turn = turn
    env.logic.game_over = False
    env.logic.winner = None
    env.logic.win_reason = ""
    env.logic.last_move_result = None
    env.move_trace = []
    env.opponent_mask_violations = 0


def test_observation_for_each_player():
    env = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=BLUE_MODEL,
        opponent_deterministic=True,
    )
    env.logic.board = [[0] * 9 for _ in range(9)]
    env.logic.board[0][0] = 1
    env.logic.board[0][1] = 2
    env.logic.board[4][4] = 3

    blue_obs = env._get_obs_for(1)
    red_obs = env._get_obs_for(2)
    assert blue_obs[:, 0, 0].tolist() == [1, 0, 0]
    assert blue_obs[:, 0, 1].tolist() == [0, 1, 0]
    assert red_obs[:, 0, 0].tolist() == [0, 1, 0]
    assert red_obs[:, 0, 1].tolist() == [1, 0, 0]
    assert blue_obs[2, 4, 4] == red_obs[2, 4, 4] == 1


def test_deterministic_frozen_blue_opens_and_uses_mask():
    env = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=BLUE_MODEL,
        opponent_deterministic=True,
    )
    obs, _ = env.reset(seed=1234)
    board = np.asarray(env.logic.board)

    assert obs.shape == (3, 9, 9)
    assert np.count_nonzero(board == 1) == 1
    assert np.count_nonzero(board == 2) == 0
    assert env.logic.turn == 2
    assert len(env.move_trace) == 1
    assert env.move_trace[0]["player"] == 1
    assert env.move_trace[0]["move_result"] == "NORMAL"
    assert env.opponent_mask_violations == 0


def test_frozen_opponent_suicide_is_selectable_and_loses():
    env = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=BLUE_MODEL,
        opponent_deterministic=False,
    )
    board = [[0] * 9 for _ in range(9)]
    board[3][4] = 2
    board[4][3] = 2
    board[4][5] = 2
    board[5][4] = 2
    set_position(env, board, turn=1)
    env.opponent_model = ScriptedModel(4 + 4 * 9)

    result = env._opponent_move_random()
    assert env.opponent_model.mask_seen
    assert result == MoveResult.SUICIDE_LOSS
    assert env.logic.game_over
    assert env.logic.winner == 2
    assert env.opponent_mask_violations == 0


def test_frozen_opponent_capture_is_immediate_win():
    env = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=BLUE_MODEL,
        opponent_deterministic=False,
    )
    board = [[0] * 9 for _ in range(9)]
    board[4][4] = 2
    board[3][4] = 1
    board[4][3] = 1
    board[4][5] = 1
    set_position(env, board, turn=1)
    env.opponent_model = ScriptedModel(4 + 5 * 9)

    result = env._opponent_move_random()
    assert env.opponent_model.mask_seen
    assert result == MoveResult.CAPTURE_WIN
    assert env.logic.game_over
    assert env.logic.winner == 1
    assert env.opponent_mask_violations == 0


def test_stochastic_opponent_seed_is_reproducible():
    first = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=BLUE_MODEL,
        opponent_deterministic=False,
        opponent_seed=777,
    )
    second = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=BLUE_MODEL,
        opponent_deterministic=False,
        opponent_seed=777,
    )
    first.reset(seed=888)
    second.reset(seed=888)
    assert first.move_trace[0]["action"] == second.move_trace[0]["action"]


def test_blue_learner_and_frozen_red_perspectives_and_turn_order():
    env = FrozenPolicyOpponentEnv(
        agent_player=1,
        opponent_model_path=RED1_MODEL,
        opponent_deterministic=False,
    )
    obs, _ = env.reset(seed=1234)
    assert env.agent_player == 1
    assert env.opponent_player == 2
    assert env.logic.turn == 1
    assert env.move_trace == []
    assert int(obs[0].sum()) == 0  # Blue learner stones
    assert int(obs[1].sum()) == 0  # Red opponent stones

    scripted_red = ScriptedModel(1)
    env.opponent_model = scripted_red
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == 0.0
    assert not terminated and not truncated
    assert scripted_red.mask_seen
    assert scripted_red.last_obs[:, 0, 0].tolist() == [0, 1, 0]
    assert env.logic.board[0][0] == 1
    assert env.logic.board[0][1] == 2
    assert env.logic.turn == 1
    assert [move["player"] for move in env.move_trace] == [1, 2]
    assert env.opponent_mask_violations == 0


def test_frozen_red_suicide_is_selectable_and_loses():
    env = FrozenPolicyOpponentEnv(
        agent_player=1,
        opponent_model_path=RED1_MODEL,
        opponent_deterministic=False,
    )
    board = [[0] * 9 for _ in range(9)]
    board[3][4] = 1
    board[4][3] = 1
    board[4][5] = 1
    board[5][4] = 1
    set_position(env, board, turn=2)
    env.opponent_model = ScriptedModel(4 + 4 * 9)

    result = env._opponent_move_random()
    assert env.opponent_model.mask_seen
    assert result == MoveResult.SUICIDE_LOSS
    assert env.logic.game_over
    assert env.logic.winner == 1
    assert env.opponent_mask_violations == 0


def test_frozen_red_capture_is_immediate_win():
    env = FrozenPolicyOpponentEnv(
        agent_player=1,
        opponent_model_path=RED1_MODEL,
        opponent_deterministic=False,
    )
    board = [[0] * 9 for _ in range(9)]
    board[4][4] = 1
    board[3][4] = 2
    board[4][3] = 2
    board[4][5] = 2
    set_position(env, board, turn=2)
    env.opponent_model = ScriptedModel(4 + 5 * 9)

    result = env._opponent_move_random()
    assert env.opponent_model.mask_seen
    assert result == MoveResult.CAPTURE_WIN
    assert env.logic.game_over
    assert env.logic.winner == 2
    assert env.opponent_mask_violations == 0


def test_frozen_red_normal_last_move_triggers_score_end():
    env = FrozenPolicyOpponentEnv(
        agent_player=1,
        opponent_model_path=RED1_MODEL,
        opponent_deterministic=False,
    )
    board = [
        [0, 1, 1, 2, 0, 0, 2, 1, 0],
        [1, 1, 1, 2, 2, 2, 2, 1, 0],
        [1, 2, 2, 2, 0, 0, 2, 1, 1],
        [1, 2, 2, 2, 0, 2, 2, 2, 2],
        [2, 2, 2, 0, 3, 1, 0, 1, 1],
        [0, 2, 2, 2, 1, 1, 2, 2, 1],
        [0, 2, 1, 1, 1, 0, 1, 1, 0],
        [2, 2, 2, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 0, 0, 0, 0, 1, 0],
    ]
    set_position(env, board, turn=2)
    env.opponent_model = ScriptedModel(6 + 4 * 9)
    assert env.logic.get_playable_spots() == [(6, 4)]

    result = env._opponent_move_random()
    assert result == MoveResult.NORMAL
    assert env.logic.game_over
    assert env.logic.winner in (0, 1, 2)
    assert env.logic.get_playable_spots() == []
    assert env.opponent_mask_violations == 0


if __name__ == "__main__":
    test_observation_for_each_player()
    test_deterministic_frozen_blue_opens_and_uses_mask()
    test_frozen_opponent_suicide_is_selectable_and_loses()
    test_frozen_opponent_capture_is_immediate_win()
    test_stochastic_opponent_seed_is_reproducible()
    test_blue_learner_and_frozen_red_perspectives_and_turn_order()
    test_frozen_red_suicide_is_selectable_and_loses()
    test_frozen_red_capture_is_immediate_win()
    test_frozen_red_normal_last_move_triggers_score_end()
    print("frozen-policy env tests: PASS")
