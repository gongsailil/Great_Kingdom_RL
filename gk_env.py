import gymnasium as gym
from gymnasium import spaces
import numpy as np

from great_kingdom import GreatKingdomLogic, MoveResult, BOARD_SIZE


class GreatKingdomEnv(gym.Env):
    """Minimal experiment environment.

    Agent: Red (2, default) or Blue (1).
    Opponent: the other color, uniformly random over selectable moves.
    Masked actions: occupied cells + territory-forbidden cells only.
    Suicide remains selectable and is an immediate loss.
    """

    metadata = {"render_modes": []}

    def __init__(self, agent_player=2):
        super().__init__()
        if agent_player not in (1, 2):
            raise ValueError("agent_player must be Blue (1) or Red (2)")
        self.board_size = BOARD_SIZE
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3, self.board_size, self.board_size),
            dtype=np.uint8,
        )
        self.logic = GreatKingdomLogic()
        self.agent_player = agent_player
        self.opponent_player = 3 - agent_player
        self.agent_moves = 0
        self.first_agent_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.logic = GreatKingdomLogic()
        self.agent_moves = 0
        self.first_agent_action = None

        # Blue starts. For the default Red agent, Blue is the opponent and
        # makes the opening move. A Blue agent acts immediately after reset.
        if self.agent_player == 2:
            self._opponent_move_random()

        return self._get_obs(), {}

    def action_masks(self):
        """Mask only impossible actions.

        Required directly on the env for MaskablePPO + SubprocVecEnv.
        True = selectable, False = impossible.
        """
        mask = np.zeros(self.board_size * self.board_size, dtype=bool)
        for action in range(self.board_size * self.board_size):
            x = action % self.board_size
            y = action // self.board_size
            mask[action] = not self.logic.is_impossible_action(x, y)
        return mask

    def step(self, action):
        action = int(action)
        x = action % self.board_size
        y = action // self.board_size

        if self.logic.is_impossible_action(x, y):
            # This should never happen when masks are consumed correctly.
            return self._get_obs(), -1.0, True, False, {
                "outcome": "mask_violation",
                "winner": self.opponent_player,
            }

        if self.first_agent_action is None:
            self.first_agent_action = action
        self.agent_moves += 1

        result = self.logic.place_stone_detailed(x, y)

        if result == MoveResult.SUICIDE_LOSS:
            return self._get_obs(), -1.0, True, False, self._terminal_info("agent_suicide")

        if result == MoveResult.CAPTURE_WIN:
            return self._get_obs(), 1.0, True, False, self._terminal_info("agent_capture_win")

        if self.logic.game_over:
            return self._terminal_transition("agent_terminal")

        # Random opponent (Blue for a Red agent, Red for a Blue agent).
        opponent_result = self._opponent_move_random()

        if opponent_result == MoveResult.SUICIDE_LOSS:
            return self._get_obs(), 1.0, True, False, self._terminal_info("opponent_suicide")

        if opponent_result == MoveResult.CAPTURE_WIN:
            return self._get_obs(), -1.0, True, False, self._terminal_info("opponent_capture_win")

        if self.logic.game_over:
            return self._terminal_transition("score")

        # Sparse reward: no survival bonus.
        return self._get_obs(), 0.0, False, False, {}

    def _terminal_transition(self, outcome):
        if self.logic.winner == self.agent_player:
            reward = 1.0
        elif self.logic.winner == self.opponent_player:
            reward = -1.0
        else:
            reward = 0.0
        return self._get_obs(), reward, True, False, self._terminal_info(outcome)

    def _terminal_info(self, outcome):
        return {
            "outcome": outcome,
            "winner": self.logic.winner,
            "agent_moves": self.agent_moves,
            "first_agent_action": self.first_agent_action,
        }

    def _get_obs(self):
        return self._get_obs_for(self.agent_player)

    def _get_obs_for(self, player):
        """Return the canonical observation from either player's view."""
        if player not in (1, 2):
            raise ValueError("player must be Blue (1) or Red (2)")
        board = np.asarray(self.logic.board)
        my_stones = (board == player).astype(np.uint8)
        opp_stones = (board == 3 - player).astype(np.uint8)
        neutral = ((board == 0) | (board == 3)).astype(np.uint8)
        return np.stack([my_stones, opp_stones, neutral], axis=0)

    def _opponent_move_random(self):
        if self.logic.game_over:
            return self.logic.last_move_result

        playable = self.logic.get_playable_spots()
        if not playable:
            self.logic.check_game_end_simple()
            return self.logic.last_move_result

        idx = int(self.np_random.integers(len(playable)))
        x, y = playable[idx]
        return self.logic.place_stone_detailed(x, y)
