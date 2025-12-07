import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from great_kingdom import GreatKingdomLogic, BOARD_SIZE


class GreatKingdomEnv(gym.Env):
    def __init__(self):
        super(GreatKingdomEnv, self).__init__()
        self.board_size = BOARD_SIZE

        # Action: 0~80
        self.action_space = spaces.Discrete(self.board_size * self.board_size)

        # Observation: (3, 9, 9) 3-Channel Image Format
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.board_size, self.board_size), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.logic = GreatKingdomLogic()
        # AI (Red) Post -> Place Blue first
        self._opponent_move_smart()
        return self._get_obs(), {}

    def step(self, action):
        x = int(action % self.board_size)
        y = int(action // self.board_size)

        # 1. AI's Placement
        valid = self.logic.place_stone(x, y)
        if not valid:
            # Foul play (for early learning)
            return self._get_obs(), -5, True, False, {}

        # 2. Check AI's win
        if self.logic.game_over:
            reward = 20 if self.logic.winner == 2 else -20
            return self._get_obs(), reward, True, False, {}

        # 3. Opponent's Placement
        self._opponent_move_smart()

        # 4. Check Opponent's win (AI's Defeat)
        if self.logic.game_over:
            reward = 20 if self.logic.winner == 2 else -20
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), 0.1, False, False, {}

    def _get_obs(self):
        board = np.array(self.logic.board)
        # Split into 3-channels: [My stone (2), Opponent's stone (1), blank/neutral (0,3)]
        my_stones = (board == 2).astype(np.uint8)
        opp_stones = (board == 1).astype(np.uint8)
        neutral = ((board == 0) | (board == 3)).astype(np.uint8)
        return np.stack([my_stones, opp_stones, neutral], axis=0)

    def _opponent_move_smart(self):
        if self.logic.game_over: return
        empty_spots = self.logic.get_empty_spots()
        if not empty_spots:
            self.logic.check_game_end_simple()
            return

        # 1. If there is a way to win, it will do it unconditionally
        for ex, ey in empty_spots:
            test_game = self.logic.copy()
            if test_game.place_stone(ex, ey):
                if test_game.game_over and test_game.winner == 1:
                    self.logic.place_stone(ex, ey)
                    return

        # 2. Central Oriented Random
        weighted_spots = []
        for ex, ey in empty_spots:
            dist = abs(ex - 4) + abs(ey - 4)
            weight = 10 - dist
            weighted_spots.append(((ex, ey), weight))

        total = sum(w for _, w in weighted_spots)
        r = random.uniform(0, total)
        curr = 0
        selected = weighted_spots[0][0]

        for move, w in weighted_spots:
            curr += w
            if curr >= r:
                selected = move
                break

        if not self.logic.place_stone(selected[0], selected[1]):
            # Fully random backup in case of failure
            random.shuffle(empty_spots)
            for rx, ry in empty_spots:
                if self.logic.place_stone(rx, ry): return

        self.logic.check_game_end_simple()