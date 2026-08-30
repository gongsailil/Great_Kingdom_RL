"""Mechanics-only Gymnasium interface for Great Kingdom Rules V2."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    CASTLES_PER_PLAYER,
    LEGAL_PLACEMENT_RESULTS,
    NUM_ACTIONS,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
)


def action_mask_for_logic(logic, player):
    """Return a player-dependent 82-action legality mask."""
    logic._validate_player(player)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if logic.game_over:
        return mask
    for action in range(PASS_ACTION):
        x = action % BOARD_SIZE
        y = action // BOARD_SIZE
        mask[action] = (
            logic.classify_placement(player, x, y) in LEGAL_PLACEMENT_RESULTS
        )
    mask[PASS_ACTION] = True
    return mask


class GreatKingdomEnvV2(gym.Env):
    """Two-player, turn-level rules environment with no opponent or reward design."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=3,
                    shape=(BOARD_SIZE, BOARD_SIZE),
                    dtype=np.uint8,
                ),
                "turn": spaces.Discrete(3),
                "consecutive_passes": spaces.Discrete(3),
                "castles_remaining": spaces.Box(
                    low=0,
                    high=CASTLES_PER_PLAYER,
                    shape=(2,),
                    dtype=np.uint8,
                ),
            }
        )
        self.logic = GreatKingdomLogicV2()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.logic = GreatKingdomLogicV2()
        return self._get_obs(), {}

    def action_masks(self, player=None):
        if player is None:
            player = self.logic.turn
        return action_mask_for_logic(self.logic, player)

    def step(self, action):
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise ValueError("action must be scalar")
        action = int(action_array.item())
        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError(f"action outside V2 action space: {action}")

        acting_player = self.logic.turn
        if not bool(self.action_masks(acting_player)[action]):
            raise ValueError(
                f"illegal V2 action {action} for player {acting_player}"
            )
        result = self.logic.apply_action(action)
        terminated = self.logic.game_over
        # Rules V2 intentionally defines no learning reward. Terminal outcome
        # is exposed through info without introducing reward shaping here.
        reward = 0.0
        info = {
            "acting_player": acting_player,
            "move_result": result.name,
            "winner": self.logic.winner,
        }
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        return {
            "board": np.asarray(self.logic.board, dtype=np.uint8),
            "turn": int(self.logic.turn),
            "consecutive_passes": int(self.logic.consecutive_passes),
            "castles_remaining": np.asarray(
                [
                    self.logic.castles_remaining[BLUE],
                    self.logic.castles_remaining[RED],
                ],
                dtype=np.uint8,
            ),
        }
