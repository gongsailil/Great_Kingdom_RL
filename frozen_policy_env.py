from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from gk_env import GreatKingdomEnv


def predict_with_local_rng(model, obs, action_mask, deterministic, rng):
    """Masked prediction whose stochastic sampling is driven by a local RNG."""
    if deterministic:
        action, _ = model.predict(
            obs,
            action_masks=action_mask,
            deterministic=True,
        )
    else:
        torch_seed = int(rng.integers(0, 2**31 - 1))
        # MaskablePPO samples through PyTorch's global RNG. fork_rng restores
        # that global state after this one prediction, while the local NumPy
        # stream makes the opponent sequence reproducible.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch_seed)
            action, _ = model.predict(
                obs,
                action_masks=action_mask,
                deterministic=False,
            )
    return int(np.asarray(action).item())


class FrozenPolicyOpponentEnv(GreatKingdomEnv):
    """Great Kingdom learner environment backed by one frozen PPO opponent."""

    def __init__(
        self,
        *,
        agent_player=2,
        opponent_model_path,
        opponent_deterministic=False,
        opponent_seed=20260822,
    ):
        super().__init__(agent_player=agent_player)
        model_path = Path(opponent_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        self.opponent_model_path = str(model_path)
        self.opponent_model = MaskablePPO.load(model_path, device="cpu")
        self.opponent_deterministic = opponent_deterministic
        self._opponent_seed = int(opponent_seed)
        self._opponent_policy_rng = np.random.default_rng(self._opponent_seed)
        self.opponent_mask_violations = 0
        self.move_trace = []

    def reset(self, seed=None, options=None):
        self.opponent_mask_violations = 0
        self.move_trace = []
        if seed is not None:
            seed_sequence = np.random.SeedSequence(
                [self._opponent_seed, int(seed)]
            )
            self._opponent_policy_rng = np.random.default_rng(seed_sequence)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = int(action)
        trace_entry = self._trace_entry(self.agent_player, action)
        self.move_trace.append(trace_entry)

        transition = super().step(action)
        info = transition[4]
        outcome = info.get("outcome")
        if outcome == "mask_violation":
            trace_entry["move_result"] = "MASK_VIOLATION"
        elif outcome == "agent_suicide":
            trace_entry["move_result"] = "SUICIDE_LOSS"
        elif outcome == "agent_capture_win":
            trace_entry["move_result"] = "CAPTURE_WIN"
        else:
            trace_entry["move_result"] = "NORMAL"
        return transition

    def _opponent_move_random(self):
        if self.logic.game_over:
            return self.logic.last_move_result

        action_mask = self.action_masks()
        if not np.any(action_mask):
            self.logic.check_game_end_simple()
            return self.logic.last_move_result

        opponent_obs = self._get_obs_for(self.opponent_player)
        action = predict_with_local_rng(
            self.opponent_model,
            opponent_obs,
            action_mask,
            self.opponent_deterministic,
            self._opponent_policy_rng,
        )
        if action < 0 or action >= action_mask.size or not bool(action_mask[action]):
            self.opponent_mask_violations += 1
            raise RuntimeError(f"frozen opponent selected masked action {action}")

        x = action % self.board_size
        y = action // self.board_size
        result = self.logic.place_stone_detailed(x, y)
        if result.name == "NORMAL":
            self.logic.check_game_end_simple()
        trace_entry = self._trace_entry(self.opponent_player, action)
        trace_entry["move_result"] = result.name
        self.move_trace.append(trace_entry)
        return result

    def _trace_entry(self, player, action):
        return {
            "ply": len(self.move_trace) + 1,
            "player": player,
            "color": "Blue" if player == 1 else "Red",
            "action": action,
            "coordinate": [action % self.board_size, action // self.board_size],
            "move_result": None,
        }
