"""Bounded FIFO replay storage for AlphaZero V2 training examples."""

import numpy as np

from great_kingdom_v2 import NUM_ACTIONS

from .encoder import ENCODED_SHAPE
from .self_play import TrainingExample


class ReplayBuffer:
    def __init__(self, max_positions, encoded_shape=ENCODED_SHAPE):
        if int(max_positions) <= 0:
            raise ValueError("max_positions must be positive")
        self.max_positions = int(max_positions)
        self.encoded_shape = tuple(int(value) for value in encoded_shape)
        if len(self.encoded_shape) != 3 or any(
            value <= 0 for value in self.encoded_shape
        ):
            raise ValueError("encoded_shape must contain three positive dimensions")
        self.samples = []
        self.total_samples_seen = 0
        self.generation_metadata = {
            "iteration": 0,
            "total_self_play_games": 0,
            "total_samples_generated": 0,
        }

    def __len__(self):
        return len(self.samples)

    def _validated_copy(self, example):
        state = np.asarray(example.state, dtype=np.float32)
        policy = np.asarray(example.policy, dtype=np.float32)
        value = float(example.value)
        player = int(example.player)
        if state.shape != self.encoded_shape:
            raise ValueError(
                f"replay state must have shape {self.encoded_shape}"
            )
        if policy.shape != (NUM_ACTIONS,):
            raise ValueError("replay policy must have shape (82,)")
        if not np.isclose(policy.sum(), 1.0):
            raise ValueError("replay policy must sum to one")
        if np.any(policy < 0.0) or not np.all(np.isfinite(policy)):
            raise ValueError("replay policy must be finite and non-negative")
        if value not in (-1.0, 1.0):
            raise ValueError("replay outcome must be -1 or +1")
        if player not in (1, 2):
            raise ValueError("replay player must be Blue (1) or Red (2)")
        return TrainingExample(state.copy(), policy.copy(), value, player)

    def extend(self, examples):
        additions = [self._validated_copy(example) for example in examples]
        self.total_samples_seen += len(additions)
        self.samples.extend(additions)
        overflow = len(self.samples) - self.max_positions
        if overflow > 0:
            del self.samples[:overflow]

    def sample(self, batch_size, rng):
        if not self.samples:
            raise ValueError("cannot sample an empty replay buffer")
        size = min(int(batch_size), len(self.samples))
        if size <= 0:
            raise ValueError("batch_size must be positive")
        indices = rng.choice(len(self.samples), size=size, replace=False)
        return [self.samples[int(index)] for index in indices]

    def metadata(self):
        return {
            "current_size": len(self),
            "max_positions": self.max_positions,
            "total_samples_seen": self.total_samples_seen,
            "generation": dict(self.generation_metadata),
        }

    def state_dict(self):
        if self.samples:
            states = np.stack([sample.state for sample in self.samples])
            policies = np.stack([sample.policy for sample in self.samples])
            values = np.asarray(
                [sample.value for sample in self.samples], dtype=np.float32
            )
            players = np.asarray(
                [sample.player for sample in self.samples], dtype=np.int8
            )
        else:
            states = np.empty((0, *self.encoded_shape), dtype=np.float32)
            policies = np.empty((0, NUM_ACTIONS), dtype=np.float32)
            values = np.empty((0,), dtype=np.float32)
            players = np.empty((0,), dtype=np.int8)
        return {
            "max_positions": self.max_positions,
            "encoded_shape": self.encoded_shape,
            "total_samples_seen": self.total_samples_seen,
            "generation_metadata": dict(self.generation_metadata),
            "states": states,
            "policies": policies,
            "values": values,
            "players": players,
        }

    @classmethod
    def from_state_dict(cls, payload):
        states = np.asarray(payload["states"], dtype=np.float32)
        encoded_shape = tuple(
            payload.get("encoded_shape", states.shape[1:])
        )
        buffer = cls(payload["max_positions"], encoded_shape=encoded_shape)
        policies = np.asarray(payload["policies"], dtype=np.float32)
        values = np.asarray(payload["values"], dtype=np.float32)
        players = np.asarray(payload["players"], dtype=np.int8)
        count = len(states)
        if not (len(policies) == len(values) == len(players) == count):
            raise ValueError("replay checkpoint arrays have inconsistent lengths")
        buffer.extend(
            TrainingExample(states[i], policies[i], values[i], players[i])
            for i in range(count)
        )
        buffer.total_samples_seen = int(payload["total_samples_seen"])
        if buffer.total_samples_seen < len(buffer):
            raise ValueError("replay total_samples_seen is smaller than current size")
        buffer.generation_metadata = dict(payload["generation_metadata"])
        return buffer
