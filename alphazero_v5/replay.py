"""Compact 200k FIFO replay and exact-state, held-out duplicate-aware views."""

from dataclasses import dataclass
import hashlib

import numpy as np

from great_kingdom_v2 import NUM_ACTIONS
from .encoder import ENCODED_SHAPE
from .symmetry import transform_policy, transform_state


@dataclass
class AggregatedSample:
    state: np.ndarray
    policy: np.ndarray
    win_probability: float
    occurrence_count: int
    digest: bytes


class CompactReplayBuffer:
    FORMAT_VERSION = 1

    def __init__(self, max_positions=200_000):
        if int(max_positions) <= 0:
            raise ValueError("max_positions must be positive")
        self.max_positions = int(max_positions)
        self.states = np.empty((self.max_positions, *ENCODED_SHAPE), dtype=np.float16)
        self.policies = np.empty((self.max_positions, NUM_ACTIONS), dtype=np.float16)
        self.values = np.empty(self.max_positions, dtype=np.int8)
        self.players = np.empty(self.max_positions, dtype=np.uint8)
        self.start = 0
        self.size = 0
        self.total_samples_seen = 0
        self.generation_metadata = {}

    def __len__(self):
        return self.size

    def extend(self, examples):
        for example in examples:
            state = np.asarray(example.state, dtype=np.float32)
            policy = np.asarray(example.policy, dtype=np.float32)
            value, player = float(example.value), int(example.player)
            if state.shape != ENCODED_SHAPE or policy.shape != (NUM_ACTIONS,):
                raise ValueError("V5 replay sample shape mismatch")
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(policy)):
                raise ValueError("V5 replay sample is non-finite")
            if np.any(policy < 0) or not np.isclose(policy.sum(), 1.0, atol=1e-5):
                raise ValueError("V5 replay policy is invalid")
            if value not in (-1.0, 1.0) or player not in (1, 2):
                raise ValueError("V5 replay value/player invalid")
            if self.size < self.max_positions:
                index = (self.start + self.size) % self.max_positions
                self.size += 1
            else:
                index = self.start
                self.start = (self.start + 1) % self.max_positions
            self.states[index] = state
            stored_policy = policy.astype(np.float16)
            stored_policy /= stored_policy.sum(dtype=np.float32)
            self.policies[index] = stored_policy
            self.values[index] = int(value)
            self.players[index] = player
            self.total_samples_seen += 1

    def _indices(self):
        return (self.start + np.arange(self.size, dtype=np.int64)) % self.max_positions

    def arrays(self):
        indices = self._indices()
        return self.states[indices], self.policies[indices], self.values[indices], self.players[indices]

    def memory_bytes(self):
        return int(self.states.nbytes + self.policies.nbytes + self.values.nbytes + self.players.nbytes)

    def metadata(self):
        return {"current_size": self.size, "max_positions": self.max_positions,
                "total_samples_seen": self.total_samples_seen,
                "allocated_bytes": self.memory_bytes(), "generation": dict(self.generation_metadata)}

    def state_dict(self):
        states, policies, values, players = self.arrays()
        return {"format_version": self.FORMAT_VERSION, "max_positions": self.max_positions,
                "total_samples_seen": self.total_samples_seen,
                "generation_metadata": dict(self.generation_metadata),
                "states": states, "policies": policies, "values": values, "players": players}

    @classmethod
    def from_state_dict(cls, payload):
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported V5 replay format")
        buffer = cls(payload["max_positions"])
        states = np.asarray(payload["states"])
        policies = np.asarray(payload["policies"])
        values = np.asarray(payload["values"])
        players = np.asarray(payload["players"])
        if not (len(states) == len(policies) == len(values) == len(players)):
            raise ValueError("V5 replay arrays have inconsistent lengths")
        if len(states) > buffer.max_positions:
            raise ValueError("V5 replay exceeds configured capacity")
        count = len(states)
        buffer.states[:count] = states
        buffer.policies[:count] = policies
        buffer.values[:count] = values
        buffer.players[:count] = players
        buffer.size = count
        buffer.total_samples_seen = int(payload["total_samples_seen"])
        buffer.generation_metadata = dict(payload["generation_metadata"])
        if buffer.total_samples_seen < count:
            raise ValueError("total samples seen is smaller than replay")
        return buffer


class DuplicateAwareSplitView:
    def __init__(self, replay, validation_fraction=0.05):
        states, policies, values, _ = replay.arrays()
        groups = {}
        for state, policy, value in zip(states, policies, values):
            contiguous = np.ascontiguousarray(state)
            raw = contiguous.tobytes()
            digest = hashlib.sha256(raw).digest()
            group = groups.get(digest)
            if group is None:
                groups[digest] = [contiguous, raw, policy.astype(np.float64),
                                  1.0 if value == 1 else 0.0, 1, 1 if value < 0 else 2]
            else:
                if group[1] != raw:
                    raise RuntimeError("SHA-256 collision in V5 replay")
                group[2] += policy
                group[3] += 1.0 if value == 1 else 0.0
                group[4] += 1
                group[5] |= 1 if value < 0 else 2
        self.train, self.validation, masks = [], [], []
        for digest, (state, _, policy_sum, win_sum, count, mask) in groups.items():
            sample = AggregatedSample(state, (policy_sum / count).astype(np.float32),
                                      float(win_sum / count), count, digest)
            fraction = int.from_bytes(digest[:8], "big") / 2**64
            target = self.validation if fraction < validation_fraction else self.train
            target.append(sample)
            masks.append((mask, count, target is self.validation))
        if not self.train or not self.validation:
            raise ValueError("replay cannot produce non-empty train and validation splits")
        self.raw_size = len(states)
        self._masks = masks
        self.train_weights = np.asarray([s.occurrence_count for s in self.train], dtype=np.float64)
        self.train_weights /= self.train_weights.sum()
        self.validation_weights = np.asarray([s.occurrence_count for s in self.validation], dtype=np.float64)

    def sample_training_batch(self, batch_size, rng, augment=True):
        indices = rng.choice(len(self.train), size=int(batch_size), replace=True, p=self.train_weights)
        states, policies, targets = [], [], []
        for index in indices:
            sample = self.train[int(index)]
            transform = int(rng.integers(8)) if augment else 0
            state = transform_state(sample.state, transform).astype(np.float32)
            policy = transform_policy(sample.policy, transform).astype(np.float32)
            policy /= policy.sum()
            states.append(state); policies.append(policy); targets.append(sample.win_probability)
        states = np.stack(states)
        return states, np.stack(policies), np.asarray(targets, dtype=np.float32), states[:, 9:11], states[:, 11, 0, 0]

    def validation_arrays(self):
        return (np.stack([s.state for s in self.validation]).astype(np.float32),
                np.asarray([s.win_probability for s in self.validation], dtype=np.float32),
                self.validation_weights.astype(np.float32))

    def metrics(self):
        all_samples = self.train + self.validation
        duplicate = [s for s in all_samples if s.occurrence_count > 1]
        contradictory = [(mask, count, is_val) for mask, count, is_val in self._masks if mask == 3]
        return {"raw_replay_size": self.raw_size, "unique_state_groups": len(all_samples),
                "train_groups": len(self.train), "validation_groups": len(self.validation),
                "validation_raw_weight": int(self.validation_weights.sum()),
                "duplicate_group_count": len(duplicate),
                "contradictory_z_group_count": len(contradictory),
                "samples_in_contradictory_groups": int(sum(c for _, c, _ in contradictory)),
                "mixed_outcome_group_fraction": len(contradictory) / len(all_samples),
                "validation_fraction_actual": float(self.validation_weights.sum() / self.raw_size)}
