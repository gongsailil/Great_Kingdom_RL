"""Duplicate-aware training view over the unchanged raw replay buffer."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from great_kingdom_v2 import NUM_ACTIONS


@dataclass
class AggregatedSample:
    state: np.ndarray
    policy: np.ndarray
    win_probability: float
    occurrence_count: int


class DuplicateAwareTrainingView:
    def __init__(self, samples):
        groups = {}
        total = 0
        for sample in samples:
            state = np.asarray(sample.state, dtype=np.float32)
            policy = np.asarray(sample.policy, dtype=np.float32)
            value = float(sample.value)
            if policy.shape != (NUM_ACTIONS,) or value not in (-1.0, 1.0):
                raise ValueError("raw replay sample has invalid policy/value")
            state_bytes = np.ascontiguousarray(state).tobytes()
            key = hashlib.sha256(state_bytes).digest()
            group = groups.get(key)
            if group is None:
                group = {
                    "state": state,
                    "state_bytes": state_bytes,
                    "policy_sum": policy.astype(np.float64),
                    "win_sum": 1.0 if value == 1.0 else 0.0,
                    "count": 1,
                    "outcomes": 1 if value == -1.0 else 2,
                }
                groups[key] = group
            else:
                if group["state_bytes"] != state_bytes:
                    raise RuntimeError("SHA-256 collision in exact replay grouping")
                group["policy_sum"] += policy
                group["win_sum"] += 1.0 if value == 1.0 else 0.0
                group["count"] += 1
                group["outcomes"] |= 1 if value == -1.0 else 2
            total += 1
        if total == 0:
            raise ValueError("cannot build a training view from empty replay")

        self.samples = []
        self._outcome_masks = []
        for group in groups.values():
            count = int(group["count"])
            self.samples.append(
                AggregatedSample(
                    state=group["state"],
                    policy=(group["policy_sum"] / count).astype(np.float32),
                    win_probability=float(group["win_sum"] / count),
                    occurrence_count=count,
                )
            )
            self._outcome_masks.append(int(group["outcomes"]))
        self.raw_size = total
        self.occurrence_counts = np.asarray(
            [sample.occurrence_count for sample in self.samples], dtype=np.float64
        )
        self.sampling_probabilities = self.occurrence_counts / self.raw_size

    def __len__(self):
        return len(self.samples)

    def sample(self, batch_size, rng):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = rng.choice(
            len(self.samples),
            size=batch_size,
            replace=True,
            p=self.sampling_probabilities,
        )
        return [self.samples[int(index)] for index in indices]

    def metrics(self):
        duplicate_groups = int(np.sum(self.occurrence_counts > 1))
        contradictory = np.asarray(self._outcome_masks) == 3
        contradictory_samples = int(
            np.sum(self.occurrence_counts[contradictory])
        )
        probabilities = np.asarray(
            [sample.win_probability for sample in self.samples], dtype=np.float64
        )
        entropy = np.zeros_like(probabilities)
        mixed = (probabilities > 0.0) & (probabilities < 1.0)
        entropy[mixed] = -(
            probabilities[mixed] * np.log(probabilities[mixed])
            + (1.0 - probabilities[mixed])
            * np.log(1.0 - probabilities[mixed])
        )
        return {
            "raw_replay_size": self.raw_size,
            "unique_state_groups": len(self.samples),
            "duplicate_group_count": duplicate_groups,
            "contradictory_z_group_count": int(np.sum(contradictory)),
            "samples_in_contradictory_groups": contradictory_samples,
            "mixed_outcome_group_fraction": float(np.mean(contradictory)),
            "mean_group_win_probability_entropy": float(np.mean(entropy)),
        }
