"""Small shared primitives used by the self-contained AlphaZero V5 package."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from great_kingdom_v2 import NUM_ACTIONS


@dataclass
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    value: float
    player: int


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + inputs)


class Node:
    def __init__(self, prior, to_play):
        self.prior = float(prior)
        self.to_play = int(to_play)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def expanded(self):
        return bool(self.children)

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def terminal_value(logic, perspective_player):
    if not logic.game_over or logic.winner not in (1, 2):
        raise ValueError("terminal value requires a terminal Rules V2 winner")
    return 1.0 if logic.winner == perspective_player else -1.0


def backup(search_path, value, value_player):
    for node in reversed(search_path):
        node.value_sum += float(value if node.to_play == value_player else -value)
        node.visit_count += 1


def masked_policy(logits, legal_mask):
    logits = np.asarray(logits, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if logits.shape != (NUM_ACTIONS,) or legal_mask.shape != (NUM_ACTIONS,):
        raise ValueError("policy logits and legal mask must both have shape (82,)")
    if not legal_mask.any():
        raise RuntimeError("active Rules V2 state has no legal action")
    probabilities = np.zeros(NUM_ACTIONS, dtype=np.float64)
    maximum = np.max(logits[legal_mask])
    probabilities[legal_mask] = np.exp(logits[legal_mask] - maximum)
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float32)


def visit_count_policy(root, temperature=1.0):
    visits = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for action, child in root.children.items():
        visits[action] = child.visit_count
    if temperature <= 0.0:
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[int(np.argmax(visits))] = 1.0
        return policy
    adjusted = np.power(visits, 1.0 / temperature)
    if adjusted.sum() == 0.0:
        for action, child in root.children.items():
            adjusted[action] = child.prior
    adjusted /= adjusted.sum()
    return adjusted.astype(np.float32)


def temperature_for_ply(schedule, ply):
    if schedule not in ("all_hot", "early8", "greedy"):
        raise ValueError(f"unknown temperature schedule: {schedule}")
    if int(ply) < 0:
        raise ValueError("ply must be non-negative")
    if schedule == "all_hot":
        return 1.0
    if schedule == "early8":
        return 1.0 if int(ply) < 8 else 0.0
    return 0.0


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_metric(path, metric):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
