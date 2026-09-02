"""Compact shared policy-value ResNet."""

import torch
from torch import nn

from great_kingdom_v2 import BOARD_SIZE, NUM_ACTIONS

from .encoder import NUM_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class PolicyValueNetwork(nn.Module):
    def __init__(self, channels=64, residual_blocks=3, input_planes=NUM_PLANES):
        super().__init__()
        self.channels = int(channels)
        self.residual_blocks = int(residual_blocks)
        self.input_planes = int(input_planes)
        if self.input_planes <= 0:
            raise ValueError("input_planes must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_planes,
                self.channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(self.channels) for _ in range(self.residual_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_ACTIONS),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        features = self.tower(self.stem(inputs))
        policy_logits = self.policy_head(features)
        value = self.value_fc(self.value_conv(features)).squeeze(-1)
        return policy_logits, value

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
