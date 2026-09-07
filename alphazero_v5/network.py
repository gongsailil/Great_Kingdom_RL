"""V5 128x6 policy/value network with Rules-derived auxiliary heads."""

import torch
from torch import nn

from alphazero_v2.network import ResidualBlock
from great_kingdom_v2 import BOARD_SIZE, NUM_ACTIONS


class PolicyValueAuxNetwork(nn.Module):
    def __init__(self, channels=128, residual_blocks=6, input_planes=12):
        super().__init__()
        self.channels = int(channels)
        self.residual_blocks = int(residual_blocks)
        self.input_planes = int(input_planes)
        if self.input_planes != 12:
            raise ValueError("V5 network requires twelve input planes")
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_planes, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels), nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(self.channels) for _ in range(self.residual_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.channels, 2, 1, bias=False), nn.BatchNorm2d(2),
            nn.ReLU(inplace=True), nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_ACTIONS),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1, bias=False), nn.BatchNorm2d(1),
            nn.ReLU(inplace=True), nn.Flatten(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 128), nn.ReLU(inplace=True), nn.Linear(128, 1),
        )
        self.liberty_head = nn.Sequential(
            nn.Conv2d(self.channels, 32, 1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.Conv2d(32, 2, 1),
        )
        self.score_conv = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1, bias=False), nn.BatchNorm2d(1),
            nn.ReLU(inplace=True), nn.Flatten(),
        )
        self.score_fc = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 64), nn.ReLU(inplace=True), nn.Linear(64, 1),
        )

    def forward(self, inputs):
        features = self.tower(self.stem(inputs))
        policy_logits = self.policy_head(features)
        value_logit = self.value_fc(self.value_conv(features)).squeeze(-1)
        liberty_logits = self.liberty_head(features)
        score_logit = self.score_fc(self.score_conv(features)).squeeze(-1)
        return policy_logits, value_logit, liberty_logits, score_logit

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def calibrated_probability(value_logit, temperature):
    temperature = torch.as_tensor(temperature, dtype=value_logit.dtype, device=value_logit.device)
    if torch.any(temperature <= 0):
        raise ValueError("calibration temperature must be positive")
    return torch.sigmoid(value_logit / temperature)


def calibrated_scalar(value_logit, temperature):
    return 2.0 * calibrated_probability(value_logit, temperature) - 1.0


def liberty_prediction(liberty_logits):
    return torch.sigmoid(liberty_logits)


def score_prediction(score_logit):
    return torch.tanh(score_logit)
