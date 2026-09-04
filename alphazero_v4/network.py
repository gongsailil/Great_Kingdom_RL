"""V4 policy/value network with an unsquashed Bernoulli value logit."""

import torch
from torch import nn

from alphazero_v2.network import ResidualBlock
from great_kingdom_v2 import BOARD_SIZE, NUM_ACTIONS


def value_logit_to_probability(value_logit):
    return torch.sigmoid(value_logit)


def value_logit_to_scalar(value_logit):
    return 2.0 * value_logit_to_probability(value_logit) - 1.0


class PolicyValueLogitNetwork(nn.Module):
    """Same V3-sized tower/policy head; value output is a raw logit."""

    def __init__(self, channels=64, residual_blocks=3, input_planes=9):
        super().__init__()
        self.channels = int(channels)
        self.residual_blocks = int(residual_blocks)
        self.input_planes = int(input_planes)
        if self.input_planes != 9:
            raise ValueError("V4 network requires exactly nine input planes")
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_planes, self.channels, 3, padding=1, bias=False),
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
        )

    def forward(self, inputs):
        features = self.tower(self.stem(inputs))
        policy_logits = self.policy_head(features)
        value_logit = self.value_fc(self.value_conv(features)).squeeze(-1)
        return policy_logits, value_logit

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class ScalarValueNetworkAdapter(nn.Module):
    """Expose V4 logits as the [-1,+1] scalar expected by search/audits."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs):
        policy_logits, value_logit = self.network(inputs)
        return policy_logits, value_logit_to_scalar(value_logit)
