import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """
    Custom CNN for (3, 9, 9) input
    Output: 64-dim feature vector
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(SmallCNN, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # Input shape: (3, 9, 9)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        # Calculate the size after convolution
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            conv_out = self.cnn(sample)
            conv_out_dim = conv_out.shape[1]

        # Linear layer → feature vector
        self.linear = nn.Sequential(
            nn.Linear(conv_out_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(self.cnn(x))
