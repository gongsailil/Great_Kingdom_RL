"""Small serializable configuration for the minimal AlphaZero smoke."""

from dataclasses import asdict, dataclass


@dataclass
class AlphaZeroConfig:
    channels: int = 64
    residual_blocks: int = 3
    mcts_simulations: int = 32
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    temperature: float = 1.0
    self_play_games: int = 2
    max_game_moves: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    training_epochs: int = 12
    seed: int = 20260830

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
