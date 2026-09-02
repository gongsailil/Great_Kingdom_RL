"""Fixed configuration for the territory representation pilot."""

from dataclasses import dataclass

from alphazero_v2.training_runner import TrainingRunConfig


@dataclass(frozen=True)
class TerritoryPilotConfig(TrainingRunConfig):
    mcts_simulations: int = 256
    input_planes: int = 9
    encoder_version: str = "v3_territory"
    diagnostic_simulations: int = 256
    diagnostic_interval: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.input_planes != 9:
            raise ValueError("the V3 territory pilot requires exactly 9 planes")
        if self.encoder_version != "v3_territory":
            raise ValueError("unsupported V3 encoder version")
        if self.diagnostic_simulations <= 0:
            raise ValueError("diagnostic_simulations must be positive")
        if self.diagnostic_interval <= 0:
            raise ValueError("diagnostic_interval must be positive")
