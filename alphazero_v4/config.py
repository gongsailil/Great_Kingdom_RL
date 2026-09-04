"""Fixed configuration for the bounded AlphaZero V4 stability run."""

from dataclasses import dataclass

from alphazero_v2.training_runner import TrainingRunConfig


@dataclass(frozen=True)
class V4Config(TrainingRunConfig):
    mcts_simulations: int = 256
    input_planes: int = 9
    encoder_version: str = "v3_territory"
    value_head: str = "raw_logit"
    temperature_schedule: str = "early8"
    diagnostic_interval: int = 10
    value_oracle_max_states: int = 2_000

    def __post_init__(self):
        super().__post_init__()
        if self.input_planes != 9:
            raise ValueError("V4 requires the unchanged nine-plane V3 encoder")
        if self.encoder_version != "v3_territory":
            raise ValueError("V4 requires the V3 territory encoder")
        if self.value_head != "raw_logit":
            raise ValueError("V4 requires a raw-logit value head")
        if self.temperature_schedule != "early8":
            raise ValueError("V4 stability run requires the early8 schedule")
        if self.diagnostic_interval <= 0 or self.value_oracle_max_states <= 0:
            raise ValueError("V4 diagnostic settings must be positive")
