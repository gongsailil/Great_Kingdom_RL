"""Integrated single-machine AlphaZero V5 strategy configuration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V5Config:
    input_planes: int = 12
    channels: int = 128
    residual_blocks: int = 6
    policy_actions: int = 82
    c_puct: float = 1.5
    opening_mcts_simulations: int = 256
    later_mcts_simulations: int = 128
    search_budget_cutoff_ply: int = 12
    arena_mcts_simulations: int = 256
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    temperature_schedule: str = "early8"
    concurrent_games: int = 64
    self_play_games_per_cycle: int = 256
    replay_max_positions: int = 200_000
    validation_fraction: float = 0.05
    batch_size: int = 512
    training_updates_per_cycle: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    d4_augmentation: bool = True
    candidate_arena_openings: int = 32
    candidate_arena_games: int = 64
    promotion_win_rate: float = 0.55
    territory_start_fraction: float = 0.20
    territory_pool_max_states: int = 4096
    seed: int = 20260830
    target_self_play_games: int = 10_000
    target_promotions: int = 20
    max_game_moves: int = 200

    def __post_init__(self):
        if (self.input_planes, self.policy_actions) != (12, 82):
            raise ValueError("V5 requires 12 planes and 82 actions")
        if self.channels != 128 or self.residual_blocks != 6:
            raise ValueError("V5 requires a 128x6 residual network")
        if self.concurrent_games < 2 or self.self_play_games_per_cycle % self.concurrent_games:
            raise ValueError("self-play games must divide into concurrent cohorts")
        if not 0 < self.validation_fraction < 1 or not 0 <= self.territory_start_fraction <= 1:
            raise ValueError("fractions outside valid range")
        if self.candidate_arena_games != 2 * self.candidate_arena_openings:
            raise ValueError("candidate arena must pair every opening by color")
        if not 0.5 < self.promotion_win_rate <= 1:
            raise ValueError("promotion threshold must exceed 50%")

    def simulations_for_ply(self, ply):
        return self.opening_mcts_simulations if int(ply) < self.search_budget_cutoff_ply else self.later_mcts_simulations

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in known})
