"""Iteration runner, checkpointing, and metrics for sustained V2 training."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .config import AlphaZeroConfig
from .network import PolicyValueNetwork
from .replay_buffer import ReplayBuffer
from .self_play import generate_self_play
from .train import loss_components


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TrainingRunConfig:
    channels: int = 64
    residual_blocks: int = 3
    mcts_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    temperature: float = 1.0
    self_play_games_per_iteration: int = 32
    max_game_moves: int = 200
    replay_max_positions: int = 50_000
    batch_size: int = 256
    training_updates_per_iteration: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_milestone_interval: int = 10
    checkpoint_keep_recent: int = 3
    seed: int = 20260830

    def __post_init__(self):
        positive_integer_fields = (
            "channels",
            "residual_blocks",
            "mcts_simulations",
            "self_play_games_per_iteration",
            "max_game_moves",
            "replay_max_positions",
            "batch_size",
            "training_updates_per_iteration",
            "checkpoint_milestone_interval",
            "checkpoint_keep_recent",
        )
        for name in positive_integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings must be non-negative")
        if not 0.0 <= self.dirichlet_fraction <= 1.0:
            raise ValueError("dirichlet_fraction must be between zero and one")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def self_play_config(self):
        return AlphaZeroConfig(
            channels=self.channels,
            residual_blocks=self.residual_blocks,
            mcts_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
            temperature=self.temperature,
            self_play_games=self.self_play_games_per_iteration,
            max_game_moves=self.max_game_moves,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            training_epochs=1,
            seed=self.seed,
        )


@dataclass
class TrainingRunState:
    network: PolicyValueNetwork
    optimizer: torch.optim.Optimizer
    replay: ReplayBuffer
    rng: np.random.Generator
    iteration: int = 0
    total_self_play_games: int = 0
    total_samples_generated: int = 0
    elapsed_seconds: float = 0.0
    last_metric: dict | None = None


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def _atomic_torch_save(payload, path):
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


def _atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
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


def _record_missing_last_metric(run_dir, metric):
    if metric is None:
        return
    metrics_path = Path(run_dir) / "metrics.jsonl"
    last_iteration = None
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_iteration = json.loads(line)["iteration"]
    if last_iteration is None or last_iteration < metric["iteration"]:
        append_metric(metrics_path, metric)
    elif last_iteration > metric["iteration"]:
        raise RuntimeError("metrics are ahead of the latest checkpoint")


def _optimizer_for(network, config):
    return torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _checkpoint_payload(state, config):
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "network_state_dict": state.network.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "iteration": state.iteration,
        "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
        "elapsed_seconds": state.elapsed_seconds,
        "replay_metadata": state.replay.metadata(),
        "config": config.to_dict(),
        "numpy_rng_state": state.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "last_metric": state.last_metric,
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def retain_iteration_checkpoints(run_dir, config):
    checkpoint_dir = Path(run_dir) / "checkpoints"
    candidates = []
    for path in checkpoint_dir.glob("iteration_*.pt"):
        try:
            iteration = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        candidates.append((iteration, path))
    candidates.sort()
    recent = {
        iteration
        for iteration, _ in candidates[-config.checkpoint_keep_recent :]
    }
    for iteration, path in candidates:
        milestone = iteration % config.checkpoint_milestone_interval == 0
        if not milestone and iteration not in recent:
            path.unlink()


def save_run_checkpoint(run_dir, state, config, *, iteration_checkpoint=True):
    run_dir = Path(run_dir)
    state.replay.generation_metadata = {
        "iteration": state.iteration,
        "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
    }
    # Replay first: a crash between files can leave extra valid data beside the
    # previous intact network checkpoint, never a new network with missing data.
    _atomic_torch_save(state.replay.state_dict(), run_dir / "replay_buffer.pt")
    payload = _checkpoint_payload(state, config)
    _atomic_torch_save(payload, run_dir / "latest.pt")
    if iteration_checkpoint and state.iteration > 0:
        checkpoint = (
            run_dir / "checkpoints" / f"iteration_{state.iteration:06d}.pt"
        )
        _atomic_torch_save(payload, checkpoint)
    retain_iteration_checkpoints(run_dir, config)


def initialize_run(run_dir, config, device):
    run_dir = Path(run_dir)
    if (run_dir / "latest.pt").exists():
        raise FileExistsError(f"run already exists; use --resume {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _atomic_json_save(config.to_dict(), run_dir / "config.json")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    network = PolicyValueNetwork(
        channels=config.channels,
        residual_blocks=config.residual_blocks,
    ).to(device)
    state = TrainingRunState(
        network=network,
        optimizer=_optimizer_for(network, config),
        replay=ReplayBuffer(config.replay_max_positions),
        rng=np.random.default_rng(config.seed),
    )
    save_run_checkpoint(run_dir, state, config, iteration_checkpoint=False)
    _write_summary(run_dir, state, config, device)
    return state


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    with (run_dir / "config.json").open(encoding="utf-8") as handle:
        config = TrainingRunConfig.from_dict(json.load(handle))
    payload = torch.load(
        run_dir / "latest.pt", map_location=device, weights_only=False
    )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint format")
    if payload["config"] != config.to_dict():
        raise ValueError("config.json and latest.pt disagree")
    replay_payload = torch.load(
        run_dir / "replay_buffer.pt", map_location="cpu", weights_only=False
    )
    replay = ReplayBuffer.from_state_dict(replay_payload)
    if replay.max_positions != config.replay_max_positions:
        raise ValueError("replay buffer capacity disagrees with training config")
    replay_iteration = int(replay.generation_metadata["iteration"])
    if replay_iteration < int(payload["iteration"]):
        raise RuntimeError("replay buffer is older than latest network checkpoint")

    network = PolicyValueNetwork(
        channels=config.channels,
        residual_blocks=config.residual_blocks,
    ).to(device)
    optimizer = _optimizer_for(network, config)
    network.load_state_dict(payload["network_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    state = TrainingRunState(
        network=network,
        optimizer=optimizer,
        replay=replay,
        rng=rng,
        iteration=int(payload["iteration"]),
        total_self_play_games=max(
            int(payload["total_self_play_games"]),
            int(replay.generation_metadata["total_self_play_games"]),
        ),
        total_samples_generated=max(
            int(payload["total_samples_generated"]),
            int(replay.generation_metadata["total_samples_generated"]),
        ),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        last_metric=payload["last_metric"],
    )
    _record_missing_last_metric(run_dir, state.last_metric)
    return config, state


def latest_completed_iteration(run_dir):
    """Read the last atomically published iteration without mutating the run."""
    payload = torch.load(
        Path(run_dir) / "latest.pt", map_location="cpu", weights_only=False
    )
    return int(payload["iteration"])


def train_replay_updates(state, config, device):
    if not state.replay:
        raise ValueError("cannot train from an empty replay buffer")
    totals = np.zeros(3, dtype=np.float64)
    state.network.train()
    for _ in range(config.training_updates_per_iteration):
        batch = state.replay.sample(config.batch_size, state.rng)
        states = torch.from_numpy(np.stack([item.state for item in batch])).to(device)
        policies = torch.from_numpy(np.stack([item.policy for item in batch])).to(
            device
        )
        values = torch.tensor(
            [item.value for item in batch], dtype=torch.float32, device=device
        )
        state.optimizer.zero_grad(set_to_none=True)
        losses = loss_components(state.network, states, policies, values)
        if not all(torch.isfinite(loss) for loss in losses):
            raise RuntimeError("non-finite AlphaZero training loss")
        losses[2].backward()
        state.optimizer.step()
        totals += np.asarray([float(loss.item()) for loss in losses])
    totals /= config.training_updates_per_iteration
    return {
        "policy_loss": float(totals[0]),
        "value_loss": float(totals[1]),
        "total_loss": float(totals[2]),
    }


def _write_summary(run_dir, state, config, device):
    summary = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "iteration": state.iteration,
        "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
        "replay_size": len(state.replay),
        "elapsed_seconds": state.elapsed_seconds,
        "config": config.to_dict(),
        "last_metric": state.last_metric,
    }
    _atomic_json_save(summary, Path(run_dir) / "summary.json")


def run_iteration(run_dir, state, config, device):
    iteration_started = time.perf_counter()
    self_play_started = time.perf_counter()
    examples, games = generate_self_play(
        state.network,
        config.self_play_config(),
        device,
        state.rng,
    )
    self_play_seconds = time.perf_counter() - self_play_started
    if not examples:
        raise RuntimeError("self-play iteration generated no samples")
    if any(game["illegal_probability_violations"] for game in games):
        raise RuntimeError("illegal self-play probability violation")
    state.replay.extend(examples)

    training_started = time.perf_counter()
    losses = train_replay_updates(state, config, device)
    training_seconds = time.perf_counter() - training_started
    state.iteration += 1
    state.total_self_play_games += len(games)
    state.total_samples_generated += len(examples)
    iteration_seconds = time.perf_counter() - iteration_started
    state.elapsed_seconds += iteration_seconds

    winners = [game["winner"] for game in games]
    reasons = [game["terminal_reason"] for game in games]
    metric = {
        "iteration": state.iteration,
        "elapsed_seconds": state.elapsed_seconds,
        "iteration_seconds": iteration_seconds,
        "self_play_seconds": self_play_seconds,
        "training_seconds": training_seconds,
        "total_self_play_games": state.total_self_play_games,
        "new_games": len(games),
        "new_samples": len(examples),
        "total_samples_generated": state.total_samples_generated,
        "replay_size": len(state.replay),
        "blue_wins": winners.count(1),
        "red_wins": winners.count(2),
        "capture_endings": reasons.count("CAPTURE_WIN"),
        "pass_score_endings": reasons.count("PASS_SCORE_END"),
        "mean_game_length": float(np.mean([game["game_length"] for game in games])),
        "mean_pass_count": float(np.mean([game["pass_usage"] for game in games])),
        **losses,
        "training_updates": config.training_updates_per_iteration,
        "mcts_simulations": config.mcts_simulations,
        "gpu_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "games": games,
    }
    state.last_metric = metric
    save_run_checkpoint(run_dir, state, config)
    append_metric(Path(run_dir) / "metrics.jsonl", metric)
    _write_summary(run_dir, state, config, device)
    return metric


def run_until_budget(run_dir, state, config, device, hours=None):
    if hours is not None:
        hours = float(hours)
        if hours < 0:
            raise ValueError("hours must be non-negative")
        if hours == 0:
            hours = None
    budget_seconds = None if hours is None else hours * 3600.0
    completed = []
    while budget_seconds is None or state.elapsed_seconds < budget_seconds:
        metric = run_iteration(run_dir, state, config, device)
        completed.append(metric)
        print(json.dumps(metric, sort_keys=True), flush=True)
    # The last iteration always completes and checkpoints before this boundary.
    _write_summary(run_dir, state, config, device)
    return completed
