"""Crash-safe bounded runner for the integrated AlphaZero V4 architecture."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from alphazero_v2.replay_buffer import ReplayBuffer
from alphazero_v2.training_runner import (
    _atomic_json_save,
    _atomic_torch_save,
    append_metric,
    retain_iteration_checkpoints,
)
from alphazero_v3.encoder import ENCODED_SHAPE

from .config import V4Config
from .diagnostics import (
    VALUE_ORACLE_ITERATIONS,
    run_fixed_tactical_diagnostics,
    run_value_oracle_monitor,
)
from .network import PolicyValueLogitNetwork, value_logit_to_probability
from .replay import DuplicateAwareTrainingView
from .self_play import generate_self_play


V4_CHECKPOINT_FORMAT_VERSION = 1
TACTICAL_DIAGNOSTIC_ITERATIONS = (0, 10, 20, 30, 40, 50)


@dataclass
class V4TrainingState:
    network: PolicyValueLogitNetwork
    optimizer: torch.optim.Optimizer
    replay: ReplayBuffer
    rng: np.random.Generator
    iteration: int = 0
    total_self_play_games: int = 0
    total_samples_generated: int = 0
    elapsed_seconds: float = 0.0
    last_metric: dict | None = None


def choose_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def value_loss_components(network, states, policies, win_probabilities):
    policy_logits, value_logits = network(states)
    policy_loss = -(
        policies * F.log_softmax(policy_logits, dim=1)
    ).sum(dim=1).mean()
    value_loss = F.binary_cross_entropy_with_logits(
        value_logits, win_probabilities
    )
    total_loss = policy_loss + value_loss
    probabilities = value_logit_to_probability(value_logits)
    brier = torch.mean(torch.square(probabilities - win_probabilities))
    return policy_loss, value_loss, total_loss, brier


def train_duplicate_aware_updates(state, config, device, view):
    totals = np.zeros(4, dtype=np.float64)
    state.network.train()
    for _ in range(config.training_updates_per_iteration):
        batch = view.sample(config.batch_size, state.rng)
        states = torch.from_numpy(np.stack([sample.state for sample in batch])).to(
            device
        )
        policies = torch.from_numpy(
            np.stack([sample.policy for sample in batch])
        ).to(device)
        targets = torch.tensor(
            [sample.win_probability for sample in batch],
            dtype=torch.float32,
            device=device,
        )
        state.optimizer.zero_grad(set_to_none=True)
        losses = value_loss_components(state.network, states, policies, targets)
        if not all(torch.isfinite(loss) for loss in losses):
            raise RuntimeError("non-finite V4 training loss")
        losses[2].backward()
        state.optimizer.step()
        totals += np.asarray([float(loss.item()) for loss in losses])
    totals /= config.training_updates_per_iteration
    return {
        "policy_loss": float(totals[0]),
        "value_bce_loss": float(totals[1]),
        "total_loss": float(totals[2]),
        "brier_score": float(totals[3]),
    }


def _optimizer(network, config):
    return torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _checkpoint_payload(state, config):
    payload = {
        "format_version": V4_CHECKPOINT_FORMAT_VERSION,
        "architecture": "alphazero_v4_raw_value_logit",
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
        payload["cuda_rng_state_all"] = [
            value.detach().cpu().to(dtype=torch.uint8)
            for value in torch.cuda.get_rng_state_all()
        ]
    return payload


def save_checkpoint(run_dir, state, config, *, iteration_checkpoint=True):
    run_dir = Path(run_dir)
    state.replay.generation_metadata = {
        "iteration": state.iteration,
        "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
    }
    _atomic_torch_save(state.replay.state_dict(), run_dir / "replay_buffer.pt")
    payload = _checkpoint_payload(state, config)
    _atomic_torch_save(payload, run_dir / "latest.pt")
    if iteration_checkpoint and state.iteration > 0:
        _atomic_torch_save(
            payload,
            run_dir
            / "checkpoints"
            / f"iteration_{state.iteration:06d}.pt",
        )
    retain_iteration_checkpoints(run_dir, config)


def _write_summary(run_dir, state, config, device):
    _atomic_json_save(
        {
            "iteration": state.iteration,
            "total_self_play_games": state.total_self_play_games,
            "total_samples_generated": state.total_samples_generated,
            "replay_size": len(state.replay),
            "elapsed_seconds": state.elapsed_seconds,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "network_parameters": state.network.parameter_count(),
            "config": config.to_dict(),
            "last_metric": state.last_metric,
        },
        Path(run_dir) / "summary.json",
    )


def _append_unique(path, record):
    path = Path(path)
    completed = set()
    if path.exists():
        completed = {
            int(json.loads(line)["iteration"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if int(record["iteration"]) not in completed:
        append_metric(path, record)


def initialize_run(run_dir, config, device):
    run_dir = Path(run_dir)
    if (run_dir / "latest.pt").exists():
        raise FileExistsError(f"V4 run exists; use --resume {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _atomic_json_save(config.to_dict(), run_dir / "config.json")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    network = PolicyValueLogitNetwork(
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        input_planes=config.input_planes,
    ).to(device)
    state = V4TrainingState(
        network=network,
        optimizer=_optimizer(network, config),
        replay=ReplayBuffer(
            config.replay_max_positions, encoded_shape=ENCODED_SHAPE
        ),
        rng=np.random.default_rng(config.seed),
    )
    diagnostic = run_fixed_tactical_diagnostics(state, config, device)
    _append_unique(run_dir / "diagnostics.jsonl", diagnostic)
    save_checkpoint(run_dir, state, config, iteration_checkpoint=False)
    _write_summary(run_dir, state, config, device)
    return state


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    config = V4Config.from_dict(
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    )
    payload = torch.load(
        run_dir / "latest.pt", map_location=device, weights_only=False
    )
    if payload.get("format_version") != V4_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported V4 checkpoint format")
    if payload.get("architecture") != "alphazero_v4_raw_value_logit":
        raise ValueError("checkpoint is not a V4 raw-logit network")
    if payload["config"] != config.to_dict():
        raise ValueError("V4 config.json and checkpoint disagree")
    replay_payload = torch.load(
        run_dir / "replay_buffer.pt", map_location="cpu", weights_only=False
    )
    replay = ReplayBuffer.from_state_dict(replay_payload)
    if replay.encoded_shape != ENCODED_SHAPE:
        raise ValueError("V4 replay encoder shape mismatch")
    if int(replay.generation_metadata["iteration"]) < int(payload["iteration"]):
        raise RuntimeError("V4 replay is older than latest checkpoint")
    network = PolicyValueLogitNetwork(
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        input_planes=config.input_planes,
    ).to(device)
    optimizer = _optimizer(network, config)
    network.load_state_dict(payload["network_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(
            [
                value.detach().cpu().to(dtype=torch.uint8)
                for value in payload["cuda_rng_state_all"]
            ]
        )
    return config, V4TrainingState(
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


def run_iteration(run_dir, state, config, device, oracle_replay_path):
    iteration_started = time.perf_counter()
    self_play_started = time.perf_counter()
    examples, games = generate_self_play(state.network, config, device, state.rng)
    self_play_seconds = time.perf_counter() - self_play_started
    if not examples or any(
        game["illegal_probability_violations"] for game in games
    ):
        raise RuntimeError("invalid V4 self-play output")
    state.replay.extend(examples)
    view_started = time.perf_counter()
    view = DuplicateAwareTrainingView(state.replay.samples)
    duplicate_metrics = view.metrics()
    view_seconds = time.perf_counter() - view_started

    training_started = time.perf_counter()
    losses = train_duplicate_aware_updates(state, config, device, view)
    training_seconds = time.perf_counter() - training_started
    state.iteration += 1
    state.total_self_play_games += len(games)
    state.total_samples_generated += len(examples)

    tactical = None
    if state.iteration in TACTICAL_DIAGNOSTIC_ITERATIONS:
        tactical = run_fixed_tactical_diagnostics(state, config, device)
        _append_unique(Path(run_dir) / "diagnostics.jsonl", tactical)
    oracle = None
    oracle_seconds = 0.0
    if state.iteration in VALUE_ORACLE_ITERATIONS:
        oracle_started = time.perf_counter()
        oracle = run_value_oracle_monitor(
            state,
            config,
            device,
            oracle_replay_path,
        )
        oracle_seconds = time.perf_counter() - oracle_started
        _append_unique(Path(run_dir) / "value_oracle.jsonl", oracle)

    iteration_seconds = time.perf_counter() - iteration_started
    state.elapsed_seconds += iteration_seconds
    winners = [game["winner"] for game in games]
    reasons = [game["terminal_reason"] for game in games]
    tactical_totals = {
        name: int(sum(game[name] for game in games))
        for name in (
            "immediate_win_opportunities",
            "immediate_win_taken",
            "defense_threat_states",
            "defense_states_with_safe_action",
            "unsafe_actions_filtered",
            "forced_loss_states",
        )
    }
    targets = np.asarray([example.value for example in examples], dtype=np.float32)
    metric = {
        "iteration": state.iteration,
        "elapsed_seconds": state.elapsed_seconds,
        "iteration_seconds": iteration_seconds,
        "self_play_seconds": self_play_seconds,
        "training_view_seconds": view_seconds,
        "training_seconds": training_seconds,
        "value_oracle_seconds": oracle_seconds,
        "total_self_play_games": state.total_self_play_games,
        "new_games": len(games),
        "new_samples": len(examples),
        "total_samples_generated": state.total_samples_generated,
        "replay_size": len(state.replay),
        "blue_wins": winners.count(1),
        "red_wins": winners.count(2),
        "capture_endings": reasons.count("CAPTURE_WIN"),
        "pass_score_endings": reasons.count("PASS_SCORE_END"),
        "pass_action_count": int(sum(game["pass_usage"] for game in games)),
        "mean_game_length": float(np.mean([game["game_length"] for game in games])),
        "mean_pass_count": float(np.mean([game["pass_usage"] for game in games])),
        "value_target_mean": float(np.mean(targets)),
        "value_target_std": float(np.std(targets)),
        **tactical_totals,
        **duplicate_metrics,
        **losses,
        "training_updates": config.training_updates_per_iteration,
        "mcts_simulations": config.mcts_simulations,
        "temperature_schedule": config.temperature_schedule,
        "gpu_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "games": games,
        "tactical_diagnostic": tactical,
        "value_oracle_monitor": oracle,
    }
    state.last_metric = metric
    save_checkpoint(run_dir, state, config)
    append_metric(Path(run_dir) / "metrics.jsonl", metric)
    _write_summary(run_dir, state, config, device)
    return metric


def run_to_iteration(
    run_dir,
    state,
    config,
    device,
    max_iterations,
    oracle_replay_path,
):
    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if state.iteration > max_iterations:
        raise ValueError("run already exceeds requested V4 iteration")
    metrics = []
    while state.iteration < max_iterations:
        metric = run_iteration(
            run_dir,
            state,
            config,
            device,
            oracle_replay_path,
        )
        metrics.append(metric)
        print(
            json.dumps(
                {
                    key: metric[key]
                    for key in (
                        "iteration",
                        "iteration_seconds",
                        "total_self_play_games",
                        "new_samples",
                        "capture_endings",
                        "pass_score_endings",
                        "immediate_win_opportunities",
                        "defense_states_with_safe_action",
                        "unsafe_actions_filtered",
                        "contradictory_z_group_count",
                        "policy_loss",
                        "value_bce_loss",
                        "brier_score",
                        "total_loss",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return metrics


def latest_completed_iteration(run_dir):
    payload = torch.load(
        Path(run_dir) / "latest.pt", map_location="cpu", weights_only=False
    )
    return int(payload["iteration"])
