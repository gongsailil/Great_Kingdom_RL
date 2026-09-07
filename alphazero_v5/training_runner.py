"""Crash-safe BEST/candidate AlphaZero V5 training cycles."""

from dataclasses import dataclass
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from alphazero_v2.training_runner import _atomic_json_save, _atomic_torch_save, append_metric
from alphazero_v4.acceptance import generate_acceptance_openings

from .arena import ArenaAgent, play_paired_arena, summarize_arena
from .calibration import fit_value_temperature
from .config import V5Config
from .diagnostics import run_fixed_tactical_diagnostics, symmetry_consistency
from .encoder import encode_state
from .network import (
    PolicyValueAuxNetwork,
    calibrated_scalar,
    liberty_prediction,
    score_prediction,
)
from .replay import CompactReplayBuffer, DuplicateAwareSplitView
from .self_play import generate_self_play
from .start_states import load_v4_territory_pool


FORMAT_VERSION = 1
ARCHITECTURE = "alphazero_v5_strategy_12plane_128x6"


@dataclass
class V5TrainingState:
    best_network: PolicyValueAuxNetwork
    replay: CompactReplayBuffer
    rng: np.random.Generator
    cycle: int = 0
    best_version: int = 0
    promotion_count: int = 0
    total_self_play_games: int = 0
    total_samples_generated: int = 0
    calibration_temperature: float = 1.0
    elapsed_seconds: float = 0.0
    last_metric: dict | None = None


def choose_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def make_network(config, device):
    return PolicyValueAuxNetwork(config.channels, config.residual_blocks, config.input_planes).to(device)


def clone_candidate(best, config, device):
    candidate = make_network(config, device)
    candidate.load_state_dict(copy.deepcopy(best.state_dict()))
    return candidate


def network_digest(network):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loss_components(network, states, policies, value_targets, liberty_targets, score_targets):
    policy_logits, value_logits, liberty_logits, score_logits = network(states)
    policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.binary_cross_entropy_with_logits(value_logits, value_targets)
    liberty_loss = F.mse_loss(liberty_prediction(liberty_logits), liberty_targets)
    score_loss = F.mse_loss(score_prediction(score_logits), score_targets)
    total_loss = policy_loss + value_loss + liberty_loss + score_loss
    brier = torch.mean(torch.square(torch.sigmoid(value_logits) - value_targets))
    return policy_loss, value_loss, liberty_loss, score_loss, total_loss, brier


def train_candidate(candidate, view, config, device, rng):
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    totals = np.zeros(6, dtype=np.float64)
    candidate.train()
    for _ in range(config.training_updates_per_cycle):
        batch = view.sample_training_batch(config.batch_size, rng, config.d4_augmentation)
        tensors = [torch.from_numpy(item).to(device) for item in batch]
        optimizer.zero_grad(set_to_none=True)
        losses = loss_components(candidate, *tensors)
        if not all(torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("non-finite V5 candidate loss")
        losses[4].backward()
        optimizer.step()
        totals += np.asarray([float(loss.item()) for loss in losses])
    totals /= config.training_updates_per_cycle
    return {
        "policy_loss": float(totals[0]), "value_bce_loss": float(totals[1]),
        "liberty_loss": float(totals[2]), "score_loss": float(totals[3]),
        "total_loss": float(totals[4]), "training_brier": float(totals[5]),
        "training_updates": config.training_updates_per_cycle,
    }, optimizer


def _value_transform(temperature):
    return lambda logits: calibrated_scalar(logits, temperature)


def _agent(name, network, temperature, config, device):
    return ArenaAgent(name, network, encode_state, _value_transform(temperature), device,
                      config.arena_mcts_simulations, config.c_puct, True)


def should_promote(arena, tactical, threshold):
    return bool(
        arena["win_rate"] >= float(threshold)
        and arena.get("illegal_violations", 0) == 0
        and arena.get("tactical_failures", 0) == 0
        and tactical["all_success"]
    )


def aggregate_self_play(games):
    total_moves = sum(game["game_length"] for game in games)
    keys = (
        "immediate_win_opportunities", "immediate_win_taken", "defense_threat_states",
        "defense_states_with_safe_action", "unsafe_actions_filtered", "forced_loss_states",
        "pass_usage", "early_placements", "early_edge_placements", "placements",
        "own_adjacent_placements", "own_adjacent_two_plus", "territory_creation_delta",
        "territory_disruption_delta", "states_with_nonzero_territory",
        "network_value_count", "network_value_abs_ge_0_9", "network_value_abs_ge_0_99",
    )
    totals = {key: int(sum(game[key] for game in games)) for key in keys}
    return {
        "new_games": len(games), "new_samples": total_moves,
        "initial_start_games": sum(game["start_type"] == "initial" for game in games),
        "territory_start_games": sum(game["start_type"] == "territory_midgame" for game in games),
        "blue_wins": sum(game["winner"] == 1 for game in games),
        "red_wins": sum(game["winner"] == 2 for game in games),
        "capture_endings": sum(game["terminal_reason"] == "CAPTURE_WIN" for game in games),
        "pass_score_endings": sum(game["terminal_reason"] == "PASS_SCORE_END" for game in games),
        "mean_game_length": float(np.mean([game["game_length"] for game in games])),
        "games_entering_forced_loss": sum(game["entered_forced_loss"] for game in games),
        "mean_policy_entropy": float(np.mean([game["mean_policy_entropy"] for game in games])),
        "illegal_violations": int(sum(game["illegal_probability_violations"] for game in games)),
        **totals,
        "early_edge_fraction": totals["early_edge_placements"] / max(1, totals["early_placements"]),
        "own_adjacent_fraction": totals["own_adjacent_placements"] / max(1, totals["placements"]),
        "own_adjacent_two_plus_fraction": totals["own_adjacent_two_plus"] / max(1, totals["placements"]),
        "territory_state_fraction": totals["states_with_nonzero_territory"] / max(1, total_moves),
        "forced_loss_entry_rate": sum(game["entered_forced_loss"] for game in games) / len(games),
        "value_abs_ge_0_9_fraction": totals["network_value_abs_ge_0_9"] / max(1, totals["network_value_count"]),
        "value_abs_ge_0_99_fraction": totals["network_value_abs_ge_0_99"] / max(1, totals["network_value_count"]),
    }


def _checkpoint_payload(state, config):
    payload = {
        "format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
        "network_state_dict": state.best_network.state_dict(), "config": config.to_dict(),
        "cycle": state.cycle, "best_version": state.best_version,
        "promotion_count": state.promotion_count,
        "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
        "calibration_temperature": state.calibration_temperature,
        "elapsed_seconds": state.elapsed_seconds, "last_metric": state.last_metric,
        "numpy_rng_state": state.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = [s.detach().cpu().to(torch.uint8) for s in torch.cuda.get_rng_state_all()]
    return payload


def save_run(run_dir, state, config, promoted=False):
    run_dir = Path(run_dir)
    state.replay.generation_metadata = {
        "cycle": state.cycle, "best_version": state.best_version,
        "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
    }
    replay_path = run_dir / "replay_buffer.pt"
    previous_replay = run_dir / "replay_buffer_previous.pt"
    if replay_path.exists():
        if previous_replay.exists():
            previous_replay.unlink()
        os.link(replay_path, previous_replay)
    _atomic_torch_save(state.replay.state_dict(), replay_path)
    payload = _checkpoint_payload(state, config)
    _atomic_torch_save(payload, run_dir / "latest.pt")
    if promoted:
        _atomic_torch_save(payload, run_dir / "checkpoints" / f"best_{state.best_version:03d}_cycle_{state.cycle:04d}.pt")
    _atomic_json_save({
        "cycle": state.cycle, "best_version": state.best_version,
        "promotion_count": state.promotion_count,
        "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
        "replay": state.replay.metadata(), "calibration_temperature": state.calibration_temperature,
        "elapsed_seconds": state.elapsed_seconds, "network_parameters": state.best_network.parameter_count(),
        "last_metric": state.last_metric,
    }, run_dir / "summary.json")


def initialize_run(run_dir, config, device):
    run_dir = Path(run_dir)
    if (run_dir / "latest.pt").exists():
        raise FileExistsError(f"V5 run exists; use --resume {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _atomic_json_save(config.to_dict(), run_dir / "config.json")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    state = V5TrainingState(make_network(config, device), CompactReplayBuffer(config.replay_max_positions),
                            np.random.default_rng(config.seed))
    save_run(run_dir, state, config)
    return state


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    config = V5Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    payload = torch.load(run_dir / "latest.pt", map_location=device, weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("not a supported V5 checkpoint")
    if payload["config"] != config.to_dict():
        raise ValueError("V5 config/checkpoint mismatch")
    replay = None
    for replay_path in (run_dir / "replay_buffer.pt", run_dir / "replay_buffer_previous.pt"):
        if not replay_path.exists():
            continue
        candidate_replay = CompactReplayBuffer.from_state_dict(
            torch.load(replay_path, map_location="cpu", weights_only=False))
        if int(candidate_replay.generation_metadata.get("cycle", -1)) == int(payload["cycle"]):
            replay = candidate_replay
            break
    if replay is None:
        raise RuntimeError("no V5 replay matches the last completed checkpoint boundary")
    network = make_network(config, device)
    network.load_state_dict(payload["network_state_dict"])
    rng = np.random.default_rng(); rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all([s.detach().cpu().to(torch.uint8) for s in payload["cuda_rng_state_all"]])
    state = V5TrainingState(
        network, replay, rng, int(payload["cycle"]), int(payload["best_version"]),
        int(payload["promotion_count"]), int(payload["total_self_play_games"]),
        int(payload["total_samples_generated"]), float(payload["calibration_temperature"]),
        float(payload["elapsed_seconds"]), payload.get("last_metric"),
    )
    return config, state


def run_cycle(run_dir, state, config, device, start_pool, arena_openings):
    started = time.perf_counter()
    before_digest = network_digest(state.best_network)
    selfplay_started = time.perf_counter()
    examples, games, inference = generate_self_play(
        state.best_network, state.calibration_temperature, config, device, state.rng, start_pool)
    selfplay_seconds = time.perf_counter() - selfplay_started
    if not examples:
        raise RuntimeError("V5 self-play generated no samples")
    behavior = aggregate_self_play(games)
    if behavior["illegal_violations"]:
        raise RuntimeError("V5 self-play illegal-action violation")
    if network_digest(state.best_network) != before_digest:
        raise RuntimeError("frozen BEST mutated during self-play")
    state.replay.extend(examples)
    view_started = time.perf_counter()
    view = DuplicateAwareSplitView(state.replay, config.validation_fraction)
    duplicate = view.metrics()
    view_seconds = time.perf_counter() - view_started

    candidate = clone_candidate(state.best_network, config, device)
    training_started = time.perf_counter()
    losses, _ = train_candidate(candidate, view, config, device, state.rng)
    training_seconds = time.perf_counter() - training_started
    calibration = fit_value_temperature(candidate, view, device)
    tactical = run_fixed_tactical_diagnostics(candidate, calibration["temperature"], config, device)
    if not tactical["all_success"]:
        raise RuntimeError("V5 candidate exact tactical regression")

    arena_started = time.perf_counter()
    candidate_name = f"candidate_cycle_{state.cycle + 1}"
    best_name = f"best_v{state.best_version}"
    arena_games, arena_inference = play_paired_arena(
        _agent(candidate_name, candidate, calibration["temperature"], config, device),
        _agent(best_name, state.best_network, state.calibration_temperature, config, device),
        arena_openings, config.concurrent_games)
    arena = summarize_arena(arena_games, candidate_name)
    arena_seconds = time.perf_counter() - arena_started
    promoted = should_promote(arena, tactical, config.promotion_win_rate)
    if promoted:
        state.best_network = candidate
        state.calibration_temperature = float(calibration["temperature"])
        state.best_version += 1
        state.promotion_count += 1
    elif network_digest(state.best_network) != before_digest:
        raise RuntimeError("rejected candidate changed frozen BEST")

    state.cycle += 1
    state.total_self_play_games += len(games)
    state.total_samples_generated += len(examples)
    cycle_seconds = time.perf_counter() - started
    state.elapsed_seconds += cycle_seconds
    symmetry = symmetry_consistency(state.best_network, state.calibration_temperature, device, arena_openings)
    metric = {
        "cycle": state.cycle, "candidate_version": state.cycle,
        "best_version": state.best_version, "promotion_count": state.promotion_count,
        "promoted": bool(promoted), "total_self_play_games": state.total_self_play_games,
        "total_samples_generated": state.total_samples_generated,
        "replay_size": len(state.replay), "cycle_seconds": cycle_seconds,
        "self_play_seconds": selfplay_seconds, "training_view_seconds": view_seconds,
        "training_seconds": training_seconds, "arena_seconds": arena_seconds,
        "calibration_temperature": state.calibration_temperature,
        "candidate_calibration": calibration, "self_play_inference": inference,
        "arena_inference": arena_inference, "behavior": behavior,
        "duplicates": duplicate, "losses": losses, "candidate_arena": arena,
        "tactical": tactical, "symmetry": symmetry,
    }
    state.last_metric = metric
    save_run(run_dir, state, config, promoted=promoted)
    append_metric(Path(run_dir) / "metrics.jsonl", metric)
    append_metric(Path(run_dir) / "promotion_history.jsonl", {
        "cycle": state.cycle, "candidate": candidate_name, "best_before": best_name,
        "candidate_wins": arena["wins"], "games": arena["games"],
        "win_rate": arena["win_rate"], "threshold": config.promotion_win_rate,
        "promoted": bool(promoted), "best_version_after": state.best_version,
    })
    return metric


def prepare_run_assets(run_dir, config, v4_replay_path):
    run_dir = Path(run_dir)
    opening_path = run_dir / "promotion_openings.json"
    if not opening_path.exists():
        _atomic_json_save(generate_acceptance_openings(config.candidate_arena_openings, config.seed), opening_path)
    opening_payload = json.loads(opening_path.read_text())
    start_pool = load_v4_territory_pool(v4_replay_path, config.territory_pool_max_states, config.seed)
    return start_pool, opening_payload["openings"]


def run_until_target(run_dir, state, config, device, start_pool, openings):
    metrics = []
    while (state.total_self_play_games < config.target_self_play_games and
           state.promotion_count < config.target_promotions):
        metric = run_cycle(run_dir, state, config, device, start_pool, openings)
        metrics.append(metric)
        print(json.dumps({
            "cycle": metric["cycle"], "best_version": metric["best_version"],
            "promoted": metric["promoted"], "candidate_wins": metric["candidate_arena"]["wins"],
            "games": metric["total_self_play_games"], "samples": metric["total_samples_generated"],
            "replay": metric["replay_size"], "cycle_seconds": metric["cycle_seconds"],
        }, sort_keys=True), flush=True)
    return metrics


def run_throughput_measurement(network, temperature, config, device, rng, start_pool, games=64):
    if int(games) != 64:
        raise ValueError("engineering throughput gate is fixed at 64 games")
    started = time.perf_counter()
    examples, results, inference = generate_self_play(network, temperature, config, device, rng, start_pool, games)
    elapsed = time.perf_counter() - started
    behavior = aggregate_self_play(results)
    return {
        "games": len(results), "positions": len(examples), "elapsed_seconds": elapsed,
        "games_per_second": len(results) / elapsed, "positions_per_second": len(examples) / elapsed,
        "configured_concurrent_games": config.concurrent_games,
        **inference, "behavior": behavior,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
