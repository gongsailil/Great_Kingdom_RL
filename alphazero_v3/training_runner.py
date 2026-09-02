"""Bounded V3 territory pilot built from the sustained V2 runner."""

import json
from pathlib import Path

import numpy as np

from great_kingdom_v2 import PASS_ACTION

from alphazero_v2.training_runner import (
    append_metric,
    initialize_run,
    load_run,
    run_iteration,
)

from .config import TerritoryPilotConfig
from .diagnostics import run_fixed_diagnostics
from .encoder import encode_state


PILOT_DIAGNOSTIC_ITERATIONS = (0, 10, 20, 30, 40, 50)


def _sample_signal_metrics(examples):
    states = np.stack([example.state for example in examples])
    policies = np.stack([example.policy for example in examples])
    current_territory = np.any(states[:, 7] > 0.0, axis=(1, 2))
    opponent_territory = np.any(states[:, 8] > 0.0, axis=(1, 2))
    any_territory = current_territory | opponent_territory
    pass_targets = policies[:, PASS_ACTION]
    pass_nonzero = pass_targets > 0.0

    if np.any(any_territory):
        territory_pass_targets = pass_targets[any_territory]
        territory_mean_pass = float(np.mean(territory_pass_targets))
        territory_nonzero_pass = float(
            np.mean(territory_pass_targets > 0.0)
        )
    else:
        territory_mean_pass = 0.0
        territory_nonzero_pass = 0.0

    count = len(examples)
    return {
        "states_with_current_territory": int(np.sum(current_territory)),
        "states_with_current_territory_fraction": float(
            np.mean(current_territory)
        ),
        "states_with_opponent_territory": int(np.sum(opponent_territory)),
        "states_with_opponent_territory_fraction": float(
            np.mean(opponent_territory)
        ),
        "territory_state_count": int(np.sum(any_territory)),
        "territory_state_fraction": float(np.mean(any_territory)),
        "mean_pass_target_probability": float(np.mean(pass_targets)),
        "pass_target_nonzero_fraction": float(np.mean(pass_nonzero)),
        "territory_mean_pass_target_probability": territory_mean_pass,
        "territory_pass_target_nonzero_fraction": territory_nonzero_pass,
        "signal_sample_count": count,
    }


def enrich_pilot_metric(state, config, device, examples, games):
    metrics = _sample_signal_metrics(examples)
    metrics["pass_action_count"] = int(
        sum(game["pass_usage"] for game in games)
    )
    metrics["illegal_action_probability_violations"] = int(
        sum(game["illegal_probability_violations"] for game in games)
    )
    if state.iteration % config.diagnostic_interval == 0:
        metrics["tactical_diagnostic"] = run_fixed_diagnostics(
            state,
            config,
            device,
        )
    return metrics


def _read_diagnostic_iterations(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {
        int(json.loads(line)["iteration"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _append_diagnostic_once(run_dir, diagnostic):
    path = Path(run_dir) / "diagnostics.jsonl"
    if diagnostic["iteration"] not in _read_diagnostic_iterations(path):
        append_metric(path, diagnostic)


def synchronize_diagnostics(run_dir, state, config, device):
    """Recover diagnostic JSONL from atomic checkpoints/metrics when needed."""
    run_dir = Path(run_dir)
    if state.iteration == 0:
        path = run_dir / "diagnostics.jsonl"
        if 0 not in _read_diagnostic_iterations(path):
            _append_diagnostic_once(
                run_dir,
                run_fixed_diagnostics(state, config, device),
            )

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metric = json.loads(line)
            diagnostic = metric.get("tactical_diagnostic")
            if diagnostic is not None:
                _append_diagnostic_once(run_dir, diagnostic)


def initialize_pilot(run_dir, config, device):
    return initialize_run(run_dir, config, device)


def load_pilot(run_dir, device):
    return load_run(run_dir, device, config_class=TerritoryPilotConfig)


def run_pilot(run_dir, state, config, device, max_iterations):
    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if state.iteration > max_iterations:
        raise ValueError("run is already beyond requested max_iterations")

    synchronize_diagnostics(run_dir, state, config, device)
    completed = []
    while state.iteration < max_iterations:
        metric = run_iteration(
            run_dir,
            state,
            config,
            device,
            state_encoder=encode_state,
            metric_enricher=enrich_pilot_metric,
        )
        diagnostic = metric.get("tactical_diagnostic")
        if diagnostic is not None:
            _append_diagnostic_once(run_dir, diagnostic)
        completed.append(metric)
        progress = {
            key: metric[key]
            for key in (
                "iteration",
                "iteration_seconds",
                "total_self_play_games",
                "new_samples",
                "capture_endings",
                "pass_score_endings",
                "pass_action_count",
                "territory_state_fraction",
                "mean_pass_target_probability",
                "pass_target_nonzero_fraction",
                "policy_loss",
                "value_loss",
                "total_loss",
            )
        }
        print(json.dumps(progress, sort_keys=True), flush=True)
    return completed
