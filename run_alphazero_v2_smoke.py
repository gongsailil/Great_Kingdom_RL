"""Run one bounded AlphaZero V2 self-play/training/checkpoint smoke."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from alphazero_v2.config import AlphaZeroConfig
from alphazero_v2.encoder import encode_state
from alphazero_v2.network import PolicyValueNetwork
from alphazero_v2.self_play import generate_self_play
from alphazero_v2.train import (
    load_checkpoint,
    save_checkpoint,
    train_on_examples,
)
from great_kingdom_v2 import GreatKingdomLogicV2, NUM_ACTIONS


DEFAULT_CHECKPOINT = Path("models/alphazero_v2/minimal_e2e.pt")
DEFAULT_REPORT_DIR = Path("reports/alphazero_v2_minimal_e2e_20260830")


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def outputs_for_initial_state(network, device):
    inputs = torch.from_numpy(encode_state(GreatKingdomLogicV2()))
    inputs = inputs.unsqueeze(0).to(device)
    network.eval()
    with torch.no_grad():
        logits, value = network(inputs)
    return logits.detach().cpu(), value.detach().cpu()


def write_report(report_dir, summary):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terminal_text = ", ".join(
        f"{reason}={count}"
        for reason, count in sorted(summary["games_terminal_reason"].items())
    )
    lines = [
        "# AlphaZero V2 Minimal E2E",
        "",
        f"Result: {'PASS' if summary['minimal_e2e_pass'] else 'FAIL'}",
        f"Device: {summary['device']}",
        f"Network parameters: {summary['network_params_count']}",
        f"MCTS simulations: {summary['mcts_simulations']}",
        f"Self-play games: {summary['self_play_games']}",
        f"Terminal reasons: {terminal_text}",
        f"Generated samples: {summary['generated_sample_count']}",
        f"Mean game length: {summary['mean_game_length']:.3f}",
        f"PASS usage: {summary['pass_usage_count']}",
        (
            "Loss before: "
            f"policy={summary['loss_before']['policy']:.6f}, "
            f"value={summary['loss_before']['value']:.6f}, "
            f"total={summary['loss_before']['total']:.6f}"
        ),
        (
            "Loss after: "
            f"policy={summary['loss_after']['policy']:.6f}, "
            f"value={summary['loss_after']['value']:.6f}, "
            f"total={summary['loss_after']['total']:.6f}"
        ),
        (
            "Illegal probability violations: "
            f"{summary['illegal_probability_violations']}"
        ),
        f"Checkpoint save/load: {summary['save_load_test']}",
        f"Checkpoint: {summary['checkpoint_path']}",
        f"Elapsed seconds: {summary['elapsed_seconds']:.3f}",
        "",
        "This smoke validates pipeline correctness only; it is not a strength claim.",
    ]
    (report_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_smoke(args):
    started = time.perf_counter()
    config = AlphaZeroConfig(
        mcts_simulations=args.simulations,
        self_play_games=args.games,
        training_epochs=args.epochs,
        seed=args.seed,
    )
    device = choose_device(args.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    network = PolicyValueNetwork(
        channels=config.channels,
        residual_blocks=config.residual_blocks,
    ).to(device)
    # Explicit device forward smoke before search starts.
    outputs_for_initial_state(network, device)

    rng = np.random.default_rng(config.seed)
    examples, game_stats = generate_self_play(network, config, device, rng)
    if not examples:
        raise RuntimeError("self-play generated no training examples")
    for example in examples:
        if example.policy.shape != (NUM_ACTIONS,):
            raise RuntimeError("self-play policy has wrong shape")
        if not np.isclose(example.policy.sum(), 1.0):
            raise RuntimeError("self-play policy is not normalized")
        if example.value not in (-1.0, 1.0):
            raise RuntimeError("self-play value target is not +/-1")

    optimizer, losses_before, losses_after = train_on_examples(
        network,
        examples,
        config,
        device,
    )
    training_metadata = {
        "device": str(device),
        "generated_sample_count": len(examples),
        "loss_before": losses_before,
        "loss_after": losses_after,
        "self_play_games": len(game_stats),
    }
    before_load = outputs_for_initial_state(network, device)
    save_checkpoint(
        args.checkpoint,
        network,
        optimizer,
        config,
        training_metadata,
    )
    loaded_network, _, loaded_config, loaded_metadata = load_checkpoint(
        args.checkpoint,
        device,
    )
    after_load = outputs_for_initial_state(loaded_network, device)
    save_load_pass = (
        loaded_config.to_dict() == config.to_dict()
        and loaded_metadata == training_metadata
        and torch.allclose(before_load[0], after_load[0], atol=1e-7, rtol=1e-7)
        and torch.allclose(before_load[1], after_load[1], atol=1e-7, rtol=1e-7)
    )

    terminal_reasons = Counter(item["terminal_reason"] for item in game_stats)
    illegal_violations = sum(
        item["illegal_probability_violations"] for item in game_stats
    )
    pass_usage = sum(item["pass_usage"] for item in game_stats)
    all_terminal = all(item["terminal_reason"] is not None for item in game_stats)
    loss_decreased = losses_after["total"] < losses_before["total"]
    minimal_e2e_pass = all(
        (
            all_terminal,
            len(examples) > 0,
            illegal_violations == 0,
            all(example.value in (-1.0, 1.0) for example in examples),
            np.isfinite(losses_before["total"]),
            np.isfinite(losses_after["total"]),
            loss_decreased,
            save_load_pass,
        )
    )
    elapsed = time.perf_counter() - started
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    summary = {
        "checkpoint_path": str(args.checkpoint),
        "device": str(device),
        "device_name": device_name,
        "elapsed_seconds": elapsed,
        "games": game_stats,
        "games_terminal_reason": dict(terminal_reasons),
        "generated_sample_count": len(examples),
        "illegal_probability_violations": illegal_violations,
        "loss_after": losses_after,
        "loss_before": losses_before,
        "loss_decreased": loss_decreased,
        "mean_game_length": float(
            np.mean([item["game_length"] for item in game_stats])
        ),
        "mcts_simulations": config.mcts_simulations,
        "minimal_e2e_pass": minimal_e2e_pass,
        "network_params_count": network.parameter_count(),
        "pass_usage_count": pass_usage,
        "save_load_test": "PASS" if save_load_pass else "FAIL",
        "self_play_games": config.self_play_games,
    }
    write_report(args.report_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not minimal_e2e_pass:
        raise RuntimeError("AlphaZero V2 minimal E2E criteria failed")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_smoke(parse_args())
