"""Audit V3 iteration-10/50 value predictions against exact one-ply outcomes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from alphazero_v2.evaluate import load_evaluation_checkpoint
from alphazero_v3.encoder import encode_state
from alphazero_v3.temperature_audit import network_state_digest
from alphazero_v3.value_oracle_audit import (
    audit_networks_on_states,
    classify_value_audit,
    select_audit_indices,
    validate_replay,
)


DEFAULT_RUN_DIR = Path("runs/alphazero_v3/territory_pilot_20260901")
DEFAULT_REPORT_DIR = Path("reports/alphazero_value_oracle_audit_20260902")
FIXED_SEED = 20260902
MAX_AUDITED_STATES = 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path, content):
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _percent(value):
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def render_summary(summary):
    lines = [
        "AlphaZero V3 exact one-ply value-oracle audit (2026-09-02)",
        "=" * 64,
        "",
        "No training, self-play, MCTS, or checkpoint mutation was performed.",
        f"Replay states: {summary['replay_validation']['total_states']}",
        f"Audited states: {summary['audited_states']} (seed {summary['seed']})",
        f"Audited turns Blue/Red: {summary['audited_player_counts']['blue']}/"
        f"{summary['audited_player_counts']['red']}",
        f"Exact -1 actions: {summary['state_class_counts']['exact_loss_actions']}",
        f"Device: {summary['device']}",
        "",
        "Exact -1 calibration",
        "--------------------",
        "Iter  Count  Mean Q  Median  p10  p90  MAE  MSE  Q>0  Q>=0.5",
    ]
    for iteration in (10, 50):
        data = summary["networks"][str(iteration)]["exact_loss"]
        lines.append(
            f"{iteration:>4}  {data['count']:>5}  {data['mean']:+.4f}  "
            f"{data['median']:+.4f}  {data['p10']:+.4f}  {data['p90']:+.4f}  "
            f"{data['mae']:.4f}  {data['mse']:.4f}  "
            f"{_percent(data['positive_fraction']):>7}  "
            f"{_percent(data['at_least_0_5_fraction']):>7}"
        )

    lines.extend(["", "Defense value ranking", "---------------------"])
    for iteration in (10, 50):
        data = summary["networks"][str(iteration)]["defense"]
        lines.append(
            f"iter {iteration}: opportunities={data['opportunity_states']}, "
            f"rankable={data['rankable_states']}, failures="
            f"{data['ranking_failures']} "
            f"({_percent(data['ranking_failure_fraction'])}), mean(max unsafe - "
            f"max safe)={data['mean_unsafe_minus_safe_max_q']:+.6f}"
        )

    lines.extend(["", "Immediate-win alternative saturation", "------------------------------------"])
    for iteration in (10, 50):
        data = summary["networks"][str(iteration)][
            "immediate_win_alternatives"
        ]
        lines.append(
            f"iter {iteration}: win states={data['immediate_win_state_count']}, "
            f"alternatives={data['nonterminal_alternative_count']}, mean/max Q="
            f"{data['mean_alternative_predicted_q']:+.6f}/"
            f"{data['max_alternative_predicted_q']:+.6f}; >=0.9/0.95/0.99="
            f"{data['at_least_0_9_count']}/"
            f"{data['at_least_0_95_count']}/"
            f"{data['at_least_0_99_count']}"
        )

    lines.extend(["", "Color split for exact -1 actions", "--------------------------------"])
    for iteration in (10, 50):
        split = summary["networks"][str(iteration)][
            "exact_loss_by_root_player"
        ]
        lines.append(
            f"iter {iteration}: Blue count/mean/MAE/Q>0="
            f"{split['blue']['count']}/{split['blue']['mean']:+.6f}/"
            f"{split['blue']['mae']:.6f}/"
            f"{_percent(split['blue']['positive_fraction'])}; Red="
            f"{split['red']['count']}/{split['red']['mean']:+.6f}/"
            f"{split['red']['mae']:.6f}/"
            f"{_percent(split['red']['positive_fraction'])}"
        )

    replay = summary["replay_validation"]
    lines.extend(
        [
            "",
            "Replay target audit",
            "-------------------",
            f"Exact duplicate states: {replay['duplicate_unique_state_count']} "
            f"unique keys ({replay['duplicate_extra_sample_count']} extra samples)",
            f"Contradictory-z duplicate states: "
            f"{replay['contradictory_z_unique_state_count']} unique keys / "
            f"{replay['contradictory_z_sample_count']} samples",
            "Roundtrip failures/player mismatches/invalid targets: "
            f"{replay['roundtrip_failures']}/{replay['player_mismatches']}/"
            f"{replay['invalid_targets']}",
            "",
            "Interpretation",
            "--------------",
            f"Primary classification: {summary['classification']['primary']}",
            f"Regression signals: "
            f"{summary['classification']['regression_signal_count']}/3",
            f"Color-asymmetry flag: "
            f"{summary['classification']['color_asymmetry_flag']}",
            "Rules-exact terminal actions were never passed through the network.",
            f"Network/checkpoint/replay unchanged: {summary['network_unchanged']}/"
            f"{summary['checkpoints_unchanged']}/{summary['replay_unchanged']}",
            f"Elapsed seconds: {summary['elapsed_seconds']:.3f}",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args):
    started = time.perf_counter()
    replay_path = args.run_dir / "replay_buffer.pt"
    checkpoint_paths = {
        10: args.run_dir / "checkpoints" / "iteration_000010.pt",
        50: args.run_dir / "checkpoints" / "iteration_000050.pt",
    }
    for path in (replay_path, *checkpoint_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"required audit input does not exist: {path}")
    replay_sha_before = file_sha256(replay_path)
    checkpoint_sha_before = {
        iteration: file_sha256(path)
        for iteration, path in checkpoint_paths.items()
    }

    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    for key in ("states", "values", "players"):
        if key not in replay:
            raise ValueError(f"replay payload is missing {key}")
    states = np.asarray(replay["states"], dtype=np.float32)
    values = np.asarray(replay["values"], dtype=np.float32)
    players = np.asarray(replay["players"], dtype=np.int8)

    validation_started = time.perf_counter()
    replay_validation = validate_replay(states, values, players)
    validation_seconds = time.perf_counter() - validation_started
    indices = select_audit_indices(
        len(states), maximum=MAX_AUDITED_STATES, seed=FIXED_SEED
    )
    subset_digest = hashlib.sha256(indices.tobytes()).hexdigest()
    checkpoints = [
        load_evaluation_checkpoint(
            checkpoint_paths[iteration],
            device=args.device,
            expected_iteration=iteration,
            state_encoder=encode_state,
        )
        for iteration in (10, 50)
    ]
    network_digests_before = {
        checkpoint.iteration: network_state_digest(checkpoint.network)
        for checkpoint in checkpoints
    }

    audit_started = time.perf_counter()

    def progress(completed):
        print(
            f"audited {completed:>5}/{len(indices)} states "
            f"({100.0 * completed / len(indices):5.1f}%)",
            flush=True,
        )

    audited = audit_networks_on_states(
        checkpoints,
        states,
        indices,
        chunk_size=128,
        progress_callback=progress,
    )
    audit_seconds = time.perf_counter() - audit_started
    network_digests_after = {
        checkpoint.iteration: network_state_digest(checkpoint.network)
        for checkpoint in checkpoints
    }
    checkpoint_sha_after = {
        iteration: file_sha256(path)
        for iteration, path in checkpoint_paths.items()
    }
    replay_sha_after = file_sha256(replay_path)

    network_unchanged = network_digests_before == network_digests_after
    checkpoints_unchanged = checkpoint_sha_before == checkpoint_sha_after
    replay_unchanged = replay_sha_before == replay_sha_after
    if not (network_unchanged and checkpoints_unchanged and replay_unchanged):
        raise RuntimeError("fixed network/checkpoint/replay changed during audit")

    iteration_10 = audited["networks"]["10"]
    iteration_50 = audited["networks"]["50"]
    summary = {
        "seed": FIXED_SEED,
        "maximum_audited_states": MAX_AUDITED_STATES,
        "total_replay_states": len(states),
        "audited_states": len(indices),
        "audited_indices_sha256": subset_digest,
        "same_state_action_subset_for_both_networks": True,
        "device": str(checkpoints[0].device),
        "replay_path": str(replay_path),
        "checkpoint_paths": {
            str(iteration): str(path)
            for iteration, path in checkpoint_paths.items()
        },
        "replay_sha256": replay_sha_before,
        "checkpoint_sha256": checkpoint_sha_before,
        "network_state_digests_before": network_digests_before,
        "network_state_digests_after": network_digests_after,
        "network_unchanged": network_unchanged,
        "checkpoints_unchanged": checkpoints_unchanged,
        "replay_unchanged": replay_unchanged,
        "replay_validation": replay_validation,
        **audited,
        "classification": classify_value_audit(iteration_10, iteration_50),
        "validation_seconds": validation_seconds,
        "oracle_and_inference_seconds": audit_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        args.report_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(args.report_dir / "summary.txt", render_summary(summary))
    print(json.dumps(summary["classification"], sort_keys=True), flush=True)
    print(
        f"report={args.report_dir} elapsed={summary['elapsed_seconds']:.2f}s",
        flush=True,
    )
    return summary


def main(argv=None):
    run_audit(parse_args(argv))


if __name__ == "__main__":
    main()
