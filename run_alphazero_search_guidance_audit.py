"""Audit learned policy and value guidance without changing or training V3."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from alphazero_v2.evaluate import load_evaluation_checkpoint
from alphazero_v3.encoder import encode_state
from alphazero_v3.search_guidance_audit import (
    SEARCH_GUIDANCE_MODES,
    audit_checkpoint,
    classify_mode_outcomes,
)


DEFAULT_RUN_DIR = Path("runs/alphazero_v3/territory_pilot_20260901")
DEFAULT_REPORT_DIR = Path("reports/alphazero_search_guidance_audit_20260902")
DEFAULT_ITERATIONS = (0, 10, 50)
DEFAULT_SIMULATIONS = 256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=list(DEFAULT_ITERATIONS)
    )
    parser.add_argument(
        "--simulations", type=int, default=DEFAULT_SIMULATIONS
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.simulations <= 0:
        parser.error("--simulations must be positive")
    if any(iteration < 0 for iteration in args.iterations):
        parser.error("--iterations must be non-negative")
    return args


def atomic_json_save(payload, path):
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


def atomic_text_save(text, path):
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def checkpoint_path(run_dir, iteration):
    return Path(run_dir) / "checkpoints" / f"iteration_{iteration:06d}.pt"


def _compact_row(record):
    expected = record["expected_action_stats"]
    policy = record["policy_diagnostics"]
    return {
        "checkpoint_iteration": record["checkpoint_iteration"],
        "fixture": record["fixture"],
        "mode": record["mode"],
        "selected_action": record["selected_action"],
        "expected_action": record["expected_action"],
        "safe_actions": record["safe_actions"],
        "success": record["success"],
        "selected_visit_count": record["selected_action_stats"]["visit_count"],
        "expected_visit_count": expected["visit_count"],
        "expected_visit_fraction": expected["visit_fraction"],
        "expected_search_prior": expected["prior"],
        "expected_q_root_player": expected["q_value_root_player"],
        "expected_network_prior": policy["expected_action_legal_probability"],
        "expected_network_rank": policy["expected_action_legal_rank"],
        "root_network_value": record["root_network_value"],
        "expected_child_value": record["expected_child_value"],
    }


def build_summary(
    records,
    requested,
    available,
    missing,
    simulations,
    elapsed_seconds,
):
    fixtures = sorted({record["fixture"] for record in records})
    classifications = {}
    for iteration in available:
        checkpoint_records = [
            record
            for record in records
            if record["checkpoint_iteration"] == iteration
        ]
        classifications[str(iteration)] = {
            fixture: classify_mode_outcomes(checkpoint_records, fixture)
            for fixture in fixtures
        }
    policy_value_progression = {}
    for fixture in fixtures:
        policy_value_progression[fixture] = []
        for iteration in available:
            record = next(
                item
                for item in records
                if item["checkpoint_iteration"] == iteration
                and item["fixture"] == fixture
                and item["mode"] == "learned_policy_learned_value"
            )
            policy = record["policy_diagnostics"]
            policy_value_progression[fixture].append(
                {
                    "checkpoint_iteration": iteration,
                    "expected_action_raw_logit": policy[
                        "expected_action_raw_logit"
                    ],
                    "expected_action_legal_probability": policy[
                        "expected_action_legal_probability"
                    ],
                    "expected_action_legal_rank": policy[
                        "expected_action_legal_rank"
                    ],
                    "legal_policy_entropy": policy["legal_policy_entropy"],
                    "legal_action_count": policy["legal_action_count"],
                    "root_network_value": record["root_network_value"],
                    "expected_child_value": record["expected_child_value"],
                }
            )
    return {
        "fixed": {
            "rules": "Great Kingdom Rules V2",
            "encoder": "V3 territory, 9 planes",
            "mcts_simulations": int(simulations),
            "network_weights_unchanged": True,
            "training_performed": False,
            "terminal_value_source": "GreatKingdomLogicV2 winner",
        },
        "requested_checkpoint_iterations": list(requested),
        "available_checkpoint_iterations": list(available),
        "missing_checkpoint_iterations": list(missing),
        "missing_checkpoint_policy": "not replaced with a random network",
        "modes": [mode.name for mode in SEARCH_GUIDANCE_MODES],
        "classifications": classifications,
        "four_mode_table": [_compact_row(record) for record in records],
        "policy_value_progression": policy_value_progression,
        "elapsed_seconds": float(elapsed_seconds),
    }


def render_summary(summary):
    lines = [
        "AlphaZero V3 search-guidance audit (2026-09-02)",
        "=" * 60,
        "",
        "No training or checkpoint mutation was performed.",
        f"Rules: {summary['fixed']['rules']}",
        f"Encoder: {summary['fixed']['encoder']}",
        f"MCTS simulations: {summary['fixed']['mcts_simulations']}",
        "Available checkpoints: "
        + ", ".join(map(str, summary["available_checkpoint_iterations"])),
        "Missing checkpoints: "
        + (", ".join(map(str, summary["missing_checkpoint_iterations"])) or "none"),
        "Missing checkpoints were not replaced with newly initialized networks.",
        "",
        "Four-mode results",
        "-----------------",
        "Checkpoint  Fixture                            Mode                                  Selected  Success  Prior      Rank  Visits  Q(root)",
    ]
    for row in summary["four_mode_table"]:
        lines.append(
            f"{row['checkpoint_iteration']:>10}  "
            f"{row['fixture']:<33}  "
            f"{row['mode']:<36}  "
            f"{row['selected_action']:>8}  "
            f"{str(row['success']):>7}  "
            f"{row['expected_search_prior']:<9.6f}  "
            f"{row['expected_network_rank']:>4}  "
            f"{row['expected_visit_count']:>6}  "
            f"{row['expected_q_root_player']:+.6f}"
        )
    lines.extend(["", "Classifications", "---------------"])
    for iteration, classifications in summary["classifications"].items():
        lines.append(f"Iteration {iteration}:")
        for fixture, classification in classifications.items():
            lines.append(f"  {fixture}: {classification}")
    lines.extend(["", "Learned policy/value progression", "--------------------------------"])
    for fixture, progression in summary["policy_value_progression"].items():
        lines.append(f"{fixture}:")
        for item in progression:
            child = item["expected_child_value"]
            child_value = (
                child["exact_terminal_value_root_player"]
                if child["terminal"]
                else child["network_value_root_player"]
            )
            lines.append(
                "  iter {iteration}: prior={prior:.8f}, rank={rank}, "
                "logit={logit:+.6f}, entropy={entropy:.6f}, "
                "root_value={root:+.6f}, expected_child_root={child:+.6f}".format(
                    iteration=item["checkpoint_iteration"],
                    prior=item["expected_action_legal_probability"],
                    rank=item["expected_action_legal_rank"],
                    logit=item["expected_action_raw_logit"],
                    entropy=item["legal_policy_entropy"],
                    root=item["root_network_value"],
                    child=child_value,
                )
            )
    lines.extend(
        [
            "",
            f"Elapsed: {summary['elapsed_seconds']:.3f} seconds",
            "Full root top-10 diagnostics are in root_diagnostics.json.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(run_dir, report_dir, iterations, simulations, device):
    started = time.perf_counter()
    available = []
    missing = []
    all_records = []
    fixture_metadata = {}
    for iteration in iterations:
        path = checkpoint_path(run_dir, iteration)
        if not path.is_file():
            missing.append(int(iteration))
            continue
        checkpoint = load_evaluation_checkpoint(
            path,
            device=device,
            expected_iteration=iteration,
            state_encoder=encode_state,
        )
        available.append(int(iteration))
        result = audit_checkpoint(checkpoint, simulations=simulations)
        fixture_metadata[str(iteration)] = result["fixtures"]
        all_records.extend(result["records"])
    if not available:
        raise FileNotFoundError("none of the requested checkpoints exist")
    elapsed = time.perf_counter() - started
    summary = build_summary(
        all_records,
        iterations,
        available,
        missing,
        simulations,
        elapsed,
    )
    root_diagnostics = {
        "available_checkpoint_iterations": available,
        "missing_checkpoint_iterations": missing,
        "fixture_metadata": fixture_metadata,
        "records": all_records,
    }
    report_dir = Path(report_dir)
    atomic_json_save(root_diagnostics, report_dir / "root_diagnostics.json")
    atomic_json_save(summary, report_dir / "summary.json")
    atomic_text_save(render_summary(summary), report_dir / "summary.txt")
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run_audit(
        args.run_dir,
        args.report_dir,
        args.iterations,
        args.simulations,
        args.device,
    )
    print(json.dumps(summary["classifications"], indent=2, sort_keys=True))
    print(f"Report: {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
