"""Run the fixed post-training V5 automatic strategy evaluation."""

import argparse
import json
from pathlib import Path

from alphazero_v2.training_runner import _atomic_json_save
from alphazero_v4.evaluation import load_v4_evaluation_checkpoint
from alphazero_v5.evaluation import (
    evaluate_human_challenge,
    extract_human_challenge_states,
    historical_agents,
    load_v5_checkpoint,
    run_final_arenas,
    run_strategic_suite,
)


DEFAULT_RUN = Path("runs/alphazero_v5/strategy_20260908")
DEFAULT_REPORT = Path("reports/alphazero_v5_strategy")
DEFAULT_OPENINGS = Path("reports/alphazero_v4_acceptance_20260906/openings.json")


def _load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _summary_text(summary):
    lines = [
        "AlphaZero V5 strategy integrated run",
        "====================================",
        f"Cycles: {summary['training']['cycles']}",
        f"BEST version/promotions: {summary['training']['best_version']} / {summary['training']['promotions']}",
        f"Generated games/samples: {summary['training']['games']} / {summary['training']['samples']}",
        f"Replay size: {summary['training']['replay_size']}",
        "",
        "Final paired arenas (V5 wins / games):",
    ]
    for opponent, row in summary["final_arenas"].items():
        ci = row["paired_bootstrap"]["win_rate_ci95"]
        lines.append(f"- {opponent}: {row['wins']}/{row['games']} ({row['win_rate']:.1%}), paired 95% CI [{ci[0]:.1%}, {ci[1]:.1%}]")
    lines.extend((
        "",
        f"Fixed tactical suite: {'PASS' if summary['strategic_exact_pass'] else 'FAIL'}",
        f"Human-log challenge states: {summary['human_challenge_states']}",
        "Human game logs were evaluation-only and were never inserted into replay.",
    ))
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--v4", type=Path, default=Path("runs/alphazero_v4/stability_20260903/latest.pt"))
    parser.add_argument("--v3", type=Path, default=Path("runs/alphazero_v3/territory_pilot_20260901/checkpoints/iteration_000050.pt"))
    parser.add_argument("--v2", type=Path, default=Path("runs/alphazero_v2/main_20260830/latest.pt"))
    parser.add_argument("--human-logs", type=Path, default=Path("human_games/v4_20260906"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    v5 = load_v5_checkpoint(args.run_dir / "latest.pt", args.device)
    openings_payload = json.loads(args.openings.read_text())
    openings = openings_payload["openings"]
    opponents = historical_agents(v5.device, args.v4, args.v3, args.v2)
    arenas, games = run_final_arenas(v5, opponents, openings, v5.config.concurrent_games)
    strategic = run_strategic_suite(v5)
    v4 = load_v4_evaluation_checkpoint(args.v4, v5.device, expected_iteration=50)
    challenge_states = extract_human_challenge_states(args.human_logs, 20)
    challenge = evaluate_human_challenge(v5, v4, challenge_states)
    run_summary = json.loads((args.run_dir / "summary.json").read_text())
    cycles = _load_jsonl(args.run_dir / "metrics.jsonl")
    promotions = _load_jsonl(args.run_dir / "promotion_history.jsonl")
    summary = {
        "architecture": "12-plane 128x6 policy/value/liberty/score network",
        "parameter_count": v5.network.parameter_count(),
        "checkpoint": {"cycle": v5.cycle, "best_version": v5.best_version,
                       "calibration_temperature": v5.calibration_temperature},
        "training": {"cycles": run_summary["cycle"], "best_version": run_summary["best_version"],
                     "promotions": run_summary["promotion_count"],
                     "games": run_summary["total_self_play_games"],
                     "samples": run_summary["total_samples_generated"],
                     "replay_size": run_summary["replay"]["current_size"],
                     "elapsed_seconds": run_summary["elapsed_seconds"]},
        "final_arenas": arenas,
        "strategic_exact_pass": all(row["success"] for row in strategic if row["exact_assertion"]),
        "human_challenge_states": len(challenge),
        "human_logs_used_for_training": False,
    }
    _write_jsonl(args.report_dir / "cycles.jsonl", cycles)
    _atomic_json_save(promotions, args.report_dir / "promotion_history.json")
    _atomic_json_save(arenas, args.report_dir / "final_arena.json")
    _atomic_json_save([row["symmetry"] for row in cycles], args.report_dir / "symmetry_metrics.json")
    _atomic_json_save(challenge, args.report_dir / "human_challenge_suite.json")
    _atomic_json_save(strategic, args.report_dir / "strategic_suite.json")
    _atomic_json_save(summary, args.report_dir / "summary.json")
    (args.report_dir / "summary.txt").write_text(_summary_text(summary), encoding="utf-8")
    _write_jsonl(args.report_dir / "arena_games.jsonl", games)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
