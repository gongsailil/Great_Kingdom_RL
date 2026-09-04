"""Run the fixed post-training V4 stability evaluation and curated report."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from alphazero_v4.diagnostics import (
    run_fixed_tactical_diagnostics,
    run_value_oracle_monitor,
)
from alphazero_v4.evaluation import (
    aggregate_arena,
    classify_result,
    load_v4_evaluation_checkpoint,
    run_cross_version_arena,
    training_trends,
)


DEFAULT_RUN_DIR = Path("runs/alphazero_v4/stability_20260903")
DEFAULT_V3_CHECKPOINT = Path(
    "runs/alphazero_v3/territory_pilot_20260901/"
    "checkpoints/iteration_000050.pt"
)
DEFAULT_REFERENCE_REPLAY = Path(
    "runs/alphazero_v3/territory_pilot_20260901/replay_buffer.pt"
)
DEFAULT_OPENINGS = Path("reports/alphazero_v2_evaluation_20260901/openings.json")
DEFAULT_REPORT_DIR = Path("reports/alphazero_v4_stability_20260903")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--v3-checkpoint", type=Path, default=DEFAULT_V3_CHECKPOINT)
    parser.add_argument("--reference-replay", type=Path, default=DEFAULT_REFERENCE_REPLAY)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def render_summary(summary):
    lines = [
        "AlphaZero V4 integrated stability run (2026-09-03)",
        "=" * 60,
        "",
        f"Result: {summary['classification']}",
        f"Training elapsed: {summary['training_elapsed_seconds']:.3f} seconds",
        f"Post-evaluation elapsed: {summary['post_evaluation_seconds']:.3f} seconds",
        f"Games/samples: {summary['training_totals']['games']}/"
        f"{summary['training_totals']['samples']}",
        "",
        "Ten-iteration ending blocks",
        "---------------------------",
    ]
    for block in summary["ending_trend"]:
        lines.append(
            f"{block['iterations'][0]:02d}-{block['iterations'][1]:02d}: "
            f"capture/score={block['capture_endings']}/"
            f"{block['pass_score_endings']}, PASS actions="
            f"{block['pass_action_count']}, mean length="
            f"{block['mean_game_length']:.3f}"
        )
    lines.extend(["", "Value-oracle monitors", "---------------------"])
    for monitor in summary["value_oracle_monitors"]:
        exact = monitor["exact_loss"]
        defense = monitor["defense"]
        alternatives = monitor["immediate_win_alternatives"]
        lines.append(
            f"iter {monitor['iteration']}: Q mean/median="
            f"{exact['mean']:+.6f}/{exact['median']:+.6f}, Q>0="
            f"{100*exact['positive_fraction']:.2f}%, Q>=0.5="
            f"{100*exact['at_least_0_5_fraction']:.2f}%, MAE="
            f"{exact['mae']:.6f}, defense failure="
            f"{100*defense['ranking_failure_fraction']:.2f}%, alt Q>=.99="
            f"{100*alternatives['at_least_0_99_fraction']:.2f}%"
        )
    final = summary["final_value_oracle"]
    lines.extend(
        [
            "",
            "Full 10,000-state final oracle",
            "------------------------------",
            f"Exact -1 actions: {final['exact_loss']['count']}",
            f"Q mean/median: {final['exact_loss']['mean']:+.6f}/"
            f"{final['exact_loss']['median']:+.6f}",
            f"Q>0 / Q>=0.5: {100*final['exact_loss']['positive_fraction']:.2f}% / "
            f"{100*final['exact_loss']['at_least_0_5_fraction']:.2f}%",
            f"MAE: {final['exact_loss']['mae']:.6f}",
            f"Defense ranking failure: "
            f"{100*final['defense']['ranking_failure_fraction']:.2f}%",
            f"Immediate-win alternative Q>=.9/.99: "
            f"{100*final['immediate_win_alternatives']['at_least_0_9_fraction']:.2f}%/"
            f"{100*final['immediate_win_alternatives']['at_least_0_99_fraction']:.2f}%",
            "",
            "V4 vs V3 iteration-50 arena",
            "----------------------------",
            f"V4/V3 wins: {summary['arena']['v4_wins']}/"
            f"{summary['arena']['v3_wins']}",
            f"V4 wins as Blue/Red: {summary['arena']['v4_wins_as_blue']}/"
            f"{summary['arena']['v4_wins_as_red']}",
            f"Color Blue/Red wins: {summary['arena']['blue_wins']}/"
            f"{summary['arena']['red_wins']}",
            f"Capture/score endings: {summary['arena']['capture_endings']}/"
            f"{summary['arena']['pass_score_endings']}",
            "Twenty games are diagnostic, not statistical proof.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(args):
    started = time.perf_counter()
    v4 = load_v4_evaluation_checkpoint(
        args.run_dir / "latest.pt", device=args.device, expected_iteration=50
    )
    state = type("DiagnosticState", (), {
        "network": v4.network,
        "iteration": v4.iteration,
    })()
    tactical = run_fixed_tactical_diagnostics(
        state, v4.config, v4.device
    )

    def progress(completed):
        print(f"final oracle {completed}/10000", flush=True)

    oracle = run_value_oracle_monitor(
        state,
        v4.config,
        v4.device,
        args.reference_replay,
        maximum=10_000,
        progress_callback=progress,
    )
    games = run_cross_version_arena(
        v4, args.v3_checkpoint, args.openings
    )
    metrics = [
        json.loads(line)
        for line in (args.run_dir / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    monitors = [
        json.loads(line)
        for line in (args.run_dir / "value_oracle.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    total_games = sum(metric["new_games"] for metric in metrics)
    total_samples = sum(metric["new_samples"] for metric in metrics)
    summary = {
        "architecture": {
            "encoder_shape": [9, 9, 9],
            "channels": 64,
            "residual_blocks": 3,
            "policy_actions": 82,
            "value_head": "raw logit with BCEWithLogitsLoss",
            "network_parameters": v4.network.parameter_count(),
        },
        "config": v4.config.to_dict(),
        "training_elapsed_seconds": float(metrics[-1]["elapsed_seconds"]),
        "post_evaluation_seconds": time.perf_counter() - started,
        "training_totals": {
            "games": total_games,
            "samples": total_samples,
            "capture_endings": sum(m["capture_endings"] for m in metrics),
            "pass_score_endings": sum(m["pass_score_endings"] for m in metrics),
            "pass_action_count": sum(m["pass_action_count"] for m in metrics),
            "illegal_violations": sum(
                sum(g["illegal_probability_violations"] for g in m["games"])
                for m in metrics
            ),
            "immediate_win_opportunities": sum(
                m["immediate_win_opportunities"] for m in metrics
            ),
            "immediate_win_taken": sum(m["immediate_win_taken"] for m in metrics),
            "defense_threat_states": sum(m["defense_threat_states"] for m in metrics),
            "defense_states_with_safe_action": sum(
                m["defense_states_with_safe_action"] for m in metrics
            ),
            "unsafe_actions_filtered": sum(m["unsafe_actions_filtered"] for m in metrics),
            "forced_loss_states": sum(m["forced_loss_states"] for m in metrics),
        },
        "ending_trend": training_trends(metrics),
        "final_duplicate_metrics": {
            key: metrics[-1][key]
            for key in (
                "raw_replay_size",
                "unique_state_groups",
                "duplicate_group_count",
                "contradictory_z_group_count",
                "samples_in_contradictory_groups",
                "mixed_outcome_group_fraction",
                "mean_group_win_probability_entropy",
            )
        },
        "loss_trend": [
            {
                key: metric[key]
                for key in (
                    "iteration", "policy_loss", "value_bce_loss", "brier_score", "total_loss"
                )
            }
            for metric in metrics
        ],
        "tactical_diagnostics": [
            json.loads(line)
            for line in (args.run_dir / "diagnostics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ],
        "final_tactical": tactical,
        "value_oracle_monitors": monitors,
        "final_value_oracle": oracle,
        "v3_baseline": {
            "exact_loss_q_positive": 0.26697744110546906,
            "exact_loss_q_at_least_0_5": 0.2408390909847665,
            "defense_ranking_failure": 0.914074074074074,
            "alternative_q_at_least_0_99": 0.4737191021877072,
        },
        "arena": aggregate_arena(games),
    }
    summary["classification"] = classify_result(summary)
    summary["post_evaluation_seconds"] = time.perf_counter() - started
    args.report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        args.report_dir / "arena_games.jsonl",
        "".join(json.dumps(game, sort_keys=True) + "\n" for game in games),
    )
    _atomic_write(
        args.report_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(args.report_dir / "summary.txt", render_summary(summary))
    print(f"result={summary['classification']} report={args.report_dir}", flush=True)
    return summary


def main(argv=None):
    run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
