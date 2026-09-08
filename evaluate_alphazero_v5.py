"""Run the fixed post-training V5 automatic strategy evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
        f"Automatic classification: {summary['automatic_classification']}",
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
        f"Training endings: capture {summary['behavior']['capture_endings']}, PASS-score {summary['behavior']['pass_score_endings']}",
        f"Training territory-state fraction: {summary['behavior']['territory_state_fraction']:.1%}",
        f"Training forced-loss entry rate: {summary['behavior']['forced_loss_entry_rate']:.1%}",
        "  (includes the losing side's inevitable final forced-loss root before capture)",
        f"Final calibration T: {summary['calibration']['final_temperature']:.4f}",
        f"Symmetry mean policy L1 first/final: {summary['symmetry']['first']['mean_policy_l1']:.4f} / {summary['symmetry']['final']['mean_policy_l1']:.4f}",
        f"Symmetry mean value |delta| first/final: {summary['symmetry']['first']['mean_value_abs_difference']:.4f} / {summary['symmetry']['final']['mean_value_abs_difference']:.4f}",
        "Human game logs were evaluation-only and were never inserted into replay.",
    ))
    return "\n".join(lines) + "\n"


def summarize_cycles(cycles):
    behavior_rows = [row["behavior"] for row in cycles]
    games = sum(row["new_games"] for row in behavior_rows)
    samples = sum(row["new_samples"] for row in behavior_rows)
    def weighted(name):
        return sum(row[name] * row["new_samples"] for row in behavior_rows) / max(1, samples)
    blocks = []
    maximum_cycle = max(row["cycle"] for row in cycles)
    for first in range(1, maximum_cycle + 1, 10):
        selected = [row for row in cycles if first <= row["cycle"] <= min(first + 9, maximum_cycle)]
        if not selected:
            continue
        block_games = sum(row["behavior"]["new_games"] for row in selected)
        blocks.append({
            "cycles": [first, selected[-1]["cycle"]], "games": block_games,
            "promotions": sum(row["promoted"] for row in selected),
            "capture_endings": sum(row["behavior"]["capture_endings"] for row in selected),
            "pass_score_endings": sum(row["behavior"]["pass_score_endings"] for row in selected),
            "territory_state_fraction": sum(row["behavior"]["territory_state_fraction"] * row["behavior"]["new_samples"] for row in selected) / sum(row["behavior"]["new_samples"] for row in selected),
            "forced_loss_entry_rate": sum(row["behavior"]["games_entering_forced_loss"] for row in selected) / block_games,
            "early_edge_fraction": sum(row["behavior"]["early_edge_placements"] for row in selected) / max(1, sum(row["behavior"]["early_placements"] for row in selected)),
            "value_abs_ge_0_9_fraction": sum(row["behavior"]["network_value_abs_ge_0_9"] for row in selected) / max(1, sum(row["behavior"]["network_value_count"] for row in selected)),
        })
    return {
        "games": games, "samples": samples,
        "capture_endings": sum(row["capture_endings"] for row in behavior_rows),
        "pass_score_endings": sum(row["pass_score_endings"] for row in behavior_rows),
        "pass_actions": sum(row["pass_usage"] for row in behavior_rows),
        "initial_start_games": sum(row["initial_start_games"] for row in behavior_rows),
        "territory_start_games": sum(row["territory_start_games"] for row in behavior_rows),
        "territory_state_fraction": weighted("territory_state_fraction"),
        "games_entering_forced_loss": sum(row["games_entering_forced_loss"] for row in behavior_rows),
        "forced_loss_entry_rate": sum(row["games_entering_forced_loss"] for row in behavior_rows) / games,
        "early_edge_fraction": sum(row["early_edge_placements"] for row in behavior_rows) / max(1, sum(row["early_placements"] for row in behavior_rows)),
        "own_adjacent_fraction": sum(row["own_adjacent_placements"] for row in behavior_rows) / max(1, sum(row["placements"] for row in behavior_rows)),
        "territory_creation_delta": sum(row["territory_creation_delta"] for row in behavior_rows),
        "territory_disruption_delta": sum(row["territory_disruption_delta"] for row in behavior_rows),
        "immediate_win_taken": sum(row["immediate_win_taken"] for row in behavior_rows),
        "immediate_win_opportunities": sum(row["immediate_win_opportunities"] for row in behavior_rows),
        "unsafe_actions_filtered": sum(row["unsafe_actions_filtered"] for row in behavior_rows),
        "blocks": blocks,
    }


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
    behavior = summarize_cycles(cycles)
    final_cycle = cycles[-1]
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
        "behavior": behavior,
        "final_replay_duplicates": final_cycle["duplicates"],
        "final_losses": final_cycle["losses"],
        "calibration": {
            "final_temperature": v5.calibration_temperature,
            "last_candidate": final_cycle["candidate_calibration"],
            "mean_before_nll": sum(row["candidate_calibration"]["before_nll"] for row in cycles) / len(cycles),
            "mean_after_nll": sum(row["candidate_calibration"]["after_nll"] for row in cycles) / len(cycles),
            "mean_before_brier": sum(row["candidate_calibration"]["before_brier"] for row in cycles) / len(cycles),
            "mean_after_brier": sum(row["candidate_calibration"]["after_brier"] for row in cycles) / len(cycles),
        },
        "symmetry": {"first": cycles[0]["symmetry"], "final": final_cycle["symmetry"]},
        "final_arenas": arenas,
        "strategic_exact_pass": all(row["success"] for row in strategic if row["exact_assertion"]),
        "human_challenge_states": len(challenge),
        "human_challenge": {
            "states": len(challenge),
            "v5_action_differs_from_logged_v4": sum(row["v5"]["action"] != row["v4_action"] for row in challenge),
            "v5_tactical_modes": {mode: sum(row["v5"]["tactical_mode"] == mode for row in challenge)
                                    for mode in ("NORMAL", "IMMEDIATE_WIN", "SAFE_DEFENSE", "FORCED_LOSS")},
            "note": "Evaluation-only representative V4 human-game states; no human label is treated as correct.",
        },
        "human_logs_used_for_training": False,
    }
    summary["automatic_classification"] = (
        "AUTO_PASS_PENDING_HUMAN"
        if summary["strategic_exact_pass"]
        and all(row["illegal_violations"] == 0 and row["tactical_failures"] == 0
                and row["win_rate"] > 0.5 for row in arenas.values())
        else "AUTO_FAIL_OR_INCONCLUSIVE"
    )
    replay_payload = torch.load(
        args.run_dir / "replay_buffer.pt", map_location="cpu", weights_only=False)
    replay_values = np.asarray(replay_payload["values"], dtype=np.float32)
    summary["final_replay_value_targets"] = {
        "samples": len(replay_values), "mean_z": float(replay_values.mean()),
        "std_z": float(replay_values.std()),
        "win_target_fraction": float(np.mean(replay_values == 1)),
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
