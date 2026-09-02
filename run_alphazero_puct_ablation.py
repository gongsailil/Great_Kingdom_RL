"""Run the fixed iteration-50 V3 c_puct tactical and small-arena ablation."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from alphazero_v2.evaluate import (
    load_evaluation_checkpoint,
    validate_opening_suite,
)
from alphazero_v3.encoder import encode_state
from alphazero_v3.puct_ablation import (
    DEFAULT_C_PUCT_VALUES,
    FIXED_SIMULATIONS,
    aggregate_puct_matchup,
    classify_puct_sweep,
    play_puct_arena_game,
    puct_agent_id,
    run_puct_tactical_sweep,
    select_arena_candidates,
)


DEFAULT_CHECKPOINT = Path(
    "runs/alphazero_v3/territory_pilot_20260901/"
    "checkpoints/iteration_000050.pt"
)
DEFAULT_OPENINGS = Path("reports/alphazero_v2_evaluation_20260901/openings.json")
DEFAULT_REPORT_DIR = Path("reports/alphazero_puct_ablation_20260902")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--c-puct", nargs="+", type=float, default=list(DEFAULT_C_PUCT_VALUES)
    )
    parser.add_argument("--simulations", type=int, default=FIXED_SIMULATIONS)
    parser.add_argument(
        "--tactical-only",
        action="store_true",
        help="write the tactical sweep without running paired-opening games",
    )
    args = parser.parse_args(argv)
    if tuple(args.c_puct) != DEFAULT_C_PUCT_VALUES:
        parser.error("fixed ablation requires --c-puct 1.5 3.0 6.0 12.0")
    if args.simulations != FIXED_SIMULATIONS:
        parser.error("fixed ablation requires --simulations 256")
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


def append_json_line(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_for(tactical, c_puct, fixture):
    return next(
        record
        for record in tactical["records"]
        if record["c_puct"] == float(c_puct) and record["fixture"] == fixture
    )


def _compact_tactical_table(tactical):
    table = []
    for c_puct in tactical["c_puct_values"]:
        item = {"c_puct": c_puct}
        for fixture in (
            "immediate_capture",
            "immediate_capture_threat_defense",
            "immediate_winning_pass",
        ):
            record = _record_for(tactical, c_puct, fixture)
            expected = record["expected_action_stats"]
            selected = record["selected_action_stats"]
            item[fixture] = {
                "selected_action": record["selected_action"],
                "success": record["success"],
                "expected_action": record["expected_action"],
                "expected_prior": expected["prior"],
                "expected_visits": expected["visit_count"],
                "expected_visit_fraction": expected["visit_fraction"],
                "expected_q": expected["q_value_root_player"],
                "selected_visits": selected["visit_count"],
                "selected_q": selected["q_value_root_player"],
                "legal_child_count": record["legal_child_count"],
                "visited_child_count": record["visited_child_count"],
                "visited_child_fraction": record["visited_child_fraction"],
                "visit_entropy": record["visit_entropy"],
            }
        table.append(item)
    return table


def render_summary(summary):
    lines = [
        "AlphaZero V3 c_puct ablation (2026-09-02)",
        "=" * 60,
        "",
        "No training or checkpoint mutation was performed.",
        f"Checkpoint: {summary['checkpoint_path']} (iteration 50)",
        f"Network parameters: {summary['network_parameters']}",
        f"Device: {summary['device']}",
        "Fixed search: learned policy + learned value, 256 simulations, "
        "temperature 0, root noise off",
        "c_puct values: 1.5, 3.0, 6.0, 12.0",
        "",
        "Tactical sweep",
        "--------------",
        "c_puct  Capture  Defense  Winning PASS  Capture action  Defense action  PASS action",
    ]
    for item in summary["tactical_table"]:
        capture = item["immediate_capture"]
        defense = item["immediate_capture_threat_defense"]
        winning_pass = item["immediate_winning_pass"]
        lines.append(
            f"{item['c_puct']:>6g}  {str(capture['success']):>7}  "
            f"{str(defense['success']):>7}  {str(winning_pass['success']):>12}  "
            f"{capture['selected_action']:>14}  "
            f"{defense['selected_action']:>14}  "
            f"{winning_pass['selected_action']:>11}"
        )
    lines.extend(["", "Search coverage", "---------------"])
    for item in summary["tactical_table"]:
        lines.append(f"c_puct {item['c_puct']:g}:")
        for fixture in (
            "immediate_capture",
            "immediate_capture_threat_defense",
            "immediate_winning_pass",
        ):
            data = item[fixture]
            lines.append(
                f"  {fixture}: expected visits={data['expected_visits']} "
                f"({data['expected_visit_fraction']:.6f}), "
                f"Q={data['expected_q']:+.6f}; visited="
                f"{data['visited_child_count']}/{data['legal_child_count']} "
                f"({data['visited_child_fraction']:.6f}), "
                f"visit entropy={data['visit_entropy']:.6f}"
            )
    lines.extend(["", "Small paired-opening arena", "--------------------------"])
    if summary["arena"]:
        for arena in summary["arena"]:
            lines.append(
                f"c_puct 1.5 vs {arena['candidate_c_puct']:g}: "
                f"{arena['baseline_wins']}-{arena['candidate_wins']}; "
                f"candidate Blue/Red wins "
                f"{arena['candidate_wins_as_blue']}/"
                f"{arena['candidate_wins_as_red']}; color Blue/Red "
                f"{arena['blue_total_wins']}/{arena['red_total_wins']}; "
                f"capture/score {arena['capture_endings']}/"
                f"{arena['pass_score_endings']}; PASS actions "
                f"{arena['pass_action_count']}; mean length "
                f"{arena['mean_game_length']:.3f}"
            )
    else:
        lines.append("No c_puct exceeded the baseline tactical result; arena skipped.")
    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            f"Classification: {summary['classification']}",
            "Each 20-game arena matchup is diagnostic, not statistical proof.",
            f"Elapsed seconds: {summary['elapsed_seconds']:.3f}",
            "",
        ]
    )
    return "\n".join(lines)


def run_ablation(args):
    started = time.perf_counter()
    checkpoint = load_evaluation_checkpoint(
        args.checkpoint,
        device=args.device,
        expected_iteration=50,
        state_encoder=encode_state,
    )
    tactical = run_puct_tactical_sweep(
        checkpoint,
        args.c_puct,
        args.simulations,
    )
    candidates = select_arena_candidates(tactical, maximum=2)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(tactical, args.report_dir / "tactical_sweep.json")
    print(
        json.dumps(
            {
                "tactical_table": _compact_tactical_table(tactical),
                "arena_candidates": candidates,
                "classification": classify_puct_sweep(tactical),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    games = []
    arena_path = args.report_dir / "arena_games.jsonl"
    completed = set()
    if arena_path.exists():
        for line in arena_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            games.append(game)
            completed.add(
                (
                    float(game["candidate_c_puct"]),
                    int(game["opening_id"]),
                    game["blue_agent"],
                )
            )
    if candidates and not args.tactical_only:
        openings_payload = json.loads(args.openings.read_text(encoding="utf-8"))
        validate_opening_suite(openings_payload)
        if (
            openings_payload.get("seed") != 20260901
            or openings_payload.get("count") != 10
        ):
            raise ValueError("arena requires the committed seed-20260901 openings")
        baseline = DEFAULT_C_PUCT_VALUES[0]
        expected_games = len(candidates) * 20
        for candidate in candidates:
            for opening in openings_payload["openings"]:
                for blue_c_puct, red_c_puct in (
                    (baseline, candidate),
                    (candidate, baseline),
                ):
                    blue_agent = puct_agent_id(checkpoint, blue_c_puct)
                    game_key = (float(candidate), opening["opening_id"], blue_agent)
                    if game_key in completed:
                        continue
                    game = play_puct_arena_game(
                        checkpoint,
                        blue_c_puct,
                        red_c_puct,
                        opening,
                        simulations=args.simulations,
                    )
                    game["baseline_c_puct"] = float(baseline)
                    game["candidate_c_puct"] = float(candidate)
                    append_json_line(game, arena_path)
                    games.append(game)
                    completed.add(game_key)
                    print(
                        f"arena {len(games)}/{expected_games}: "
                        f"{game['blue_agent']} Blue vs {game['red_agent']} Red "
                        f"-> {game['winner_agent']} {game['terminal_reason']}",
                        flush=True,
                    )

    arena = [
        aggregate_puct_matchup(games, checkpoint.iteration, 1.5, candidate)
        for candidate in candidates
    ]
    summary = {
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_iteration": checkpoint.iteration,
        "network_parameters": checkpoint.network.parameter_count(),
        "device": str(checkpoint.device),
        "training_performed": False,
        "fixed": {
            "mcts_simulations": args.simulations,
            "learned_policy": True,
            "learned_value": True,
            "temperature": 0.0,
            "root_noise": False,
            "rules": "Great Kingdom Rules V2",
            "encoder": "V3 territory, 9 planes",
        },
        "c_puct_values": list(args.c_puct),
        "safe_defense_actions": tactical["safe_defense_actions"],
        "tactical_table": _compact_tactical_table(tactical),
        "arena_candidates": candidates,
        "arena_games": len(games),
        "arena": arena,
        "opening_suite": str(args.openings),
        "classification": classify_puct_sweep(tactical),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json_save(summary, args.report_dir / "summary.json")
    atomic_text_save(render_summary(summary), args.report_dir / "summary.txt")
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run_ablation(args)
    print(json.dumps(summary["arena"], indent=2, sort_keys=True), flush=True)
    print(f"Classification: {summary['classification']}", flush=True)
    print(f"Report: {args.report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
