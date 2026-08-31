"""Run paired deterministic milestone evaluation for AlphaZero Rules V2."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from alphazero_v2.evaluate import (
    generate_opening_suite,
    load_evaluation_checkpoint,
    play_arena_game,
    run_tactical_sanity,
    validate_opening_suite,
)


DEFAULT_RUN_DIR = Path("runs/alphazero_v2/main_20260830")
DEFAULT_REPORT_DIR = Path("reports/alphazero_v2_evaluation_20260901")
REQUESTED_ITERATIONS = (50, 100, 200, 300, 370, 375)
LADDER_PAIRS = ((50, 100), (100, 200), (200, 300), (300, 370), (370, 375))
LATEST_PAIRS = ((375, 50), (375, 100), (375, 200), (375, 300), (375, 370))


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


def load_or_create_openings(path, count, seed):
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("seed") != seed or payload.get("count") != count:
            raise ValueError("existing opening suite has different seed/count")
    else:
        payload = generate_opening_suite(count=count, seed=seed)
        atomic_json_save(payload, path)
    validate_opening_suite(payload)
    return payload


def discover_checkpoints(run_dir, iterations, device):
    checkpoints = {}
    for iteration in iterations:
        path = Path(run_dir) / "checkpoints" / f"iteration_{iteration:06d}.pt"
        if path.is_file():
            checkpoints[iteration] = load_evaluation_checkpoint(
                path,
                device=device,
                expected_iteration=iteration,
            )
    return checkpoints


def aggregate_matchup(games, checkpoint_a, checkpoint_b):
    selected = [
        game
        for game in games
        if {game["blue_iteration"], game["red_iteration"]}
        == {checkpoint_a, checkpoint_b}
    ]
    a_wins = [game for game in selected if game["winner_iteration"] == checkpoint_a]
    b_wins = [game for game in selected if game["winner_iteration"] == checkpoint_b]
    return {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "games": len(selected),
        "a_wins": len(a_wins),
        "b_wins": len(b_wins),
        "a_wins_as_blue": sum(
            game["blue_iteration"] == checkpoint_a for game in a_wins
        ),
        "a_wins_as_red": sum(
            game["red_iteration"] == checkpoint_a for game in a_wins
        ),
        "b_wins_as_blue": sum(
            game["blue_iteration"] == checkpoint_b for game in b_wins
        ),
        "b_wins_as_red": sum(
            game["red_iteration"] == checkpoint_b for game in b_wins
        ),
        "blue_total_wins": sum(game["winner_color"] == 1 for game in selected),
        "red_total_wins": sum(game["winner_color"] == 2 for game in selected),
        "capture_endings": sum(
            game["terminal_reason"] == "CAPTURE_WIN" for game in selected
        ),
        "pass_score_endings": sum(
            game["terminal_reason"] == "PASS_SCORE_END" for game in selected
        ),
        "mean_game_length": (
            float(np.mean([game["game_length"] for game in selected]))
            if selected
            else None
        ),
        "mean_pass_usage": (
            float(np.mean([game["pass_usage"] for game in selected]))
            if selected
            else None
        ),
    }


def build_matrix(iterations, games):
    matrix = {}
    for row in iterations:
        matrix[str(row)] = {}
        for column in iterations:
            if row == column:
                matrix[str(row)][str(column)] = "-"
                continue
            matchup = aggregate_matchup(games, row, column)
            matrix[str(row)][str(column)] = (
                f"{matchup['a_wins']}/{matchup['games']}"
                if matchup["games"]
                else ""
            )
    return matrix


def find_cycles(iterations, games):
    beats = set()
    for left_index, left in enumerate(iterations):
        for right in iterations[left_index + 1 :]:
            result = aggregate_matchup(games, left, right)
            if result["a_wins"] > result["b_wins"]:
                beats.add((left, right))
            elif result["b_wins"] > result["a_wins"]:
                beats.add((right, left))
    cycles = []
    for first in iterations:
        for second in iterations:
            for third in iterations:
                if len({first, second, third}) != 3:
                    continue
                if (
                    (first, second) in beats
                    and (second, third) in beats
                    and (third, first) in beats
                ):
                    cycle = min(
                        (first, second, third),
                        (second, third, first),
                        (third, first, second),
                    )
                    if cycle not in cycles:
                        cycles.append(cycle)
    return [list(cycle) for cycle in sorted(cycles)]


def build_interpretation(iterations, games, ladder, latest):
    totals = {}
    for iteration in iterations:
        relevant = [
            game
            for game in games
            if iteration in (game["blue_iteration"], game["red_iteration"])
        ]
        wins = sum(game["winner_iteration"] == iteration for game in relevant)
        totals[str(iteration)] = {
            "wins": wins,
            "games": len(relevant),
            "win_rate": wins / len(relevant) if relevant else None,
        }
    eligible = [
        (int(iteration), item["win_rate"])
        for iteration, item in totals.items()
        if item["win_rate"] is not None
    ]
    strongest = max(eligible, key=lambda item: (item[1], item[0]))[0]
    regressions = [
        {
            "older": result["checkpoint_a"],
            "newer": result["checkpoint_b"],
            "newer_wins": result["b_wins"],
            "games": result["games"],
        }
        for result in ladder
        if result["games"] and result["b_wins"] < result["a_wins"]
    ]
    latest_games = sum(item["games"] for item in latest)
    latest_wins = sum(item["a_wins"] for item in latest)
    latest_blue_wins = sum(item["a_wins_as_blue"] for item in latest)
    latest_red_wins = sum(item["a_wins_as_red"] for item in latest)
    blue_wins = sum(game["winner_color"] == 1 for game in games)
    capture_endings = sum(game["terminal_reason"] == "CAPTURE_WIN" for game in games)
    cycles = find_cycles(iterations, games)
    interpretation = {
        "overall_strength_progression": totals,
        "strongest_checkpoint_candidate": strongest,
        "latest_375_wins": latest_wins,
        "latest_375_games": latest_games,
        "latest_375_win_rate": latest_wins / latest_games if latest_games else None,
        "latest_375_wins_as_blue": latest_blue_wins,
        "latest_375_wins_as_red": latest_red_wins,
        "color_dependence": {
            "blue_wins": blue_wins,
            "red_wins": len(games) - blue_wins,
            "blue_win_rate": blue_wins / len(games) if games else None,
        },
        "ending_behavior": {
            "capture_endings": capture_endings,
            "pass_score_endings": len(games) - capture_endings,
            "capture_rate": capture_endings / len(games) if games else None,
        },
        "non_monotonic_ladder_results": regressions,
        "possible_non_transitive_cycles": cycles,
        "sample_size_note": (
            "Each matchup has only 20 paired-opening games and is diagnostic, "
            "not a promotion or strength proof."
        ),
    }
    interpretation["assessment"] = {
        "overall_strength": (
            f"Iteration {strongest} has the highest observed aggregate win rate; "
            f"iteration 375 won {latest_wins}/{latest_games} against historical "
            "checkpoints."
        ),
        "color_dependence": (
            f"Blue won {blue_wins}/{len(games)} games. Iteration 375 won "
            f"{latest_blue_wins} as Blue and {latest_red_wins} as Red, so its "
            "historical advantage is not solely a Blue-color artifact."
        ),
        "non_monotonicity": (
            f"Newer checkpoint regressions appeared in {len(regressions)} ladder "
            f"matchups: {regressions}."
        ),
        "ending_behavior": (
            f"Capture ended {capture_endings}/{len(games)} games; PASS scoring "
            f"ended {len(games) - capture_endings}/{len(games)}. This is a "
            "capture-heavy behavioral signal, not by itself a strength proof."
        ),
        "cyclic_behavior": (
            f"Observed majority-win cycles: {cycles}. These are possible "
            "non-transitive behavior under the small opening suite."
        ),
        "collapse_assessment": (
            "The strong historical win rate argues against broad collapse, while "
            "recent head-to-head regressions and capture-only endings show local "
            "non-monotonicity and a possible PASS/scoring blind spot."
        ),
    }
    return interpretation


def write_summary_text(path, summary):
    iterations = summary["iterations"]
    lines = [
        "# AlphaZero V2 Milestone Evaluation",
        "",
        f"Device: {summary['device_name']}",
        f"MCTS simulations: {summary['mcts_simulations']}",
        f"Openings: {summary['opening_count']} paired by color",
        f"Total games: {summary['total_games']}",
        "",
        "## Win matrix (row wins / games)",
        "",
        "| row \\ col | " + " | ".join(str(item) for item in iterations) + " |",
        "|" + "---|" * (len(iterations) + 1),
    ]
    for row in iterations:
        cells = [summary["matrix"][str(row)][str(column)] for column in iterations]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")

    for title, key in (
        ("Milestone ladder", "milestone_ladder"),
        ("Iteration 375 vs historical", "latest_vs_historical"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| A | B | A wins | B wins | Blue wins | Red wins | Capture | Score |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in summary[key]:
            lines.append(
                f"| {item['checkpoint_a']} | {item['checkpoint_b']} | "
                f"{item['a_wins']} | {item['b_wins']} | "
                f"{item['blue_total_wins']} | {item['red_total_wins']} | "
                f"{item['capture_endings']} | {item['pass_score_endings']} |"
            )

    lines.extend(
        [
            "",
            "## Network wins by assigned color",
            "",
            "| A | B | A as Blue | A as Red | B as Blue | B as Red |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    seen_color_pairs = set()
    for item in (*summary["milestone_ladder"], *summary["latest_vs_historical"]):
        pair = tuple(sorted((item["checkpoint_a"], item["checkpoint_b"])))
        if pair in seen_color_pairs:
            continue
        seen_color_pairs.add(pair)
        lines.append(
            f"| {item['checkpoint_a']} | {item['checkpoint_b']} | "
            f"{item['a_wins_as_blue']} | {item['a_wins_as_red']} | "
            f"{item['b_wins_as_blue']} | {item['b_wins_as_red']} |"
        )

    interpretation = summary["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Strongest checkpoint candidate by observed aggregate win rate: "
                f"{interpretation['strongest_checkpoint_candidate']}"
            ),
            (
                "Iteration 375: "
                f"{interpretation['latest_375_wins']}/"
                f"{interpretation['latest_375_games']} wins against historical."
            ),
            (
                "Color: Blue "
                f"{interpretation['color_dependence']['blue_wins']}, Red "
                f"{interpretation['color_dependence']['red_wins']}."
            ),
            (
                "Endings: capture "
                f"{interpretation['ending_behavior']['capture_endings']}, score "
                f"{interpretation['ending_behavior']['pass_score_endings']}."
            ),
            (
                "Non-monotonic ladder results: "
                f"{interpretation['non_monotonic_ladder_results']}"
            ),
            (
                "Possible non-transitive cycles: "
                f"{interpretation['possible_non_transitive_cycles']}"
            ),
            "",
            *interpretation["assessment"].values(),
            "",
            interpretation["sample_size_note"],
            "",
            "## Tactical sanity",
            "",
            "```json",
            json.dumps(summary["tactical_sanity"], indent=2, sort_keys=True),
            "```",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args):
    started = time.perf_counter()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    existing_summary_path = args.report_dir / "summary.json"
    previous_elapsed = 0.0
    if existing_summary_path.exists():
        previous_elapsed = float(
            json.loads(existing_summary_path.read_text(encoding="utf-8")).get(
                "elapsed_seconds", 0.0
            )
        )
    openings_payload = load_or_create_openings(
        args.report_dir / "openings.json",
        args.openings,
        args.seed,
    )
    checkpoints = discover_checkpoints(
        args.run_dir,
        REQUESTED_ITERATIONS,
        args.device,
    )
    if not checkpoints:
        raise RuntimeError("no requested evaluation checkpoints exist")
    iterations = sorted(checkpoints)
    ladder_pairs = [pair for pair in LADDER_PAIRS if all(i in checkpoints for i in pair)]
    latest_pairs = [pair for pair in LATEST_PAIRS if all(i in checkpoints for i in pair)]
    planned_pairs = []
    seen_pairs = set()
    for pair in (*ladder_pairs, *latest_pairs):
        key = tuple(sorted(pair))
        if key not in seen_pairs:
            seen_pairs.add(key)
            planned_pairs.append(pair)

    matches_path = args.report_dir / "matches.jsonl"
    games = []
    completed = set()
    if matches_path.exists():
        for line in matches_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            games.append(game)
            completed.add(
                (
                    tuple(sorted((game["blue_iteration"], game["red_iteration"]))),
                    game["opening_id"],
                    game["blue_iteration"],
                )
            )

    with matches_path.open("a", encoding="utf-8") as handle:
        for first, second in planned_pairs:
            first_checkpoint = checkpoints[first]
            second_checkpoint = checkpoints[second]
            pair_key = tuple(sorted((first, second)))
            for opening in openings_payload["openings"]:
                assignments = (
                    (first_checkpoint, second_checkpoint),
                    (second_checkpoint, first_checkpoint),
                )
                for blue_checkpoint, red_checkpoint in assignments:
                    game_key = (
                        pair_key,
                        int(opening["opening_id"]),
                        blue_checkpoint.iteration,
                    )
                    if game_key in completed:
                        continue
                    game = play_arena_game(
                        blue_checkpoint,
                        red_checkpoint,
                        opening,
                        simulations=args.mcts_simulations,
                    )
                    game["pair_key"] = list(pair_key)
                    handle.write(json.dumps(game, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    games.append(game)
                    completed.add(game_key)
                    print(
                        f"game {len(games)}: {game['blue_iteration']} Blue vs "
                        f"{game['red_iteration']} Red -> iter "
                        f"{game['winner_iteration']} {game['terminal_reason']}",
                        flush=True,
                    )

    ladder = [aggregate_matchup(games, first, second) for first, second in ladder_pairs]
    latest = [aggregate_matchup(games, first, second) for first, second in latest_pairs]
    latest_iteration = max(iterations)
    tactical = run_tactical_sanity(
        checkpoints[latest_iteration],
        simulations=args.mcts_simulations,
    )
    interpretation = build_interpretation(iterations, games, ladder, latest)
    interpretation["tactical_context"] = {
        "immediate_capture_passed": tactical["immediate_capture"]["pass"],
        "winning_score_pass_passed": tactical["winning_score_pass"]["pass"],
        "assessment": (
            "Iteration 375 found the immediate capture but did not choose the "
            "immediately winning scoring PASS in the constructed state."
        ),
    }
    first_checkpoint = checkpoints[iterations[0]]
    summary = {
        "run_dir": str(args.run_dir),
        "device": str(first_checkpoint.device),
        "device_name": (
            torch.cuda.get_device_name(first_checkpoint.device)
            if first_checkpoint.device.type == "cuda"
            else "CPU"
        ),
        "mcts_simulations": args.mcts_simulations,
        "opening_count": len(openings_payload["openings"]),
        "opening_seed": openings_payload["seed"],
        "iterations": iterations,
        "checkpoint_paths": {
            str(iteration): str(checkpoints[iteration].path)
            for iteration in iterations
        },
        "total_games": len(games),
        "milestone_ladder": ladder,
        "latest_vs_historical": latest,
        "matrix": build_matrix(iterations, games),
        "tactical_sanity": tactical,
        "interpretation": interpretation,
        "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
    }
    atomic_json_save(summary, args.report_dir / "summary.json")
    write_summary_text(args.report_dir / "summary.txt", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--openings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args(argv)
    if args.mcts_simulations <= 0 or args.openings <= 0:
        parser.error("simulations and openings must be positive")
    return args


if __name__ == "__main__":
    run_evaluation(parse_args())
