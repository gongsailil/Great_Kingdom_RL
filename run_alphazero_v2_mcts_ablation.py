"""Evaluate iteration 375 with only the MCTS simulation budget varied."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from alphazero_v2.evaluate import (
    analyze_safe_defense_actions,
    analyze_search_root,
    capture_tactical_position,
    defense_tactical_position,
    load_evaluation_checkpoint,
    play_search_budget_game,
    suicide_tactical_position,
    territory_tactical_position,
    validate_opening_suite,
    winning_pass_tactical_position,
)
from great_kingdom_v2 import BLUE, PASS_ACTION, RED, MoveResultV2


DEFAULT_CHECKPOINT = Path("runs/alphazero_v2/main_20260830/latest.pt")
DEFAULT_OPENINGS = Path("reports/alphazero_v2_evaluation_20260901/openings.json")
DEFAULT_REPORT_DIR = Path("reports/alphazero_v2_mcts_ablation_20260901")
DEFAULT_BUDGETS = (64, 128, 256, 512)


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


def action_stats(diagnostic, action):
    for item in diagnostic["children"]:
        if item["action"] == action:
            return item
    return None


def compact_diagnostic(diagnostic):
    return {
        "simulations": diagnostic["simulations"],
        "root_player": diagnostic["root_player"],
        "selected_action": diagnostic["selected_action"],
        "total_child_visits": diagnostic["total_child_visits"],
        "pass_action": diagnostic["pass_action"],
        "top_actions": diagnostic["top_actions"],
    }


def run_tactical_budget_sweep(checkpoint, budgets):
    defense_logic = defense_tactical_position()
    defense_analysis = analyze_safe_defense_actions(defense_logic)
    expected_capture_action = 1 + 2 * 9
    expected_defense_action = 1 + 2 * 9
    sweep = []
    for budget in budgets:
        capture_logic = capture_tactical_position()
        capture_diagnostic = analyze_search_root(
            checkpoint, capture_logic, budget, top_k=10
        )
        capture_selected = capture_diagnostic["selected_action"]

        defense_logic = defense_tactical_position()
        defense_diagnostic = analyze_search_root(
            checkpoint, defense_logic, budget, top_k=10
        )
        defense_selected = defense_diagnostic["selected_action"]
        safe_actions = defense_analysis["safe_defense_actions"]
        safe_records = [
            action_stats(defense_diagnostic, action)
            for action in safe_actions
            if action_stats(defense_diagnostic, action) is not None
        ]
        best_safe = (
            min(
                safe_records,
                key=lambda item: (-item["visit_count"], item["action"]),
            )
            if safe_records
            else None
        )

        winning_pass = winning_pass_tactical_position()
        original_player = winning_pass.turn
        terminal_copy = winning_pass.copy()
        terminal_result = terminal_copy.apply_action(PASS_ACTION)
        if terminal_result != MoveResultV2.PASS_SCORE_END:
            raise RuntimeError("winning PASS fixture is not immediate score terminal")
        if terminal_copy.winner != original_player:
            raise RuntimeError("winning PASS fixture does not win for current player")
        pass_diagnostic = analyze_search_root(
            checkpoint, winning_pass, budget, top_k=10
        )
        pass_selected = pass_diagnostic["selected_action"]

        territory = territory_tactical_position()
        own_diagnostic = analyze_search_root(
            checkpoint, territory, budget, top_k=10
        )
        territory.turn = RED
        opponent_diagnostic = analyze_search_root(
            checkpoint, territory, budget, top_k=10
        )
        suicide_diagnostic = analyze_search_root(
            checkpoint, suicide_tactical_position(), budget, top_k=10
        )

        selected_remaining_captures = defense_analysis[
            "opponent_immediate_captures_after"
        ][str(defense_selected)]
        sweep.append(
            {
                "simulations": budget,
                "capture": {
                    "expected_action": expected_capture_action,
                    "selected_action": capture_selected,
                    "success": capture_selected == expected_capture_action,
                    "expected_action_stats": action_stats(
                        capture_diagnostic, expected_capture_action
                    ),
                    "selected_action_stats": action_stats(
                        capture_diagnostic, capture_selected
                    ),
                    "root": compact_diagnostic(capture_diagnostic),
                },
                "defense": {
                    "reference_defense_action": expected_defense_action,
                    "original_immediate_threat_actions": defense_analysis[
                        "original_immediate_threat_actions"
                    ],
                    "safe_defense_actions": safe_actions,
                    "selected_action": defense_selected,
                    "success": defense_selected in safe_actions,
                    "defense_success": defense_selected in safe_actions,
                    "opponent_immediate_captures_after_selected": (
                        selected_remaining_captures
                    ),
                    "selected_action_stats": action_stats(
                        defense_diagnostic, defense_selected
                    ),
                    "best_safe_defense_stats": best_safe,
                    "root": compact_diagnostic(defense_diagnostic),
                },
                "winning_pass": {
                    "fixture_terminal_result": terminal_result.name,
                    "fixture_winner": terminal_copy.winner,
                    "original_player": original_player,
                    "selected_action": pass_selected,
                    "success": pass_selected == PASS_ACTION,
                    "pass_action_stats": action_stats(
                        pass_diagnostic, PASS_ACTION
                    ),
                    "selected_action_stats": action_stats(
                        pass_diagnostic, pass_selected
                    ),
                    "root": compact_diagnostic(pass_diagnostic),
                },
                "legality": {
                    "own_territory_legal": action_stats(own_diagnostic, 0)
                    is not None,
                    "opponent_territory_excluded": action_stats(
                        opponent_diagnostic, 0
                    )
                    is None,
                    "pure_suicide_excluded": action_stats(
                        suicide_diagnostic, 1 + 9
                    )
                    is None,
                },
            }
        )
    return {
        "checkpoint_iteration": checkpoint.iteration,
        "budgets": list(budgets),
        "defense_analysis": defense_analysis,
        "results": sweep,
    }


def aggregate_budget_matchup(games, baseline, higher):
    baseline_agent = f"iter375_mcts{baseline}"
    higher_agent = f"iter375_mcts{higher}"
    selected = [
        game
        for game in games
        if {game["blue_agent"], game["red_agent"]}
        == {baseline_agent, higher_agent}
    ]
    higher_wins = [game for game in selected if game["winner_agent"] == higher_agent]
    return {
        "baseline_simulations": baseline,
        "higher_simulations": higher,
        "games": len(selected),
        "baseline_wins": sum(
            game["winner_agent"] == baseline_agent for game in selected
        ),
        "higher_budget_wins": len(higher_wins),
        "higher_budget_wins_as_blue": sum(
            game["blue_agent"] == higher_agent for game in higher_wins
        ),
        "higher_budget_wins_as_red": sum(
            game["red_agent"] == higher_agent for game in higher_wins
        ),
        "blue_total_wins": sum(game["winner_color"] == BLUE for game in selected),
        "red_total_wins": sum(game["winner_color"] == RED for game in selected),
        "capture_endings": sum(
            game["terminal_reason"] == "CAPTURE_WIN" for game in selected
        ),
        "pass_score_endings": sum(
            game["terminal_reason"] == "PASS_SCORE_END" for game in selected
        ),
        "pass_action_count": sum(game["pass_usage"] for game in selected),
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


def first_successful_budget(tactical, field):
    for item in tactical["results"]:
        if item[field]["success"]:
            return item["simulations"]
    return None


def interpret_results(tactical, arena):
    defense_budget = first_successful_budget(tactical, "defense")
    pass_budget = first_successful_budget(tactical, "winning_pass")
    higher_arena_advantage = all(
        item["higher_budget_wins"] > item["baseline_wins"] for item in arena
    )
    if defense_budget and pass_budget and higher_arena_advantage:
        case = "CASE_A"
    elif defense_budget and not pass_budget:
        case = "CASE_B"
    elif not defense_budget and not pass_budget:
        case = "CASE_C"
    elif pass_budget and not defense_budget:
        case = "CASE_D"
    else:
        case = "INCONCLUSIVE"

    pass_512 = tactical["results"][-1]["winning_pass"]
    pass_stats = pass_512["pass_action_stats"]
    potential_mcts_bug = bool(
        not pass_512["success"]
        and pass_stats is not None
        and pass_stats["visit_count"] > 0
        and not np.isclose(pass_stats["q_value_root_player"], 1.0)
    )
    if potential_mcts_bug:
        mcts_bug_diagnostic = (
            "POTENTIAL_MCTS_BUG: visited terminal PASS did not back up root Q=+1."
        )
    elif pass_stats is None or pass_stats["visit_count"] == 0:
        mcts_bug_diagnostic = (
            "No terminal backup sign bug was exercised: the winning PASS was never "
            "visited, so its terminal Q was not evaluated. This points to "
            "prior/search discovery rather than a demonstrated backup bug."
        )
    else:
        mcts_bug_diagnostic = (
            "No terminal backup sign bug observed: the visited winning PASS has "
            "root-player Q=+1."
        )
    return {
        "case": case,
        "first_observed_defense_success_budget": defense_budget,
        "first_observed_winning_pass_success_budget": pass_budget,
        "all_higher_budgets_beat_64": higher_arena_advantage,
        "potential_mcts_bug": potential_mcts_bug,
        "mcts_bug_diagnostic": mcts_bug_diagnostic,
    }


def write_summary_text(path, summary):
    tactical_by_budget = {
        item["simulations"]: item for item in summary["tactical"]["results"]
    }
    lines = [
        "# AlphaZero V2 MCTS Budget Ablation",
        "",
        f"Fixed checkpoint: iteration {summary['checkpoint_iteration']}",
        f"Checkpoint path: {summary['checkpoint_path']}",
        f"Network parameters: {summary['network_parameters']}",
        f"Tested budgets: {summary['budgets']}",
        "",
        "## Tactical budget sweep",
        "",
        "| Sims | Capture | Defense safe? | Winning PASS | Selected defense | Selected PASS |",
        "|---:|:---:|:---:|:---:|---:|---:|",
    ]
    for budget in summary["budgets"]:
        item = tactical_by_budget[budget]
        lines.append(
            f"| {budget} | {item['capture']['success']} | "
            f"{item['defense']['defense_success']} | "
            f"{item['winning_pass']['success']} | "
            f"{item['defense']['selected_action']} | "
            f"{item['winning_pass']['selected_action']} |"
        )
    defense = summary["tactical"]["defense_analysis"]
    lines.extend(
        [
            "",
            "## Defense safe-action analysis",
            "",
            f"Original immediate threats: {defense['original_immediate_threat_actions']}",
            f"Safe defense actions: {defense['safe_defense_actions']}",
            "",
            "## Winning PASS and defense root diagnosis (64 vs 512)",
            "",
        ]
    )
    for budget in (summary["budgets"][0], summary["budgets"][-1]):
        item = tactical_by_budget[budget]
        lines.extend(
            [
                f"### MCTS {budget}",
                "",
                "Defense top actions:",
                "```json",
                json.dumps(item["defense"]["root"]["top_actions"], indent=2),
                "```",
                "Winning PASS top actions:",
                "```json",
                json.dumps(item["winning_pass"]["root"]["top_actions"], indent=2),
                "```",
                "PASS stats:",
                "```json",
                json.dumps(item["winning_pass"]["pass_action_stats"], indent=2),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## 64 vs higher-budget arena",
            "",
            "| Matchup | 64 wins | Higher wins | Higher Blue | Higher Red | Blue wins | Red wins | Capture | Score | PASS actions | Mean length |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["arena"]:
        lines.append(
            f"| 64 vs {item['higher_simulations']} | "
            f"{item['baseline_wins']} | {item['higher_budget_wins']} | "
            f"{item['higher_budget_wins_as_blue']} | "
            f"{item['higher_budget_wins_as_red']} | "
            f"{item['blue_total_wins']} | {item['red_total_wins']} | "
            f"{item['capture_endings']} | {item['pass_score_endings']} | "
            f"{item['pass_action_count']} | {item['mean_game_length']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Case: {summary['interpretation']['case']}",
            (
                "First tested defense success budget: "
                f"{summary['interpretation']['first_observed_defense_success_budget']}"
            ),
            (
                "First tested winning-PASS success budget: "
                f"{summary['interpretation']['first_observed_winning_pass_success_budget']}"
            ),
            summary["interpretation"]["mcts_bug_diagnostic"],
            "",
            "Each 20-game paired-opening matchup is diagnostic, not statistical proof.",
            f"Elapsed seconds: {summary['elapsed_seconds']:.3f}",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(args):
    started = time.perf_counter()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_evaluation_checkpoint(
        args.checkpoint,
        args.device,
        expected_iteration=375,
    )
    openings_payload = json.loads(args.openings.read_text(encoding="utf-8"))
    validate_opening_suite(openings_payload)
    if openings_payload.get("seed") != 20260901 or len(openings_payload["openings"]) != 10:
        raise ValueError("ablation requires the committed 10-opening seed-20260901 suite")

    tactical = run_tactical_budget_sweep(checkpoint, args.budgets)
    atomic_json_save(tactical, args.report_dir / "tactical_budget_sweep.json")

    arena_path = args.report_dir / "arena_games.jsonl"
    games = []
    completed = set()
    if arena_path.exists():
        for line in arena_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            games.append(game)
            completed.add(
                (
                    game["higher_budget"],
                    game["opening_id"],
                    game["blue_agent"],
                )
            )

    baseline = args.budgets[0]
    with arena_path.open("a", encoding="utf-8") as handle:
        for higher in args.budgets[1:]:
            for opening in openings_payload["openings"]:
                assignments = ((baseline, higher), (higher, baseline))
                for blue_budget, red_budget in assignments:
                    blue_agent = f"iter375_mcts{blue_budget}"
                    game_key = (higher, opening["opening_id"], blue_agent)
                    if game_key in completed:
                        continue
                    game = play_search_budget_game(
                        checkpoint,
                        blue_budget,
                        red_budget,
                        opening,
                    )
                    game["baseline_budget"] = baseline
                    game["higher_budget"] = higher
                    handle.write(json.dumps(game, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    games.append(game)
                    completed.add(game_key)
                    print(
                        f"game {len(games)}/60: {game['blue_agent']} Blue vs "
                        f"{game['red_agent']} Red -> {game['winner_agent']} "
                        f"{game['terminal_reason']}",
                        flush=True,
                    )

    arena = [
        aggregate_budget_matchup(games, baseline, higher)
        for higher in args.budgets[1:]
    ]
    interpretation = interpret_results(tactical, arena)
    summary = {
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_iteration": checkpoint.iteration,
        "network_parameters": checkpoint.network.parameter_count(),
        "device": str(checkpoint.device),
        "budgets": list(args.budgets),
        "opening_suite": str(args.openings),
        "opening_count": len(openings_payload["openings"]),
        "total_arena_games": len(games),
        "tactical": tactical,
        "arena": arena,
        "interpretation": interpretation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json_save(summary, args.report_dir / "summary.json")
    write_summary_text(args.report_dir / "summary.txt", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    args = parser.parse_args(argv)
    if tuple(args.budgets) != DEFAULT_BUDGETS:
        parser.error("this fixed ablation requires budgets: 64 128 256 512")
    return args


if __name__ == "__main__":
    run_ablation(parse_args())
