"""Fixed-network c_puct diagnostics and paired-opening arena helpers."""

from dataclasses import replace

import numpy as np

from alphazero_v2.evaluate import (
    analyze_safe_defense_actions,
    capture_tactical_position,
    defense_tactical_position,
    play_arena_game,
    winning_pass_tactical_position,
)
from great_kingdom_v2 import BOARD_SIZE, PASS_ACTION

from .search_guidance_audit import (
    SEARCH_GUIDANCE_MODES,
    analyze_guidance_mode,
)


DEFAULT_C_PUCT_VALUES = (1.5, 3.0, 6.0, 12.0)
FIXED_SIMULATIONS = 256
PRODUCTION_GUIDANCE_MODE = SEARCH_GUIDANCE_MODES[0]


def _fixture_specs():
    defense_safety = analyze_safe_defense_actions(defense_tactical_position())
    return (
        {
            "name": "immediate_capture",
            "builder": capture_tactical_position,
            "expected_action": 1 + 2 * BOARD_SIZE,
            "safe_actions": None,
        },
        {
            "name": "immediate_capture_threat_defense",
            "builder": defense_tactical_position,
            "expected_action": 1 + 2 * BOARD_SIZE,
            "safe_actions": defense_safety["safe_defense_actions"],
        },
        {
            "name": "immediate_winning_pass",
            "builder": winning_pass_tactical_position,
            "expected_action": PASS_ACTION,
            "safe_actions": None,
        },
    )


def run_puct_tactical_sweep(
    checkpoint,
    c_puct_values=DEFAULT_C_PUCT_VALUES,
    simulations=FIXED_SIMULATIONS,
):
    values = tuple(float(value) for value in c_puct_values)
    if values != DEFAULT_C_PUCT_VALUES:
        raise ValueError("fixed ablation requires c_puct 1.5, 3.0, 6.0, 12.0")
    records = []
    for c_puct in values:
        for fixture in _fixture_specs():
            record = analyze_guidance_mode(
                checkpoint,
                fixture["builder"](),
                fixture["expected_action"],
                PRODUCTION_GUIDANCE_MODE,
                simulations=simulations,
                c_puct=c_puct,
                safe_actions=fixture["safe_actions"],
                top_k=10,
            )
            record["fixture"] = fixture["name"]
            records.append(record)
    return {
        "checkpoint_iteration": int(checkpoint.iteration),
        "simulations": int(simulations),
        "c_puct_values": list(values),
        "guidance_mode": PRODUCTION_GUIDANCE_MODE.name,
        "safe_defense_actions": analyze_safe_defense_actions(
            defense_tactical_position()
        )["safe_defense_actions"],
        "records": records,
    }


def tactical_successes(tactical, c_puct):
    return {
        record["fixture"]: bool(record["success"])
        for record in tactical["records"]
        if np.isclose(record["c_puct"], c_puct)
    }


def select_arena_candidates(tactical, maximum=2):
    """Select improved outcomes, or a newly discovered terminal-win candidate."""
    values = [float(value) for value in tactical["c_puct_values"]]
    baseline = tactical_successes(tactical, values[0])
    baseline_score = sum(baseline.values())
    representatives = []
    seen_signatures = set()
    for value in values[1:]:
        successes = tactical_successes(tactical, value)
        signature = tuple(sorted(successes.items()))
        if sum(successes.values()) <= baseline_score or signature in seen_signatures:
            continue
        representatives.append((value, sum(successes.values())))
        seen_signatures.add(signature)
    if len(representatives) <= maximum:
        selected = [value for value, _ in representatives]
    else:
        first = representatives[0]
        best = max(representatives[1:], key=lambda item: (item[1], -item[0]))
        selected = [first[0], best[0]]
    if selected:
        return selected

    terminal_fixtures = ("immediate_capture", "immediate_winning_pass")
    baseline_visits = {
        record["fixture"]: record["expected_action_stats"]["visit_count"]
        for record in tactical["records"]
        if np.isclose(record["c_puct"], values[0])
        and record["fixture"] in terminal_fixtures
    }
    for value in values[1:]:
        records = {
            record["fixture"]: record
            for record in tactical["records"]
            if np.isclose(record["c_puct"], value)
            and record["fixture"] in terminal_fixtures
        }
        if len(records) == len(terminal_fixtures) and all(
            records[fixture]["expected_action_stats"]["visit_count"]
            > baseline_visits[fixture]
            for fixture in terminal_fixtures
        ):
            return [value]
    return []


def classify_puct_sweep(tactical):
    values = [float(value) for value in tactical["c_puct_values"]]
    outcomes = [tactical_successes(tactical, value) for value in values]
    fixtures = tuple(outcomes[0])
    regression = any(
        earlier[fixture] and not later[fixture]
        for index, earlier in enumerate(outcomes[:-1])
        for later in outcomes[index + 1 :]
        for fixture in fixtures
    )
    highest = outcomes[-1]
    terminal_recovered = (
        highest["immediate_capture"] and highest["immediate_winning_pass"]
    )
    defense_ever_improved = any(
        outcome["immediate_capture_threat_defense"] for outcome in outcomes[1:]
    )
    if regression:
        return "CASE_OVEREXPLORE"
    if not terminal_recovered:
        return "CASE_VALUE"
    if defense_ever_improved:
        return "CASE_PUCT"
    return "CASE_PARTIAL"


def checkpoint_with_c_puct(checkpoint, c_puct):
    if float(c_puct) <= 0.0:
        raise ValueError("c_puct must be positive")
    config = dict(checkpoint.config)
    config["c_puct"] = float(c_puct)
    return replace(checkpoint, config=config)


def puct_agent_id(checkpoint, c_puct):
    return f"iter{checkpoint.iteration}_cpuct{float(c_puct):g}"


def play_puct_arena_game(
    checkpoint,
    blue_c_puct,
    red_c_puct,
    opening,
    simulations=FIXED_SIMULATIONS,
):
    blue = checkpoint_with_c_puct(checkpoint, blue_c_puct)
    red = checkpoint_with_c_puct(checkpoint, red_c_puct)
    game = play_arena_game(
        blue,
        red,
        opening,
        simulations=simulations,
        blue_agent=puct_agent_id(checkpoint, blue_c_puct),
        red_agent=puct_agent_id(checkpoint, red_c_puct),
        blue_simulations=simulations,
        red_simulations=simulations,
    )
    game["blue_c_puct"] = float(blue_c_puct)
    game["red_c_puct"] = float(red_c_puct)
    return game


def aggregate_puct_matchup(games, checkpoint_iteration, baseline, candidate):
    baseline_agent = f"iter{checkpoint_iteration}_cpuct{float(baseline):g}"
    candidate_agent = f"iter{checkpoint_iteration}_cpuct{float(candidate):g}"
    selected = [
        game
        for game in games
        if {game["blue_agent"], game["red_agent"]}
        == {baseline_agent, candidate_agent}
    ]
    candidate_wins = [
        game for game in selected if game["winner_agent"] == candidate_agent
    ]
    lengths = [game["game_length"] for game in selected]
    passes = [game["pass_usage"] for game in selected]
    return {
        "baseline_c_puct": float(baseline),
        "candidate_c_puct": float(candidate),
        "games": len(selected),
        "baseline_wins": sum(
            game["winner_agent"] == baseline_agent for game in selected
        ),
        "candidate_wins": len(candidate_wins),
        "candidate_wins_as_blue": sum(
            game["blue_agent"] == candidate_agent for game in candidate_wins
        ),
        "candidate_wins_as_red": sum(
            game["red_agent"] == candidate_agent for game in candidate_wins
        ),
        "blue_total_wins": sum(game["winner_color"] == 1 for game in selected),
        "red_total_wins": sum(game["winner_color"] == 2 for game in selected),
        "capture_endings": sum(
            game["terminal_reason"] == "CAPTURE_WIN" for game in selected
        ),
        "pass_score_endings": sum(
            game["terminal_reason"] == "PASS_SCORE_END" for game in selected
        ),
        "pass_action_count": sum(passes),
        "mean_game_length": float(np.mean(lengths)) if lengths else None,
        "mean_pass_usage": float(np.mean(passes)) if passes else None,
    }
