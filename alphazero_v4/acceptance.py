"""Fixed-checkpoint acceptance: production search, paired openings, and diagnostics."""

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from alphazero_v2.evaluate import (
    _root_child_record, apply_opening, capture_tactical_position,
    defense_tactical_position, select_evaluation_action,
    winning_pass_tactical_position,
)
from alphazero_v3.value_oracle_audit import immediate_winning_actions
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE, RED, PASS_ACTION, NUM_ACTIONS, GreatKingdomLogicV2, MoveResultV2,
    determine_scoring_winner,
)
from .self_play import select_root_action


SEED = 20260906
SIMULATIONS = 256
MODES = ("NORMAL", "IMMEDIATE_WIN", "SAFE_DEFENSE", "FORCED_LOSS")
LEGAL_RESULTS = (
    MoveResultV2.NORMAL, MoveResultV2.PASS,
    MoveResultV2.CAPTURE_WIN, MoveResultV2.PASS_SCORE_END,
)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def position_record(logic):
    counts = logic.territory_counts()
    return {
        "player": logic.turn,
        "territory_blue": counts[BLUE], "territory_red": counts[RED],
        "current_player_territory": counts[logic.turn],
        "opponent_territory": counts[3 - logic.turn],
        "consecutive_passes": logic.consecutive_passes,
    }


def action_record(action):
    return {
        "action": int(action), "is_pass": action == PASS_ACTION,
        "coordinate": None if action == PASS_ACTION else [action % 9, action // 9],
    }


def select_v4_with_diagnostics(checkpoint, logic, simulations=SIMULATIONS, top_k=10):
    """The production selector, with its actual searched root exposed for logging."""
    config = replace(checkpoint.config, mcts_simulations=int(simulations))
    if config.c_puct != 1.5:
        raise ValueError("acceptance/UI requires the fixed c_puct=1.5")
    selected = select_root_action(
        checkpoint.network, logic, config, checkpoint.device,
        np.random.default_rng(0), ply=8, add_root_noise=False,
        temperature_override=0.0,
    )
    if selected.root is None:
        # These are exact solver targets, not invented MCTS visit counts/priors.
        actions = [
            {**action_record(a), "visit_count": None, "visit_fraction": None,
             "prior": None, "q_value_root_player": 1.0,
             "exact_target_probability": float(selected.policy[a])}
            for a in selected.tactical.immediate_win_actions
        ]
    else:
        total = sum(child.visit_count for child in selected.root.children.values())
        actions = [
            _root_child_record(selected.root, logic, a, total)
            for a in selected.root.children
        ]
        actions.sort(key=lambda item: (-item["visit_count"], item["action"]))
    return {
        **action_record(selected.action),
        "tactical_mode": selected.tactical.mode,
        "immediate_win_actions": list(selected.tactical.immediate_win_actions),
        "safe_defense_actions": list(selected.tactical.safe_defense_actions),
        "unsafe_actions_filtered": (
            len(selected.tactical.exact_unsafe_actions)
            if selected.tactical.mode == "SAFE_DEFENSE" else 0
        ),
        "mcts_executed": selected.root is not None,
        "top_actions": actions[:top_k],
        "mcts_simulations": int(simulations),
    }


def generate_acceptance_openings(count=50, seed=SEED):
    rng = np.random.default_rng(seed)
    openings, seen = [], set()
    for _ in range(count * 100):
        if len(openings) == count:
            break
        logic = GreatKingdomLogicV2()
        actions = []
        for _ in range(int(rng.integers(2, 7))):
            candidates = [
                int(a) for a in np.flatnonzero(action_mask_for_logic(logic, logic.turn)[:PASS_ACTION])
                if logic.classify_placement(logic.turn, int(a) % 9, int(a) // 9)
                == MoveResultV2.NORMAL
            ]
            if not candidates:
                raise RuntimeError("opening has no nonterminal placement")
            action = int(rng.choice(candidates))
            if logic.apply_action(action) != MoveResultV2.NORMAL:
                raise RuntimeError("opening unexpectedly terminal")
            actions.append(action)
        key = tuple(tuple(row) for row in logic.board)
        if key not in seen:
            seen.add(key)
            openings.append({"opening_id": len(openings), "actions": actions,
                             "resulting_turn": logic.turn})
    if len(openings) != count:
        raise RuntimeError("could not create unique openings")
    payload = {"seed": seed, "count": count, "min_plies": 2, "max_plies": 6,
               "openings": openings}
    validate_acceptance_openings(payload)
    return payload


def validate_acceptance_openings(payload):
    if len(payload["openings"]) != payload["count"]:
        raise ValueError("opening count mismatch")
    ids, boards = set(), set()
    for opening in payload["openings"]:
        if not 2 <= len(opening["actions"]) <= 6:
            raise ValueError("opening must have 2..6 placements")
        logic = apply_opening(opening["actions"])
        key = tuple(tuple(row) for row in logic.board)
        if opening["opening_id"] in ids or key in boards:
            raise ValueError("duplicate opening identity/position")
        if opening["resulting_turn"] != logic.turn:
            raise ValueError("opening turn mismatch")
        ids.add(opening["opening_id"])
        boards.add(key)
    return True


def assert_v4_tactical_choice(selection, child):
    mode = selection["tactical_mode"]
    if mode == "IMMEDIATE_WIN":
        if not child.game_over or child.winner != selection["root_player"]:
            raise RuntimeError("exact immediate win layer failed")
    elif mode == "SAFE_DEFENSE":
        if child.game_over:
            unsafe = child.winner != selection["root_player"]
        else:
            unsafe = bool(immediate_winning_actions(child, child.turn))
        if unsafe:
            raise RuntimeError("exact safe-defense layer failed")


def play_acceptance_game(v4, opponent, opening, v4_color, opponent_id,
                         v4_selector=select_v4_with_diagnostics,
                         historical_selector=select_evaluation_action):
    logic = apply_opening(opening["actions"])
    moves = []
    # Historical agents retain their checkpoint's search budget and encoder.
    opponent_sims = int(opponent.config["mcts_simulations"])
    while not logic.game_over:
        ply = len(opening["actions"]) + len(moves)
        # 80 castles plus interleaved single passes and final double pass <= 162.
        if ply >= 200:
            raise RuntimeError("Rules V2 arena exceeded 200 plies")
        before = position_record(logic)
        player = logic.turn
        is_v4 = player == v4_color
        if is_v4:
            selection = v4_selector(v4, logic, SIMULATIONS)
            action = selection["action"]
        else:
            action = int(historical_selector(opponent, logic, opponent_sims))
            selection = {**action_record(action), "tactical_mode": None}
        mask = action_mask_for_logic(logic, player)
        if not 0 <= action < NUM_ACTIONS or not mask[action]:
            raise RuntimeError(f"illegal arena action: {action}")
        result = logic.apply_action(action)
        if result not in LEGAL_RESULTS:
            raise RuntimeError(f"arena transition failed: {result.name}")
        if is_v4:
            assert_v4_tactical_choice({**selection, "root_player": player}, logic)
        moves.append({
            "ply": ply, **before, **action_record(action),
            "agent": "v4_iter50" if is_v4 else opponent_id,
            "tactical_mode": selection["tactical_mode"],
            "result": result.name,
        })
    return {
        "matchup": opponent_id, "opening_id": opening["opening_id"],
        "opening_actions": opening["actions"],
        "v4_color": v4_color,
        "blue_agent": "v4_iter50" if v4_color == BLUE else opponent_id,
        "red_agent": "v4_iter50" if v4_color == RED else opponent_id,
        "v4_mcts_simulations": SIMULATIONS, "opponent_mcts_simulations": opponent_sims,
        "winner_color": logic.winner,
        "winner_agent": "v4_iter50" if logic.winner == v4_color else opponent_id,
        "terminal_reason": result.name, "score_blue": logic.score_blue,
        "score_red": logic.score_red,
        "game_length": len(opening["actions"]) + len(moves),
        "pass_actions": sum(m["is_pass"] for m in moves),
        "illegal_violations": 0, "tactical_failures": 0, "moves": moves,
    }


def paired_bootstrap(games, replicates=10_000, seed=SEED):
    groups = defaultdict(list)
    for game in games:
        groups[game["opening_id"]].append(game)
    pair_scores = []
    for opening_id in sorted(groups):
        pair = groups[opening_id]
        if len(pair) != 2 or {g["v4_color"] for g in pair} != {BLUE, RED}:
            raise ValueError("each bootstrap unit needs exactly one game per color")
        if pair[0]["opening_actions"] != pair[1]["opening_actions"]:
            raise ValueError("paired games must reuse identical opening actions")
        pair_scores.append(np.mean([g["winner_agent"] == "v4_iter50" for g in pair]))
    if not pair_scores:
        raise ValueError("empty paired bootstrap")
    scores = np.asarray(pair_scores)
    rng = np.random.default_rng(seed)
    estimates = scores[rng.integers(len(scores), size=(replicates, len(scores)))].mean(axis=1)
    ci = np.quantile(estimates, [0.025, 0.975])
    return {
        "unit": "opening (two color-swapped games)", "pairs": len(scores),
        "replicates": replicates, "seed": seed,
        "v4_win_rate": float(scores.mean()),
        "v4_win_rate_ci95": ci.tolist(),
        "win_rate_difference_v4_minus_opponent": float(2 * scores.mean() - 1),
        "difference_ci95": (2 * ci - 1).tolist(),
        "lift_over_50_percent_ci95": (ci - 0.5).tolist(),
    }


def summarize_matchup(games):
    v4_moves = [m for g in games for m in g["moves"] if m["agent"] == "v4_iter50"]
    territorial = lambda m: m["territory_blue"] + m["territory_red"] > 0
    score_games = [g for g in games if g["terminal_reason"] == "PASS_SCORE_END"]
    return {
        "games": len(games),
        "v4_wins": sum(g["winner_agent"] == "v4_iter50" for g in games),
        "opponent_wins": sum(g["winner_agent"] != "v4_iter50" for g in games),
        "v4_by_color": {str(color): {
            "games": sum(g["v4_color"] == color for g in games),
            "wins": sum(g["v4_color"] == color and g["winner_agent"] == "v4_iter50" for g in games),
        } for color in (BLUE, RED)},
        "blue_wins": sum(g["winner_color"] == BLUE for g in games),
        "red_wins": sum(g["winner_color"] == RED for g in games),
        "capture_endings": len(games) - len(score_games),
        "pass_score_endings": len(score_games),
        "pass_actions": sum(g["pass_actions"] for g in games),
        "mean_game_length": float(np.mean([g["game_length"] for g in games])),
        "territory_score_distribution": dict(Counter(
            f"Blue={g['score_blue']},Red={g['score_red']}" for g in score_games)),
        "paired_bootstrap": paired_bootstrap(games),
        "v4_behavior": {
            "states": len(v4_moves),
            "states_with_nonzero_territory": sum(territorial(m) for m in v4_moves),
            "nonzero_territory_fraction": float(np.mean([territorial(m) for m in v4_moves])),
            "states_with_current_territory": sum(m["current_player_territory"] > 0 for m in v4_moves),
            "states_with_opponent_territory": sum(m["opponent_territory"] > 0 for m in v4_moves),
            "games_reaching_nonzero_territory": sum(any(territorial(m) for m in g["moves"]) for g in games),
            "games_with_nonzero_territory_at_v4_turn": sum(any(territorial(m) and m["agent"] == "v4_iter50" for m in g["moves"]) for g in games),
            "pass_actions": sum(m["is_pass"] for m in v4_moves),
            "tactical_modes": {mode: sum(m["tactical_mode"] == mode for m in v4_moves) for mode in MODES},
        },
        "illegal_violations": sum(g["illegal_violations"] for g in games),
        "tactical_failures": sum(g["tactical_failures"] for g in games),
    }


def strategic_positions():
    positions = []
    for name, builder in (("capture", capture_tactical_position), ("defense", defense_tactical_position)):
        for player in (BLUE, RED):
            logic = builder()
            if player == RED:
                logic.board = [[3 - cell if cell in (BLUE, RED) else cell for cell in row] for row in logic.board]
                logic.castles_remaining = {BLUE: logic.castles_remaining[RED], RED: logic.castles_remaining[BLUE]}
                logic.turn = RED
            positions.append((f"exact_{name}_{player}", logic, True))
    positions.append(("exact_winning_pass_blue", winning_pass_tactical_position(), True))
    red_pass = apply_opening([1, 80, 9])
    red_pass.consecutive_passes = 1
    positions.append(("exact_winning_pass_red", red_pass, True))
    for name, prefix in (
        ("D_established_territory", [2, 80, 11, 79, 20, 78, 18, 77, 19, 76]),
        ("E_blue_needs_more_territory", [1, 80, 9, 79]),
        ("F_red_scoring_advantage", [1, 80, 9]),
        ("G_own_territory_cost", [9, 80, 10, 79, 2, 78]),
    ):
        positions.append((name, apply_opening(prefix), False))
    return positions


def run_strategic_suite(checkpoint):
    results = []
    for name, logic, exact in strategic_positions():
        before = position_record(logic)
        selected = select_v4_with_diagnostics(checkpoint, logic)
        child = logic.copy()
        result = child.apply_action(selected["action"])
        if result not in LEGAL_RESULTS:
            raise RuntimeError("strategic suite illegal action")
        threat_actions = [] if child.game_over else immediate_winning_actions(child, child.turn)
        allows_loss = (child.game_over and child.winner != logic.turn) or bool(threat_actions)
        if exact:
            success = (not allows_loss if "defense" in name else child.game_over and child.winner == logic.turn)
        else:
            success = None
        own_costs = []
        for a in range(PASS_ACTION):
            if logic.get_territory_owner(a % 9, a // 9) == logic.turn:
                candidate = logic.copy()
                if candidate.apply_action(a) in LEGAL_RESULTS:
                    own_costs.append({"action": a, "territory_delta": candidate.count_territory(logic.turn) - before["current_player_territory"]})
        after = child.territory_counts()
        results.append({
            "scenario": name, "exact_assertion": exact, "success": success,
            "state": {"board": logic.board, "castles_remaining": logic.castles_remaining, **before},
            "scoring_winner_if_both_pass": determine_scoring_winner(before["territory_blue"], before["territory_red"]),
            "selection": selected, "result": result.name,
            "territory_delta": {"blue": after[BLUE] - before["territory_blue"], "red": after[RED] - before["territory_red"]},
            "allows_immediate_tactical_loss": bool(allows_loss),
            "opponent_immediate_winning_actions": list(threat_actions),
            "own_territory_placement_costs": own_costs,
        })
    return results


CLASSIFICATION_POLICY = {
    "borderline_win_rate": [0.45, 0.55],
    "competitive_per_color_minimum": 0.30,
    "extreme_color_collapse_maximum": 0.10,
    "auto_pass_requires_difference_ci_lower_above_zero": True,
    "note": "Conservative operational thresholds fixed before this arena; human acceptance still pending.",
}


def classify_acceptance(matchups, strategic):
    if any(r["exact_assertion"] and not r["success"] for r in strategic):
        return "AUTO_FAIL", "exact tactical fixture failure"
    if any(m["illegal_violations"] or m["tactical_failures"] for m in matchups.values()):
        return "AUTO_FAIL", "illegal action or tactical failure"
    for name, m in matchups.items():
        rates = [v["wins"] / v["games"] for v in m["v4_by_color"].values()]
        if min(rates) <= 0.10:
            return "AUTO_FAIL", f"extreme color collapse against {name}"
    m = matchups["v3_iter50"]
    rate = m["paired_bootstrap"]["v4_win_rate"]
    low, high = m["paired_bootstrap"]["difference_ci95"]
    if rate < 0.45 and high < 0:
        return "AUTO_FAIL", "clear paired-opening disadvantage against V3"
    rates = [v["wins"] / v["games"] for v in m["v4_by_color"].values()]
    if rate > 0.55 and low > 0 and min(rates) >= 0.30:
        return "AUTO_PASS", "V3 advantage with paired CI and competitive results in both colors"
    return "AUTO_INCONCLUSIVE", "borderline/uncertain V3 advantage or insufficient evidence in both colors"
