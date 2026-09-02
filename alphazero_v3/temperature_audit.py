"""Fixed-network self-play diagnostics for action-temperature schedules.

This module intentionally leaves the production self-play and MCTS code
unchanged.  Temperature is applied only after a noisy MCTS root has been
created, when the actual game action is selected.
"""

from collections import Counter
import hashlib
import math

import numpy as np
import torch

from alphazero_v2.config import AlphaZeroConfig
from alphazero_v2.evaluate import (
    analyze_safe_defense_actions,
    immediate_capture_actions,
)
from alphazero_v2.mcts import MCTS, visit_count_policy
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE,
    NUM_ACTIONS,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
)


SCHEDULES = ("all_hot", "early8", "greedy")
PHASES = ("early", "mid", "late")
LEGAL_RESULTS = frozenset(
    (
        MoveResultV2.NORMAL,
        MoveResultV2.CAPTURE_WIN,
        MoveResultV2.PASS,
        MoveResultV2.PASS_SCORE_END,
    )
)


def temperature_for_ply(schedule, ply):
    """Return the diagnostic action-selection temperature for a zero-based ply."""
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown temperature schedule: {schedule}")
    if int(ply) < 0:
        raise ValueError("ply must be non-negative")
    if schedule == "all_hot":
        return 1.0
    if schedule == "early8":
        return 1.0 if int(ply) < 8 else 0.0
    return 0.0


def phase_for_ply(ply):
    if int(ply) < 0:
        raise ValueError("ply must be non-negative")
    if int(ply) < 8:
        return "early"
    if int(ply) < 16:
        return "mid"
    return "late"


def select_root_action(root, temperature, rng):
    """Select from visit counts; temperature zero has a stable index tie-break."""
    policy = visit_count_policy(root, temperature=float(temperature))
    if policy.shape != (NUM_ACTIONS,) or not np.isclose(policy.sum(), 1.0):
        raise RuntimeError("MCTS returned a non-normalized 82-action policy")
    if float(temperature) <= 0.0:
        action = int(np.argmax(policy))
    else:
        action = int(rng.choice(NUM_ACTIONS, p=policy.astype(np.float64)))
    if action not in root.children:
        raise RuntimeError(f"temperature selection chose missing child {action}")
    return action, policy


def immediate_winning_pass(logic):
    """Whether PASS ends scoring immediately with the current player winning."""
    if logic.game_over:
        return False
    player = logic.turn
    candidate = logic.copy()
    result = candidate.apply_action(PASS_ACTION)
    return (
        result == MoveResultV2.PASS_SCORE_END
        and candidate.game_over
        and candidate.winner == player
    )


def defense_opportunity(logic):
    """Return one-ply threats and safe replies, using the established oracle."""
    opponent = 3 - logic.turn
    threats = immediate_capture_actions(logic, opponent)
    if not threats:
        return {
            "opponent_threat_actions": [],
            "safe_defense_actions": [],
            "is_opportunity": False,
        }
    analysis = analyze_safe_defense_actions(logic)
    safe = [int(action) for action in analysis["safe_defense_actions"]]
    return {
        "opponent_threat_actions": [int(action) for action in threats],
        "safe_defense_actions": safe,
        "is_opportunity": bool(safe),
    }


def network_value(checkpoint, logic):
    encoded = (
        torch.from_numpy(checkpoint.state_encoder(logic))
        .unsqueeze(0)
        .to(checkpoint.device)
    )
    was_training = checkpoint.network.training
    checkpoint.network.eval()
    with torch.no_grad():
        _, value = checkpoint.network(encoded)
    if was_training:
        checkpoint.network.train()
    return float(value[0].item())


def network_state_digest(network):
    """Hash parameters and buffers so an audit can prove weights did not change."""
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        normalized = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(np.asarray(normalized.shape, dtype=np.int64).tobytes())
        digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()


def _audit_config(checkpoint, simulations=256, c_puct=1.5):
    config = checkpoint.config
    return AlphaZeroConfig(
        channels=int(config["channels"]),
        residual_blocks=int(config["residual_blocks"]),
        mcts_simulations=int(simulations),
        c_puct=float(c_puct),
        dirichlet_alpha=float(config.get("dirichlet_alpha", 0.3)),
        dirichlet_fraction=float(config.get("dirichlet_fraction", 0.25)),
        temperature=1.0,
        max_game_moves=int(config.get("max_game_moves", 200)),
    )


def _ply_rng(game_seed, ply, stream):
    # Separate root-noise and action-sampling streams.  A hot action draw must
    # not shift the subsequent root-noise stream relative to a greedy schedule.
    return np.random.default_rng(
        np.random.SeedSequence([int(game_seed), int(ply), int(stream)])
    )


def paired_game_seeds(seed, games):
    if int(games) <= 0:
        raise ValueError("games must be positive")
    sequence = np.random.SeedSequence(int(seed))
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(int(games))
    ]


def _root_statistics(root, chosen_action):
    actions = sorted(root.children)
    visits = {action: int(root.children[action].visit_count) for action in actions}
    total = sum(visits.values())
    if total <= 0:
        raise RuntimeError("MCTS root has no child visits")
    ranked = sorted(actions, key=lambda action: (-visits[action], action))
    fractions = {
        action: visits[action] / total
        for action in actions
    }
    entropy = -sum(
        fraction * math.log(fraction)
        for fraction in fractions.values()
        if fraction > 0.0
    )
    argmax_action = ranked[0]
    return {
        "chosen_visit_count": visits[int(chosen_action)],
        "chosen_visit_fraction": fractions[int(chosen_action)],
        "max_visit_fraction": fractions[argmax_action],
        "chosen_visit_rank": ranked.index(int(chosen_action)) + 1,
        "argmax_action": int(argmax_action),
        "chosen_is_argmax": int(chosen_action) == argmax_action,
        "root_visit_policy_entropy": float(entropy),
    }


def play_temperature_audit_game(
    checkpoint,
    schedule,
    game_index,
    game_seed,
    *,
    simulations=256,
    c_puct=1.5,
):
    """Play one fixed-network game and retain per-ply behavior evidence."""
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown temperature schedule: {schedule}")
    if int(simulations) <= 0:
        raise ValueError("simulations must be positive")
    if float(c_puct) != 1.5:
        raise ValueError("temperature audit fixes c_puct at 1.5")
    config = _audit_config(checkpoint, simulations, c_puct)
    logic = GreatKingdomLogicV2()
    moves = []
    illegal_violations = 0

    for ply in range(config.max_game_moves):
        player = logic.turn
        legal_mask = action_mask_for_logic(logic, player)
        captures = immediate_capture_actions(logic, player)
        defense = defense_opportunity(logic)
        winning_pass = immediate_winning_pass(logic)
        predicted_value = network_value(checkpoint, logic)

        root_rng = _ply_rng(game_seed, ply, 0)
        search = MCTS(
            checkpoint.network,
            config,
            checkpoint.device,
            state_encoder=checkpoint.state_encoder,
        )
        root = search.run(logic, add_root_noise=True, rng=root_rng)
        temperature = temperature_for_ply(schedule, ply)
        action_rng = _ply_rng(game_seed, ply, 1)
        action, selection_policy = select_root_action(
            root, temperature, action_rng
        )
        illegal_probability = float(selection_policy[~legal_mask].sum())
        if illegal_probability > 1e-8:
            illegal_violations += 1
            raise RuntimeError(
                "temperature policy assigned probability to illegal actions: "
                f"{illegal_probability}"
            )
        if not bool(legal_mask[action]):
            illegal_violations += 1
            raise RuntimeError(f"temperature audit selected illegal action {action}")

        root_stats = _root_statistics(root, action)
        move = {
            "ply": int(ply),
            "phase": phase_for_ply(ply),
            "player": int(player),
            "temperature": float(temperature),
            "chosen_action": int(action),
            "chosen_is_pass": action == PASS_ACTION,
            "predicted_value": predicted_value,
            "immediate_capture_actions": [int(item) for item in captures],
            "immediate_capture_available": bool(captures),
            "capture_taken": bool(captures and action in captures),
            "capture_missed": bool(captures and action not in captures),
            "opponent_threat_actions": defense["opponent_threat_actions"],
            "safe_defense_actions": defense["safe_defense_actions"],
            "defense_opportunity": defense["is_opportunity"],
            "safe_defense_taken": bool(
                defense["is_opportunity"]
                and action in defense["safe_defense_actions"]
            ),
            "unsafe_defense_choice": bool(
                defense["is_opportunity"]
                and action not in defense["safe_defense_actions"]
            ),
            "winning_pass_opportunity": bool(winning_pass),
            "winning_pass_chosen": bool(
                winning_pass and action == PASS_ACTION
            ),
            **root_stats,
        }
        result = logic.apply_action(action)
        if result not in LEGAL_RESULTS:
            illegal_violations += 1
            raise RuntimeError(f"selected action became illegal: {result.name}")
        move["move_result"] = result.name
        moves.append(move)
        if logic.game_over:
            terminal_reason = result.name
            break
    else:
        raise RuntimeError("temperature audit exceeded max_game_moves")

    for move in moves:
        target = 1.0 if logic.winner == move["player"] else -1.0
        error = move["predicted_value"] - target
        move["final_value_target"] = target
        move["value_absolute_error"] = abs(error)
        move["value_squared_error"] = error * error

    return {
        "schedule": schedule,
        "game_index": int(game_index),
        "game_seed": int(game_seed),
        "winner": int(logic.winner),
        "terminal_reason": terminal_reason,
        "game_length": len(moves),
        "pass_usage": sum(move["chosen_is_pass"] for move in moves),
        "score_blue": logic.score_blue,
        "score_red": logic.score_red,
        "illegal_violations": illegal_violations,
        "actions": [move["chosen_action"] for move in moves],
        "moves": moves,
    }


def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def _aggregate_moves(moves):
    count = len(moves)
    captures = sum(move["immediate_capture_available"] for move in moves)
    captures_taken = sum(move["capture_taken"] for move in moves)
    defense = sum(move["defense_opportunity"] for move in moves)
    safe_taken = sum(move["safe_defense_taken"] for move in moves)
    winning_pass = sum(move["winning_pass_opportunity"] for move in moves)
    winning_pass_chosen = sum(move["winning_pass_chosen"] for move in moves)
    pass_choices = sum(move["chosen_is_pass"] for move in moves)
    return {
        "moves": count,
        "argmax_action_count": sum(move["chosen_is_argmax"] for move in moves),
        "argmax_action_rate": _rate(
            sum(move["chosen_is_argmax"] for move in moves), count
        ),
        "mean_chosen_visit_fraction": (
            sum(move["chosen_visit_fraction"] for move in moves) / count
            if count
            else None
        ),
        "mean_max_visit_fraction": (
            sum(move["max_visit_fraction"] for move in moves) / count
            if count
            else None
        ),
        "mean_policy_entropy": (
            sum(move["root_visit_policy_entropy"] for move in moves) / count
            if count
            else None
        ),
        "capture_opportunity_count": captures,
        "capture_taken_count": captures_taken,
        "capture_missed_count": captures - captures_taken,
        "capture_take_rate": _rate(captures_taken, captures),
        "capture_miss_rate": _rate(captures - captures_taken, captures),
        "defense_opportunity_count": defense,
        "safe_defense_taken_count": safe_taken,
        "unsafe_action_taken_count": defense - safe_taken,
        "safe_defense_rate": _rate(safe_taken, defense),
        "unsafe_defense_rate": _rate(defense - safe_taken, defense),
        "winning_pass_opportunity_count": winning_pass,
        "winning_pass_chosen_count": winning_pass_chosen,
        "winning_pass_take_rate": _rate(winning_pass_chosen, winning_pass),
        "pass_choice_count": pass_choices,
        "pass_choice_rate": _rate(pass_choices, count),
        "mean_value_absolute_error": (
            sum(move["value_absolute_error"] for move in moves) / count
            if count
            else None
        ),
        "mean_value_squared_error": (
            sum(move["value_squared_error"] for move in moves) / count
            if count
            else None
        ),
    }


def aggregate_schedule_games(schedule, games):
    selected = [game for game in games if game["schedule"] == schedule]
    if not selected:
        raise ValueError(f"no games for schedule {schedule}")
    moves = [move for game in selected for move in game["moves"]]
    action_counts = Counter(move["chosen_action"] for move in moves)
    action_total = sum(action_counts.values())
    action_entropy = -sum(
        (count / action_total) * math.log(count / action_total)
        for count in action_counts.values()
    )
    result = {
        "schedule": schedule,
        "games": len(selected),
        "blue_wins": sum(game["winner"] == BLUE for game in selected),
        "red_wins": sum(game["winner"] == RED for game in selected),
        "capture_endings": sum(
            game["terminal_reason"] == MoveResultV2.CAPTURE_WIN.name
            for game in selected
        ),
        "pass_score_endings": sum(
            game["terminal_reason"] == MoveResultV2.PASS_SCORE_END.name
            for game in selected
        ),
        "mean_game_length": sum(game["game_length"] for game in selected)
        / len(selected),
        "mean_pass_usage": sum(game["pass_usage"] for game in selected)
        / len(selected),
        "illegal_violations": sum(
            game["illegal_violations"] for game in selected
        ),
        "unique_trajectory_count": len(
            {tuple(game["actions"]) for game in selected}
        ),
        "unique_opening8_count": len(
            {tuple(game["actions"][:8]) for game in selected}
        ),
        "selected_action_entropy": float(action_entropy),
        "overall": _aggregate_moves(moves),
        "phases": {
            phase: _aggregate_moves(
                [move for move in moves if move["phase"] == phase]
            )
            for phase in PHASES
        },
    }
    return result


def classify_temperature_audit(schedule_results):
    by_name = {item["schedule"]: item for item in schedule_results}
    hot = by_name["all_hot"]
    early = by_name["early8"]
    greedy = by_name["greedy"]

    def unsafe(item):
        return item["overall"]["unsafe_defense_rate"]

    enough_defense = (
        hot["overall"]["defense_opportunity_count"] >= 3
        and early["overall"]["defense_opportunity_count"] >= 3
    )
    defense_improved = (
        enough_defense
        and unsafe(hot) is not None
        and unsafe(early) is not None
        and unsafe(early) <= unsafe(hot) - 0.10
    )
    value_improved = (
        early["overall"]["mean_value_absolute_error"]
        < hot["overall"]["mean_value_absolute_error"]
    )
    late_argmax_improved = (
        early["phases"]["late"]["argmax_action_rate"] is not None
        and hot["phases"]["late"]["argmax_action_rate"] is not None
        and early["phases"]["late"]["argmax_action_rate"]
        > hot["phases"]["late"]["argmax_action_rate"]
    )
    diversity_collapsed = (
        early["unique_trajectory_count"] * 2 < hot["unique_trajectory_count"]
        or early["unique_opening8_count"] * 2 < hot["unique_opening8_count"]
    )
    enough_greedy_defense = (
        hot["overall"]["defense_opportunity_count"] >= 3
        and greedy["overall"]["defense_opportunity_count"] >= 3
    )
    greedy_improved = (
        enough_greedy_defense
        and unsafe(greedy) is not None
        and unsafe(hot) is not None
        and unsafe(greedy) <= unsafe(hot) - 0.10
        and greedy["overall"]["mean_value_absolute_error"]
        < hot["overall"]["mean_value_absolute_error"]
    )

    if defense_improved and value_improved and late_argmax_improved:
        return "CASE_TRADEOFF" if diversity_collapsed else "CASE_TEMP"
    if greedy_improved and not defense_improved:
        return "CASE_GREEDY_ONLY"
    return "CASE_NO_EFFECT"
