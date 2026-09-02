"""Fixed tactical diagnostics for territory-aware pilot checkpoints."""

from pathlib import Path

from alphazero_v2.evaluate import (
    EvaluationCheckpoint,
    analyze_safe_defense_actions,
    analyze_search_root,
    capture_tactical_position,
    defense_tactical_position,
    suicide_tactical_position,
    territory_tactical_position,
    winning_pass_tactical_position,
)
from great_kingdom_v2 import BOARD_SIZE, PASS_ACTION, RED, MoveResultV2

from .encoder import encode_state


def _action_stats(diagnostic, action):
    return next(
        (item for item in diagnostic["children"] if item["action"] == action),
        None,
    )


def _compact_root(diagnostic, top_k=5):
    return {
        "root_player": diagnostic["root_player"],
        "simulations": diagnostic["simulations"],
        "selected_action": diagnostic["selected_action"],
        "total_child_visits": diagnostic["total_child_visits"],
        "top_actions": diagnostic["top_actions"][:top_k],
        "pass_action": diagnostic["pass_action"],
    }


def checkpoint_from_training_state(state, config, device):
    return EvaluationCheckpoint(
        path=Path(f"iteration_{state.iteration:06d}.pt"),
        iteration=state.iteration,
        config=config.to_dict(),
        network=state.network,
        device=device,
        state_encoder=encode_state,
    )


def run_fixed_diagnostics(state, config, device):
    """Run the unchanged fixture suite with one fixed MCTS budget."""
    checkpoint = checkpoint_from_training_state(state, config, device)
    simulations = int(config.diagnostic_simulations)
    capture_action = 1 + 2 * BOARD_SIZE

    capture = analyze_search_root(
        checkpoint,
        capture_tactical_position(),
        simulations,
        top_k=5,
    )

    defense_logic = defense_tactical_position()
    defense_safety = analyze_safe_defense_actions(defense_logic)
    defense = analyze_search_root(
        checkpoint,
        defense_logic,
        simulations,
        top_k=5,
    )
    selected_defense = defense["selected_action"]

    winning_pass_logic = winning_pass_tactical_position()
    original_player = winning_pass_logic.turn
    terminal = winning_pass_logic.copy()
    terminal_result = terminal.apply_action(PASS_ACTION)
    if terminal_result != MoveResultV2.PASS_SCORE_END:
        raise RuntimeError("winning PASS fixture is not an immediate score terminal")
    if terminal.winner != original_player:
        raise RuntimeError("winning PASS fixture does not win for current player")
    winning_pass = analyze_search_root(
        checkpoint,
        winning_pass_logic,
        simulations,
        top_k=5,
    )

    territory = territory_tactical_position()
    own_territory = analyze_search_root(
        checkpoint,
        territory,
        simulations,
        top_k=1,
    )
    territory.turn = RED
    opponent_territory = analyze_search_root(
        checkpoint,
        territory,
        simulations,
        top_k=1,
    )
    suicide = analyze_search_root(
        checkpoint,
        suicide_tactical_position(),
        simulations,
        top_k=1,
    )

    pass_stats = _action_stats(winning_pass, PASS_ACTION)
    selected_pass = winning_pass["selected_action"]
    return {
        "iteration": int(state.iteration),
        "mcts_simulations": simulations,
        "immediate_capture": {
            "expected_action": capture_action,
            "selected_action": capture["selected_action"],
            "success": capture["selected_action"] == capture_action,
            "expected_action_stats": _action_stats(capture, capture_action),
            "root": _compact_root(capture),
        },
        "defense": {
            "original_immediate_threat_actions": defense_safety[
                "original_immediate_threat_actions"
            ],
            "safe_defense_actions": defense_safety["safe_defense_actions"],
            "selected_action": selected_defense,
            "success": selected_defense
            in defense_safety["safe_defense_actions"],
            "opponent_immediate_captures_after_selected": defense_safety[
                "opponent_immediate_captures_after"
            ][str(selected_defense)],
            "selected_action_stats": _action_stats(defense, selected_defense),
            "root": _compact_root(defense),
        },
        "winning_pass": {
            "fixture_terminal_result": terminal_result.name,
            "fixture_winner": terminal.winner,
            "original_player": original_player,
            "selected_action": selected_pass,
            "success": selected_pass == PASS_ACTION,
            "pass_action_stats": pass_stats,
            "selected_action_stats": _action_stats(winning_pass, selected_pass),
            "root": _compact_root(winning_pass),
        },
        "legality": {
            "own_territory_legal": _action_stats(own_territory, 0) is not None,
            "opponent_territory_excluded": _action_stats(
                opponent_territory, 0
            )
            is None,
            "pure_suicide_excluded": _action_stats(
                suicide, 1 + BOARD_SIZE
            )
            is None,
        },
    }
