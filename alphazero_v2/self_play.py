"""Minimal single-network self-play for Rules V2."""

from dataclasses import dataclass

import numpy as np

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    NUM_ACTIONS,
    PASS_ACTION,
    GreatKingdomLogicV2,
    MoveResultV2,
)

from .encoder import encode_state
from .mcts import MCTS, visit_count_policy


@dataclass
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    value: float
    player: int


def play_self_play_game(network, config, device, rng):
    logic = GreatKingdomLogicV2()
    history = []
    pass_usage = 0
    illegal_probability_violations = 0
    terminal_reason = None

    for _ in range(config.max_game_moves):
        player = logic.turn
        search = MCTS(network, config, device)
        root = search.run(logic, add_root_noise=True, rng=rng)
        policy = visit_count_policy(root, config.temperature)
        legal_mask = action_mask_for_logic(logic, player)
        illegal_probability = float(policy[~legal_mask].sum())
        if illegal_probability > 1e-8:
            illegal_probability_violations += 1
        if policy.shape != (NUM_ACTIONS,) or not np.isclose(policy.sum(), 1.0):
            raise RuntimeError("self-play MCTS policy is not a normalized 82-vector")

        history.append((encode_state(logic), policy.copy(), player))
        action = int(rng.choice(NUM_ACTIONS, p=policy.astype(np.float64)))
        if action == PASS_ACTION:
            pass_usage += 1
        result = logic.apply_action(action)
        if result not in (
            MoveResultV2.NORMAL,
            MoveResultV2.CAPTURE_WIN,
            MoveResultV2.PASS,
            MoveResultV2.PASS_SCORE_END,
        ):
            raise RuntimeError(f"self-play selected illegal action: {result.name}")
        if logic.game_over:
            terminal_reason = result.name
            break
    else:
        raise RuntimeError("self-play exceeded max_game_moves without terminal state")

    examples = [
        TrainingExample(
            state=state,
            policy=policy,
            value=1.0 if logic.winner == player else -1.0,
            player=player,
        )
        for state, policy, player in history
    ]
    stats = {
        "winner": logic.winner,
        "terminal_reason": terminal_reason,
        "game_length": len(history),
        "pass_usage": pass_usage,
        "illegal_probability_violations": illegal_probability_violations,
    }
    return examples, stats


def generate_self_play(network, config, device, rng):
    all_examples = []
    game_stats = []
    for game_index in range(config.self_play_games):
        examples, stats = play_self_play_game(network, config, device, rng)
        stats["game_index"] = game_index
        all_examples.extend(examples)
        game_stats.append(stats)
    return all_examples, game_stats
