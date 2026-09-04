"""V4 self-play with exact root tactics and the fixed early8 schedule."""

from dataclasses import dataclass

import numpy as np

from alphazero_v2.mcts import visit_count_policy
from alphazero_v2.self_play import TrainingExample
from alphazero_v3.encoder import encode_state
from alphazero_v3.temperature_audit import temperature_for_ply
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    NUM_ACTIONS,
    PASS_ACTION,
    GreatKingdomLogicV2,
    MoveResultV2,
)

from .mcts import V4MCTS
from .tactical import solve_tactical_root


@dataclass
class RootSelection:
    policy: np.ndarray
    action: int
    tactical: object


def select_root_action(
    network,
    logic,
    config,
    device,
    rng,
    *,
    ply,
    add_root_noise,
    temperature_override=None,
):
    tactical = solve_tactical_root(logic)
    temperature = (
        temperature_for_ply(config.temperature_schedule, ply)
        if temperature_override is None
        else float(temperature_override)
    )
    if tactical.mode == "IMMEDIATE_WIN":
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[list(tactical.immediate_win_actions)] = (
            1.0 / len(tactical.immediate_win_actions)
        )
        if temperature > 0.0:
            action = int(rng.choice(NUM_ACTIONS, p=policy.astype(np.float64)))
        else:
            action = min(tactical.immediate_win_actions)
    else:
        search = V4MCTS(
            network,
            config.self_play_config(),
            device,
            state_encoder=encode_state,
        )
        root = search.run(
            logic,
            root_actions=tactical.allowed_actions,
            add_root_noise=add_root_noise,
            rng=rng,
        )
        policy = visit_count_policy(root, temperature=temperature)
        if temperature > 0.0:
            action = int(rng.choice(NUM_ACTIONS, p=policy.astype(np.float64)))
        else:
            action = int(np.argmax(policy))
    legal = action_mask_for_logic(logic, logic.turn)
    illegal_probability = float(policy[~legal].sum())
    if illegal_probability > 1e-8 or not bool(legal[action]):
        raise RuntimeError("V4 root selection produced an illegal action/policy")
    return RootSelection(policy=policy, action=action, tactical=tactical)


def play_self_play_game(network, config, device, rng):
    logic = GreatKingdomLogicV2()
    history = []
    pass_usage = 0
    metrics = {
        "immediate_win_opportunities": 0,
        "immediate_win_taken": 0,
        "defense_threat_states": 0,
        "defense_states_with_safe_action": 0,
        "unsafe_actions_filtered": 0,
        "forced_loss_states": 0,
    }
    for ply in range(config.max_game_moves):
        player = logic.turn
        selected = select_root_action(
            network,
            logic,
            config,
            device,
            rng,
            ply=ply,
            add_root_noise=True,
        )
        tactical = selected.tactical
        if tactical.mode == "IMMEDIATE_WIN":
            metrics["immediate_win_opportunities"] += 1
            metrics["immediate_win_taken"] += (
                selected.action in tactical.immediate_win_actions
            )
        if tactical.opponent_threat_actions:
            metrics["defense_threat_states"] += 1
        if tactical.mode == "SAFE_DEFENSE":
            metrics["defense_states_with_safe_action"] += 1
            metrics["unsafe_actions_filtered"] += len(
                tactical.exact_unsafe_actions
            )
        if tactical.forced_loss:
            metrics["forced_loss_states"] += 1
        history.append((encode_state(logic), selected.policy.copy(), player))
        if selected.action == PASS_ACTION:
            pass_usage += 1
        result = logic.apply_action(selected.action)
        if result not in (
            MoveResultV2.NORMAL,
            MoveResultV2.CAPTURE_WIN,
            MoveResultV2.PASS,
            MoveResultV2.PASS_SCORE_END,
        ):
            raise RuntimeError(f"V4 self-play selected illegal action: {result.name}")
        if logic.game_over:
            terminal_reason = result.name
            break
    else:
        raise RuntimeError("V4 self-play exceeded max_game_moves")

    examples = [
        TrainingExample(
            state=state,
            policy=policy,
            value=1.0 if logic.winner == player else -1.0,
            player=player,
        )
        for state, policy, player in history
    ]
    targets = np.asarray([example.value for example in examples], dtype=np.float32)
    return examples, {
        "winner": logic.winner,
        "terminal_reason": terminal_reason,
        "game_length": len(examples),
        "pass_usage": pass_usage,
        "illegal_probability_violations": 0,
        "score_blue": logic.score_blue,
        "score_red": logic.score_red,
        "value_target_mean": float(np.mean(targets)),
        "value_target_std": float(np.std(targets)),
        **metrics,
    }


def generate_self_play(network, config, device, rng):
    examples = []
    games = []
    for game_index in range(config.self_play_games_per_iteration):
        game_examples, stats = play_self_play_game(
            network, config, device, rng
        )
        stats["game_index"] = game_index
        examples.extend(game_examples)
        games.append(stats)
    return examples, games
