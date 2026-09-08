"""Sixteen-game self-play with batched PUCT, exact roots, and mixed starts."""

from dataclasses import dataclass, field

import numpy as np
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE, BOARD_SIZE, NUM_ACTIONS, PASS_ACTION, GreatKingdomLogicV2, MoveResultV2,
)
from .batched_mcts import BatchedMCTS, BatchedNetworkEvaluator, SearchRequest
from .common import TrainingExample, temperature_for_ply, visit_count_policy
from .encoder import encode_state
from .network import calibrated_scalar
from .start_states import restore_logic
from .tactical import solve_tactical_root


LEGAL_RESULTS = (MoveResultV2.NORMAL, MoveResultV2.CAPTURE_WIN,
                 MoveResultV2.PASS, MoveResultV2.PASS_SCORE_END)


@dataclass
class GameContext:
    logic: object
    start_type: str
    start_ply: int
    history: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _new_context(rng, start_pool, territory_fraction):
    if start_pool and rng.random() < territory_fraction:
        logic = restore_logic(start_pool[int(rng.integers(len(start_pool)))])
        start_type = "territory_midgame"
    else:
        logic, start_type = GreatKingdomLogicV2(), "initial"
    start_ply = sum(cell in (1, 2) for row in logic.board for cell in row)
    return GameContext(logic, start_type, start_ply, metrics={
        "start_type": start_type, "immediate_win_opportunities": 0,
        "immediate_win_taken": 0, "defense_threat_states": 0,
        "defense_states_with_safe_action": 0, "unsafe_actions_filtered": 0,
        "forced_loss_states": 0, "pass_usage": 0, "early_placements": 0,
        "early_edge_placements": 0, "placements": 0,
        "own_adjacent_placements": 0, "own_adjacent_two_plus": 0,
        "territory_creation_delta": 0, "territory_disruption_delta": 0,
        "states_with_nonzero_territory": 0, "policy_entropy_sum": 0.0,
        "network_value_count": 0, "network_value_abs_ge_0_9": 0,
        "network_value_abs_ge_0_99": 0,
    })


def _adjacent_own(logic, action, player):
    if action == PASS_ACTION:
        return 0
    x, y = action % BOARD_SIZE, action // BOARD_SIZE
    return sum(logic.is_on_board(x + dx, y + dy) and logic.board[y + dy][x + dx] == player
               for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))


def _edge_action(action):
    if action == PASS_ACTION:
        return False
    x, y = action % BOARD_SIZE, action // BOARD_SIZE
    return x in (0, BOARD_SIZE - 1) or y in (0, BOARD_SIZE - 1)


def _selection_policy(root, temperature, rng):
    policy = visit_count_policy(root, temperature=temperature)
    action = int(np.argmax(policy)) if temperature <= 0 else int(rng.choice(NUM_ACTIONS, p=policy.astype(np.float64)))
    return action, policy


def _record_action(context, action, policy, tactical, root_value):
    logic, metrics = context.logic, context.metrics
    player = logic.turn
    before = logic.territory_counts()
    if sum(before.values()) > 0:
        metrics["states_with_nonzero_territory"] += 1
    if tactical.mode == "IMMEDIATE_WIN":
        metrics["immediate_win_opportunities"] += 1
        metrics["immediate_win_taken"] += action in tactical.immediate_win_actions
    if tactical.opponent_threat_actions:
        metrics["defense_threat_states"] += 1
    if tactical.mode == "SAFE_DEFENSE":
        metrics["defense_states_with_safe_action"] += 1
        metrics["unsafe_actions_filtered"] += len(tactical.exact_unsafe_actions)
    if tactical.mode == "FORCED_LOSS":
        metrics["forced_loss_states"] += 1
    if action == PASS_ACTION:
        metrics["pass_usage"] += 1
    else:
        metrics["placements"] += 1
        absolute_ply = context.start_ply + len(context.history)
        if absolute_ply < 12:
            metrics["early_placements"] += 1
            metrics["early_edge_placements"] += _edge_action(action)
        adjacency = _adjacent_own(logic, action, player)
        metrics["own_adjacent_placements"] += adjacency >= 1
        metrics["own_adjacent_two_plus"] += adjacency >= 2
    nonzero = policy[policy > 0]
    metrics["policy_entropy_sum"] += float(-np.sum(nonzero * np.log(nonzero)))
    metrics["network_value_count"] += 1
    metrics["network_value_abs_ge_0_9"] += abs(root_value) >= 0.9
    metrics["network_value_abs_ge_0_99"] += abs(root_value) >= 0.99
    context.history.append((encode_state(logic), policy.copy(), player))
    result = logic.apply_action(action)
    if result not in LEGAL_RESULTS:
        raise RuntimeError(f"V5 self-play selected illegal action: {result.name}")
    after = logic.territory_counts()
    metrics["territory_creation_delta"] += max(0, after[player] - before[player])
    metrics["territory_disruption_delta"] += max(0, before[3 - player] - after[3 - player])
    return result


def _finish_context(context, result):
    logic = context.logic
    examples = [TrainingExample(state, policy, 1.0 if logic.winner == player else -1.0, player)
                for state, policy, player in context.history]
    metrics = dict(context.metrics)
    metrics.update({"winner": logic.winner, "terminal_reason": result.name,
                    "game_length": len(examples), "score_blue": logic.score_blue,
                    "score_red": logic.score_red, "illegal_probability_violations": 0,
                    "entered_forced_loss": metrics["forced_loss_states"] > 0})
    count = max(1, len(examples))
    metrics["mean_policy_entropy"] = metrics.pop("policy_entropy_sum") / count
    return examples, metrics


def play_concurrent_cohort(network, calibration_temperature, config, device, rng,
                           start_pool, game_count=None):
    target = config.concurrent_games if game_count is None else int(game_count)
    if target <= 0:
        raise ValueError("game_count must be positive")
    started = min(config.concurrent_games, target)
    contexts = [_new_context(rng, start_pool, config.territory_start_fraction) for _ in range(started)]
    evaluator = BatchedNetworkEvaluator(
        network, encode_state, device,
        lambda logits: calibrated_scalar(logits, calibration_temperature),
    )
    search = BatchedMCTS(evaluator, config.c_puct, config.dirichlet_alpha,
                         config.dirichlet_fraction)
    completed_examples, completed_games = [], []
    while contexts:
        tacticals = [solve_tactical_root(context.logic) for context in contexts]
        selections = [None] * len(contexts)
        search_indices, requests = [], []
        immediate_indices = []
        for index, (context, tactical) in enumerate(zip(contexts, tacticals)):
            ply = context.start_ply + len(context.history)
            temperature = temperature_for_ply(config.temperature_schedule, ply)
            if tactical.mode == "IMMEDIATE_WIN":
                policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                policy[list(tactical.immediate_win_actions)] = 1 / len(tactical.immediate_win_actions)
                action = min(tactical.immediate_win_actions) if temperature <= 0 else int(rng.choice(NUM_ACTIONS, p=policy))
                selections[index] = [action, policy, None]
                immediate_indices.append(index)
            else:
                search_indices.append(index)
                requests.append(SearchRequest(context.logic, config.simulations_for_ply(ply),
                                              tactical.allowed_actions, True, rng))
        if immediate_indices:
            _, values = evaluator.evaluate([contexts[i].logic for i in immediate_indices])
            for index, value in zip(immediate_indices, values):
                selections[index][2] = float(value)
        if requests:
            results = search.run_many(requests)
            for index, result in zip(search_indices, results):
                context = contexts[index]
                ply = context.start_ply + len(context.history)
                temperature = temperature_for_ply(config.temperature_schedule, ply)
                action, policy = _selection_policy(result.root, temperature, rng)
                selections[index] = [action, policy, result.root_network_value]
        survivors = []
        for context, tactical, (action, policy, root_value) in zip(contexts, tacticals, selections):
            legal = action_mask_for_logic(context.logic, context.logic.turn)
            if policy.shape != (NUM_ACTIONS,) or policy[~legal].sum() > 1e-8 or not legal[action]:
                raise RuntimeError("V5 generated illegal action/policy")
            result = _record_action(context, action, policy, tactical, root_value)
            if context.logic.game_over:
                examples, metrics = _finish_context(context, result)
                completed_examples.extend(examples); completed_games.append(metrics)
                if started < target:
                    survivors.append(_new_context(rng, start_pool, config.territory_start_fraction))
                    started += 1
            elif context.start_ply + len(context.history) >= config.max_game_moves:
                raise RuntimeError("V5 self-play exceeded max game moves")
            else:
                survivors.append(context)
        contexts = survivors
    return completed_examples, completed_games, evaluator.metrics()


def generate_self_play(network, calibration_temperature, config, device, rng, start_pool,
                       games=None):
    target = config.self_play_games_per_cycle if games is None else int(games)
    return play_concurrent_cohort(
        network, calibration_temperature, config, device, rng, start_pool, target)
