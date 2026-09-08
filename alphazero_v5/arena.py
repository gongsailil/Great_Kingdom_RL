"""Batched deterministic paired arenas for V5 promotion and final evaluation."""

from dataclasses import dataclass, field

import numpy as np
from great_kingdom_v2 import BLUE, RED, PASS_ACTION, MoveResultV2
from .batched_mcts import BatchedMCTS, BatchedNetworkEvaluator, SearchRequest
from .common import visit_count_policy
from .openings import apply_opening
from .tactical import solve_tactical_root


LEGAL_RESULTS = (MoveResultV2.NORMAL, MoveResultV2.CAPTURE_WIN,
                 MoveResultV2.PASS, MoveResultV2.PASS_SCORE_END)


@dataclass
class ArenaAgent:
    name: str
    network: object
    encoder: object
    value_transform: object
    device: object
    simulations: int = 256
    c_puct: float = 1.5
    tactical_solver: bool = False


@dataclass
class ArenaContext:
    opening: dict
    blue: ArenaAgent
    red: ArenaAgent
    logic: object
    moves: list = field(default_factory=list)


def _root_records(root, logic, limit=5):
    total = sum(child.visit_count for child in root.children.values())
    records = []
    for action, child in root.children.items():
        q = child.value() if child.to_play == root.to_play else -child.value()
        records.append({"action": action, "visits": child.visit_count,
                        "visit_fraction": child.visit_count / total if total else 0,
                        "prior": child.prior, "q_root": q})
    records.sort(key=lambda row: (-row["visits"], row["action"]))
    return records[:limit]


def _position(logic):
    counts = logic.territory_counts()
    return {"territory_blue": counts[BLUE], "territory_red": counts[RED],
            "consecutive_passes": logic.consecutive_passes,
            "current_territory": counts[logic.turn], "opponent_territory": counts[3 - logic.turn]}


def _play_cohort(assignments, openings, concurrent_games=16):
    contexts = [ArenaContext(opening, blue, red, apply_opening(opening["actions"]))
                for opening, blue, red in assignments]
    agents = {agent.name: agent for _, blue, red in assignments for agent in (blue, red)}
    evaluators = {name: BatchedNetworkEvaluator(agent.network, agent.encoder, agent.device,
                                                agent.value_transform)
                  for name, agent in agents.items()}
    searches = {name: BatchedMCTS(evaluators[name], agent.c_puct, 0.3, 0.0)
                for name, agent in agents.items()}
    active = list(contexts)
    while active:
        selections = {}
        groups = {}
        for index, context in enumerate(active):
            agent = context.blue if context.logic.turn == BLUE else context.red
            tactical = solve_tactical_root(context.logic) if agent.tactical_solver else None
            if tactical is not None and tactical.mode == "IMMEDIATE_WIN":
                selections[index] = (min(tactical.immediate_win_actions), tactical, None, [])
            else:
                groups.setdefault(agent.name, []).append((index, context, tactical))
        for name, items in groups.items():
            agent, search = agents[name], searches[name]
            requests = [SearchRequest(item[1].logic, agent.simulations,
                                      item[2].allowed_actions if item[2] else None)
                        for item in items]
            for (index, context, tactical), result in zip(items, search.run_many(requests)):
                policy = visit_count_policy(result.root, temperature=0)
                action = int(np.argmax(policy))
                selections[index] = (action, tactical, result.root_network_value,
                                     _root_records(result.root, context.logic))
        survivors = []
        for index, context in enumerate(active):
            logic = context.logic
            player = logic.turn
            agent = context.blue if player == BLUE else context.red
            action, tactical, root_value, top = selections[index]
            before = _position(logic)
            result = logic.apply_action(action)
            if result not in LEGAL_RESULTS:
                raise RuntimeError(f"arena selected illegal action {result.name}")
            if tactical is not None and tactical.mode == "SAFE_DEFENSE":
                if logic.game_over and logic.winner != player:
                    raise RuntimeError("safe defense produced terminal loss")
                if not logic.game_over and solve_tactical_root(logic).immediate_win_actions:
                    raise RuntimeError("safe defense left immediate opponent win")
            after = logic.territory_counts()
            context.moves.append({"agent": agent.name, "player": player,
                                  "action": action, "is_pass": action == PASS_ACTION,
                                  "tactical_mode": tactical.mode if tactical else "HISTORICAL",
                                  "root_value": root_value, "root_top5": top,
                                  **before, "territory_delta_blue": after[BLUE] - before["territory_blue"],
                                  "territory_delta_red": after[RED] - before["territory_red"],
                                  "result": result.name})
            if logic.game_over:
                continue
            if len(context.opening["actions"]) + len(context.moves) >= 200:
                raise RuntimeError("arena game exceeded maximum length")
            survivors.append(context)
        active = survivors
    games = []
    for context in contexts:
        logic = context.logic
        winner = context.blue.name if logic.winner == BLUE else context.red.name
        games.append({"opening_id": context.opening["opening_id"],
                      "opening_actions": context.opening["actions"],
                      "blue_agent": context.blue.name, "red_agent": context.red.name,
                      "winner_color": logic.winner, "winner_agent": winner,
                      "terminal_reason": logic.last_move_result.name,
                      "score_blue": logic.score_blue, "score_red": logic.score_red,
                      "game_length": len(context.opening["actions"]) + len(context.moves),
                      "pass_actions": sum(move["is_pass"] for move in context.moves),
                      "moves": context.moves, "illegal_violations": 0,
                      "tactical_failures": 0})
    inference = {name: evaluator.metrics() for name, evaluator in evaluators.items()}
    return games, inference


def play_paired_arena(agent_a, agent_b, openings, concurrent_games=16):
    assignments = []
    for opening in openings:
        assignments.extend(((opening, agent_a, agent_b), (opening, agent_b, agent_a)))
    games, inference = [], {}
    for start in range(0, len(assignments), concurrent_games):
        cohort, stats = _play_cohort(assignments[start:start + concurrent_games], openings, concurrent_games)
        games.extend(cohort)
        for name, values in stats.items():
            aggregate = inference.setdefault(name, {"network_inference_calls": 0,
                                                     "network_inference_positions": 0,
                                                     "max_inference_batch_size": 0})
            aggregate["network_inference_calls"] += values["network_inference_calls"]
            aggregate["network_inference_positions"] += values["network_inference_positions"]
            aggregate["max_inference_batch_size"] = max(aggregate["max_inference_batch_size"], values["max_inference_batch_size"])
    for values in inference.values():
        values["mean_inference_batch_size"] = values["network_inference_positions"] / max(1, values["network_inference_calls"])
    return games, inference


def summarize_arena(games, candidate_name):
    wins = sum(game["winner_agent"] == candidate_name for game in games)
    colors = {str(color): {"games": sum((g["blue_agent"] if color == BLUE else g["red_agent"]) == candidate_name for g in games),
                           "wins": sum(g["winner_agent"] == candidate_name and g["winner_color"] == color for g in games)}
              for color in (BLUE, RED)}
    candidate_moves = [move for game in games for move in game["moves"] if move["agent"] == candidate_name]
    return {"games": len(games), "wins": wins, "losses": len(games) - wins,
            "win_rate": wins / len(games), "by_color": colors,
            "capture_endings": sum(g["terminal_reason"] == "CAPTURE_WIN" for g in games),
            "pass_score_endings": sum(g["terminal_reason"] == "PASS_SCORE_END" for g in games),
            "pass_actions": sum(g["pass_actions"] for g in games),
            "mean_game_length": float(np.mean([g["game_length"] for g in games])),
            "territory_state_fraction": float(np.mean([m["territory_blue"] + m["territory_red"] > 0 for m in candidate_moves])),
            "forced_loss_entries": sum(m["tactical_mode"] == "FORCED_LOSS" for m in candidate_moves),
            "candidate_moves": len(candidate_moves), "illegal_violations": 0,
            "tactical_failures": 0}
