"""Post-run V4 tactical, value-oracle, and V3 cross-version evaluation."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from alphazero_v2.evaluate import (
    apply_opening,
    load_evaluation_checkpoint,
    select_evaluation_action,
    validate_opening_suite,
)
from alphazero_v3.encoder import encode_state
from great_kingdom_v2 import BLUE, PASS_ACTION, MoveResultV2

from .config import V4Config
from .diagnostics import run_fixed_tactical_diagnostics, run_value_oracle_monitor
from .network import PolicyValueLogitNetwork
from .self_play import select_root_action


@dataclass
class V4EvaluationCheckpoint:
    path: Path
    iteration: int
    config: V4Config
    network: PolicyValueLogitNetwork
    device: torch.device


def load_v4_evaluation_checkpoint(path, device="auto", expected_iteration=None):
    path = Path(path)
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("architecture") != "alphazero_v4_raw_value_logit":
        raise ValueError("not an AlphaZero V4 checkpoint")
    iteration = int(payload["iteration"])
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise ValueError("V4 checkpoint iteration mismatch")
    config = V4Config.from_dict(dict(payload["config"]))
    network = PolicyValueLogitNetwork(
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        input_planes=config.input_planes,
    ).to(device)
    network.load_state_dict(payload["network_state_dict"])
    network.eval()
    return V4EvaluationCheckpoint(path, iteration, config, network, device)


def select_v4_evaluation_action(checkpoint, logic, ply):
    selected = select_root_action(
        checkpoint.network,
        logic,
        checkpoint.config,
        checkpoint.device,
        np.random.default_rng(0),
        ply=ply,
        add_root_noise=False,
        temperature_override=0.0,
    )
    return selected.action, selected.tactical


def play_v4_v3_game(v4, v3, opening, *, v4_color, simulations=256):
    if int(simulations) != 256:
        raise ValueError("V4 post-run arena is fixed at 256 simulations")
    logic = apply_opening(opening["actions"])
    actions = []
    pass_usage = 0
    tactical_counts = {
        "immediate_win": 0,
        "safe_defense": 0,
        "forced_loss": 0,
    }
    for ply in range(len(opening["actions"]), 200):
        if logic.turn == int(v4_color):
            action, tactical = select_v4_evaluation_action(v4, logic, ply)
            if tactical.mode == "IMMEDIATE_WIN":
                tactical_counts["immediate_win"] += 1
            elif tactical.mode == "SAFE_DEFENSE":
                tactical_counts["safe_defense"] += 1
            elif tactical.mode == "FORCED_LOSS":
                tactical_counts["forced_loss"] += 1
        else:
            action = select_evaluation_action(v3, logic, simulations)
        result = logic.apply_action(action)
        if result not in (
            MoveResultV2.NORMAL,
            MoveResultV2.CAPTURE_WIN,
            MoveResultV2.PASS,
            MoveResultV2.PASS_SCORE_END,
        ):
            raise RuntimeError(f"cross-version arena action became {result.name}")
        actions.append(int(action))
        pass_usage += action == PASS_ACTION
        if logic.game_over:
            break
    else:
        raise RuntimeError("cross-version arena exceeded 200 total plies")
    return {
        "opening_id": int(opening["opening_id"]),
        "opening_actions": list(opening["actions"]),
        "v4_color": int(v4_color),
        "blue_agent": "v4_iter50" if v4_color == BLUE else "v3_iter50",
        "red_agent": "v3_iter50" if v4_color == BLUE else "v4_iter50",
        "winner_color": int(logic.winner),
        "winner_agent": (
            "v4_iter50" if logic.winner == int(v4_color) else "v3_iter50"
        ),
        "terminal_reason": result.name,
        "game_length": len(opening["actions"]) + len(actions),
        "pass_usage": int(pass_usage),
        "score_blue": logic.score_blue,
        "score_red": logic.score_red,
        "actions": actions,
        "v4_tactical_counts": tactical_counts,
        "mcts_simulations": int(simulations),
    }


def run_cross_version_arena(v4, v3_path, openings_path):
    v3 = load_evaluation_checkpoint(
        v3_path,
        device=v4.device,
        expected_iteration=50,
        state_encoder=encode_state,
    )
    openings = json.loads(Path(openings_path).read_text(encoding="utf-8"))
    validate_opening_suite(openings)
    games = []
    for opening in openings["openings"]:
        games.append(play_v4_v3_game(v4, v3, opening, v4_color=BLUE))
        games.append(play_v4_v3_game(v4, v3, opening, v4_color=2))
    return games


def aggregate_arena(games):
    return {
        "games": len(games),
        "v4_wins": sum(game["winner_agent"] == "v4_iter50" for game in games),
        "v3_wins": sum(game["winner_agent"] == "v3_iter50" for game in games),
        "v4_wins_as_blue": sum(
            game["winner_agent"] == "v4_iter50" and game["v4_color"] == BLUE
            for game in games
        ),
        "v4_wins_as_red": sum(
            game["winner_agent"] == "v4_iter50" and game["v4_color"] == 2
            for game in games
        ),
        "blue_wins": sum(game["winner_color"] == BLUE for game in games),
        "red_wins": sum(game["winner_color"] == 2 for game in games),
        "capture_endings": sum(
            game["terminal_reason"] == "CAPTURE_WIN" for game in games
        ),
        "pass_score_endings": sum(
            game["terminal_reason"] == "PASS_SCORE_END" for game in games
        ),
        "mean_game_length": float(
            np.mean([game["game_length"] for game in games])
        ),
        "mean_pass_usage": float(np.mean([game["pass_usage"] for game in games])),
    }


def training_trends(metrics):
    blocks = []
    for start in range(1, 51, 10):
        selected = [
            metric for metric in metrics if start <= metric["iteration"] < start + 10
        ]
        blocks.append(
            {
                "iterations": [start, start + 9],
                "games": sum(metric["new_games"] for metric in selected),
                "samples": sum(metric["new_samples"] for metric in selected),
                "capture_endings": sum(
                    metric["capture_endings"] for metric in selected
                ),
                "pass_score_endings": sum(
                    metric["pass_score_endings"] for metric in selected
                ),
                "pass_action_count": sum(
                    metric["pass_action_count"] for metric in selected
                ),
                "mean_game_length": float(
                    np.mean([metric["mean_game_length"] for metric in selected])
                ),
            }
        )
    return blocks


def classify_result(summary):
    baseline = summary["v3_baseline"]
    oracle = summary["final_value_oracle"]
    exact = oracle["exact_loss"]
    defense = oracle["defense"]
    alternatives = oracle["immediate_win_alternatives"]
    illegal = summary["training_totals"]["illegal_violations"]
    tactical = summary["final_tactical"]["all_success"]
    pass_signal = summary["training_totals"]["pass_score_endings"] > 0
    arena_wins = summary["arena"]["v4_wins"]
    strong = all(
        (
            illegal == 0,
            tactical,
            exact["positive_fraction"] <= baseline["exact_loss_q_positive"] - 0.05,
            defense["ranking_failure_fraction"]
            <= baseline["defense_ranking_failure"] - 0.20,
            alternatives["at_least_0_99_fraction"]
            <= baseline["alternative_q_at_least_0_99"] - 0.10,
            pass_signal,
            arena_wins >= 10,
        )
    )
    if strong:
        return "STRONG SUCCESS"
    calibration_not_better = (
        exact["positive_fraction"] >= baseline["exact_loss_q_positive"]
        and defense["ranking_failure_fraction"]
        >= baseline["defense_ranking_failure"]
    )
    if illegal or not tactical or (calibration_not_better and arena_wins <= 7):
        return "FAIL"
    return "PARTIAL SUCCESS"
