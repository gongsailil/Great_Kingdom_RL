"""Deterministic checkpoint evaluation on Great Kingdom Rules V2."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    CASTLES_PER_PLAYER,
    LEGAL_PLACEMENT_RESULTS,
    NEUTRAL,
    NUM_ACTIONS,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
)

from .config import AlphaZeroConfig
from .mcts import MCTS, visit_count_policy
from .network import PolicyValueNetwork


@dataclass
class EvaluationCheckpoint:
    path: Path
    iteration: int
    config: dict
    network: PolicyValueNetwork
    device: torch.device


def choose_evaluation_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def load_evaluation_checkpoint(path, device="auto", expected_iteration=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation checkpoint does not exist: {path}")
    device = choose_evaluation_device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    for key in ("config", "network_state_dict", "iteration"):
        if key not in payload:
            raise ValueError(f"checkpoint is missing required field: {key}")
    config = dict(payload["config"])
    if "channels" not in config or "residual_blocks" not in config:
        raise ValueError("checkpoint config lacks network architecture")
    iteration = int(payload["iteration"])
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise ValueError(
            f"checkpoint iteration {iteration} != expected {expected_iteration}"
        )
    filename_match = re.fullmatch(r"iteration_(\d+)\.pt", path.name)
    if filename_match and iteration != int(filename_match.group(1)):
        raise ValueError("checkpoint filename and payload iteration disagree")

    network = PolicyValueNetwork(
        channels=int(config["channels"]),
        residual_blocks=int(config["residual_blocks"]),
    ).to(device)
    network.load_state_dict(payload["network_state_dict"])
    network.eval()
    return EvaluationCheckpoint(path, iteration, config, network, device)


def _mcts_config(checkpoint, simulations):
    return AlphaZeroConfig(
        channels=int(checkpoint.config["channels"]),
        residual_blocks=int(checkpoint.config["residual_blocks"]),
        mcts_simulations=int(simulations),
        c_puct=float(checkpoint.config.get("c_puct", 1.5)),
        dirichlet_fraction=0.0,
        temperature=0.0,
    )


def evaluation_search_root(checkpoint, logic, simulations=64):
    if simulations <= 0:
        raise ValueError("MCTS simulations must be positive")
    search = MCTS(
        checkpoint.network,
        _mcts_config(checkpoint, simulations),
        checkpoint.device,
    )
    return search.run(logic, add_root_noise=False)


def select_evaluation_action(checkpoint, logic, simulations=64):
    """Choose maximum-visit action with no noise and no fallback."""
    root = evaluation_search_root(checkpoint, logic, simulations)
    policy = visit_count_policy(root, temperature=0.0)
    action = int(np.argmax(policy))
    legal_mask = action_mask_for_logic(logic, logic.turn)
    if not 0 <= action < NUM_ACTIONS or not bool(legal_mask[action]):
        raise RuntimeError(f"evaluation MCTS selected illegal action {action}")
    if action not in root.children:
        raise RuntimeError(f"evaluation MCTS selected missing child {action}")
    return action


def apply_opening(actions):
    logic = GreatKingdomLogicV2()
    for ply, raw_action in enumerate(actions):
        action = int(raw_action)
        if not 0 <= action < PASS_ACTION:
            raise ValueError(f"opening ply {ply} is not a placement: {action}")
        player = logic.turn
        mask = action_mask_for_logic(logic, player)
        if not bool(mask[action]):
            raise ValueError(f"opening ply {ply} is illegal: {action}")
        result = logic.apply_action(action)
        if result != MoveResultV2.NORMAL:
            raise ValueError(
                f"opening ply {ply} must be non-terminal NORMAL, got {result.name}"
            )
    placed_colors = {
        state
        for row in logic.board
        for state in row
        if state in (BLUE, RED)
    }
    if placed_colors != {BLUE, RED}:
        raise ValueError("opening must contain a placement by both players")
    return logic


def generate_opening_suite(count=10, seed=20260901):
    if count <= 0:
        raise ValueError("opening count must be positive")
    rng = np.random.default_rng(seed)
    openings = []
    seen = set()
    attempts = 0
    while len(openings) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not generate unique legal opening suite")
        logic = GreatKingdomLogicV2()
        length = int(rng.integers(2, 5))
        actions = []
        for _ in range(length):
            legal = []
            mask = action_mask_for_logic(logic, logic.turn)
            for action in np.flatnonzero(mask[:PASS_ACTION]):
                action = int(action)
                x, y = action % BOARD_SIZE, action // BOARD_SIZE
                if logic.classify_placement(logic.turn, x, y) == MoveResultV2.NORMAL:
                    legal.append(action)
            if not legal:
                break
            action = int(rng.choice(legal))
            if logic.apply_action(action) != MoveResultV2.NORMAL:
                raise RuntimeError("generated opening unexpectedly became terminal")
            actions.append(action)
        key = tuple(actions)
        if len(actions) != length or key in seen:
            continue
        checked = apply_opening(actions)
        seen.add(key)
        openings.append(
            {
                "opening_id": len(openings),
                "actions": actions,
                "resulting_turn": checked.turn,
            }
        )
    return {"seed": int(seed), "count": int(count), "openings": openings}


def validate_opening_suite(payload):
    openings = payload.get("openings", [])
    if int(payload.get("count", -1)) != len(openings):
        raise ValueError("opening suite count is inconsistent")
    identifiers = set()
    for opening in openings:
        identifier = int(opening["opening_id"])
        if identifier in identifiers:
            raise ValueError("duplicate opening_id")
        identifiers.add(identifier)
        actions = list(opening["actions"])
        if not 2 <= len(actions) <= 4:
            raise ValueError("opening length must be 2 to 4 placements")
        logic = apply_opening(actions)
        if logic.turn != int(opening["resulting_turn"]):
            raise ValueError("opening resulting_turn is incorrect")
    return True


def play_arena_game(
    blue_checkpoint,
    red_checkpoint,
    opening,
    simulations=64,
    max_moves=200,
):
    logic = apply_opening(opening["actions"])
    evaluation_actions = []
    pass_usage = 0
    last_result = None
    for _ in range(max_moves):
        checkpoint = blue_checkpoint if logic.turn == BLUE else red_checkpoint
        action = select_evaluation_action(checkpoint, logic, simulations)
        legal_mask = action_mask_for_logic(logic, logic.turn)
        if not bool(legal_mask[action]):
            raise RuntimeError(f"arena selected illegal action {action}")
        last_result = logic.apply_action(action)
        if last_result not in (
            MoveResultV2.NORMAL,
            MoveResultV2.CAPTURE_WIN,
            MoveResultV2.PASS,
            MoveResultV2.PASS_SCORE_END,
        ):
            raise RuntimeError(f"arena action became illegal: {last_result.name}")
        evaluation_actions.append(action)
        if action == PASS_ACTION:
            pass_usage += 1
        if logic.game_over:
            break
    else:
        raise RuntimeError("arena game exceeded maximum move count")

    winner_checkpoint = (
        blue_checkpoint if logic.winner == BLUE else red_checkpoint
    )
    return {
        "opening_id": int(opening["opening_id"]),
        "opening_actions": list(opening["actions"]),
        "blue_iteration": blue_checkpoint.iteration,
        "red_iteration": red_checkpoint.iteration,
        "winner_color": logic.winner,
        "winner_iteration": winner_checkpoint.iteration,
        "terminal_reason": last_result.name,
        "game_length": len(opening["actions"]) + len(evaluation_actions),
        "evaluation_moves": len(evaluation_actions),
        "pass_usage": pass_usage,
        "score_blue": logic.score_blue,
        "score_red": logic.score_red,
        "actions": evaluation_actions,
        "mcts_simulations": int(simulations),
    }


def play_paired_opening(
    checkpoint_a,
    checkpoint_b,
    opening,
    simulations=64,
):
    return [
        play_arena_game(
            checkpoint_a,
            checkpoint_b,
            opening,
            simulations=simulations,
        ),
        play_arena_game(
            checkpoint_b,
            checkpoint_a,
            opening,
            simulations=simulations,
        ),
    ]


def _empty_tactical_position():
    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    return logic


def _sync_inventory(logic):
    for player in (BLUE, RED):
        used = sum(state == player for row in logic.board for state in row)
        logic.castles_remaining[player] = CASTLES_PER_PLAYER - used
    return logic


def capture_tactical_position():
    logic = _empty_tactical_position()
    logic.board[1][1] = RED
    for x, y in ((0, 1), (1, 0), (2, 1)):
        logic.board[y][x] = BLUE
    return _sync_inventory(logic)


def defense_tactical_position():
    logic = _empty_tactical_position()
    logic.board[1][1] = BLUE
    for x, y in ((0, 1), (1, 0), (2, 1)):
        logic.board[y][x] = RED
    return _sync_inventory(logic)


def territory_tactical_position():
    logic = _empty_tactical_position()
    for x, y in ((2, 0), (2, 1), (2, 2), (0, 2), (1, 2)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED
    return _sync_inventory(logic)


def suicide_tactical_position():
    logic = _empty_tactical_position()
    for x, y in ((1, 0), (0, 1), (2, 1), (0, 2), (2, 2), (1, 3)):
        logic.board[y][x] = RED
    logic.board[2][1] = BLUE
    return _sync_inventory(logic)


def winning_pass_tactical_position():
    logic = _empty_tactical_position()
    for x, y in ((0, 1), (1, 1), (2, 0)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED
    logic.consecutive_passes = 1
    return _sync_inventory(logic)


def run_tactical_sanity(checkpoint, simulations=64):
    capture = capture_tactical_position()
    capture_action = 1 + 2 * BOARD_SIZE
    selected_capture = select_evaluation_action(checkpoint, capture, simulations)

    defense = defense_tactical_position()
    defense_action = 1 + 2 * BOARD_SIZE
    selected_defense = select_evaluation_action(checkpoint, defense, simulations)

    territory = territory_tactical_position()
    own_action = 0
    blue_root = evaluation_search_root(checkpoint, territory, simulations)
    territory.turn = RED
    red_root = evaluation_search_root(checkpoint, territory, simulations)

    suicide = suicide_tactical_position()
    suicide_action = 1 + BOARD_SIZE
    suicide_root = evaluation_search_root(checkpoint, suicide, simulations)

    winning_pass = winning_pass_tactical_position()
    assert winning_pass.copy().apply_action(PASS_ACTION) == (
        MoveResultV2.PASS_SCORE_END
    )
    selected_pass = select_evaluation_action(checkpoint, winning_pass, simulations)
    return {
        "checkpoint_iteration": checkpoint.iteration,
        "mcts_simulations": int(simulations),
        "immediate_capture": {
            "expected_action": capture_action,
            "selected_action": selected_capture,
            "pass": selected_capture == capture_action,
        },
        "defense_threat": {
            "defense_action": defense_action,
            "selected_action": selected_defense,
            "selected_defense": selected_defense == defense_action,
            "diagnostic_only": True,
        },
        "own_territory": {
            "action": own_action,
            "legal_child": own_action in blue_root.children,
        },
        "opponent_territory": {
            "action": own_action,
            "excluded": own_action not in red_root.children,
        },
        "pure_suicide": {
            "action": suicide_action,
            "excluded": suicide_action not in suicide_root.children,
        },
        "winning_score_pass": {
            "expected_action": PASS_ACTION,
            "selected_action": selected_pass,
            "pass": selected_pass == PASS_ACTION,
        },
    }
