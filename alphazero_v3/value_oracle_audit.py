"""Rules-exact one-ply value diagnostics for fixed V3 replay states."""

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    CASTLES_PER_PLAYER,
    NEUTRAL,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
)

from .encoder import ENCODED_SHAPE, encode_state
from .temperature_audit import network_state_digest


EXACT_WIN = "IMMEDIATE_WIN"
EXACT_LOSS = "ALLOWS_IMMEDIATE_LOSS"
UNRESOLVED = "UNRESOLVED"
VALUE_BUCKET_EDGES = (-1.0, -0.8, -0.4, 0.0, 0.4, 0.8, 0.95, 0.99, 1.000001)
VALUE_BUCKET_LABELS = (
    "[-1,-0.8)",
    "[-0.8,-0.4)",
    "[-0.4,0)",
    "[0,0.4)",
    "[0.4,0.8)",
    "[0.8,0.95)",
    "[0.95,0.99)",
    "[0.99,1]",
)


@dataclass
class ChildAction:
    action: int
    action_type: str
    classification: str
    opponent_winning_actions: list
    encoded_child: np.ndarray | None


@dataclass
class OracleState:
    replay_index: int
    player: int
    immediate_win_actions: list
    terminal_loss_actions: list
    original_opponent_threat_actions: list
    children: list

    @property
    def exact_loss_children(self):
        return [
            child for child in self.children if child.classification == EXACT_LOSS
        ]

    @property
    def safe_nonterminal_children(self):
        return [
            child
            for child in self.children
            if child.classification == UNRESOLVED
        ]

    @property
    def defense_opportunity(self):
        safe_actions = self.immediate_win_actions + [
            child.action for child in self.safe_nonterminal_children
        ]
        return bool(
            self.original_opponent_threat_actions
            and safe_actions
            and self.exact_loss_children
        )


def root_q_from_child_value(child_value):
    """A nonterminal child is opponent-to-play, so its root Q flips sign."""
    return -float(child_value)


def _constant_plane_value(plane, name, *, atol):
    value = float(plane[0, 0])
    if not np.allclose(plane, value, rtol=0.0, atol=atol):
        raise ValueError(f"{name} plane is not constant")
    return value


def decode_v3_state(state, *, atol=1e-6):
    """Restore an active Rules V2 state from the canonical nine-plane replay."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != ENCODED_SHAPE:
        raise ValueError(f"V3 replay state must have shape {ENCODED_SHAPE}")
    if not np.all(np.isfinite(state)):
        raise ValueError("V3 replay state contains non-finite values")

    absolute_color = _constant_plane_value(
        state[6], "absolute-color", atol=atol
    )
    if np.isclose(absolute_color, 1.0, rtol=0.0, atol=atol):
        current = BLUE
    elif np.isclose(absolute_color, 0.0, rtol=0.0, atol=atol):
        current = RED
    else:
        raise ValueError("absolute-color plane is neither Blue nor Red")
    opponent = 3 - current

    occupancy = state[:3]
    rounded = np.rint(occupancy)
    if not np.allclose(occupancy, rounded, rtol=0.0, atol=atol):
        raise ValueError("occupancy planes are not binary")
    if np.any(np.sum(rounded, axis=0) > 1.0):
        raise ValueError("occupancy planes overlap")

    passes_value = _constant_plane_value(
        state[3], "consecutive-passes", atol=atol
    )
    consecutive_passes = int(round(passes_value * 2.0))
    if consecutive_passes not in (0, 1) or not np.isclose(
        passes_value,
        consecutive_passes / 2.0,
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError("invalid active consecutive-passes plane")

    current_inventory_value = _constant_plane_value(
        state[4], "current-inventory", atol=atol
    )
    opponent_inventory_value = _constant_plane_value(
        state[5], "opponent-inventory", atol=atol
    )

    def inventory(value, name):
        remaining = int(round(value * CASTLES_PER_PLAYER))
        if not 0 <= remaining <= CASTLES_PER_PLAYER or not np.isclose(
            value,
            remaining / float(CASTLES_PER_PLAYER),
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(f"invalid {name} plane")
        return remaining

    current_remaining = inventory(current_inventory_value, "current-inventory")
    opponent_remaining = inventory(
        opponent_inventory_value, "opponent-inventory"
    )

    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if rounded[0, y, x]:
                logic.board[y][x] = current
            elif rounded[1, y, x]:
                logic.board[y][x] = opponent
            elif rounded[2, y, x]:
                logic.board[y][x] = NEUTRAL
    logic.turn = current
    logic.consecutive_passes = consecutive_passes
    logic.castles_remaining = {
        current: current_remaining,
        opponent: opponent_remaining,
    }
    logic.game_over = False
    logic.winner = None
    logic.win_reason = ""
    logic.last_move_result = None
    logic.score_blue = None
    logic.score_red = None

    neutral_count = sum(
        cell == NEUTRAL for row in logic.board for cell in row
    )
    if neutral_count != 1 or logic.board[BOARD_SIZE // 2][BOARD_SIZE // 2] != NEUTRAL:
        raise ValueError("decoded board does not contain the central neutral castle")
    for player in (BLUE, RED):
        stones = sum(cell == player for row in logic.board for cell in row)
        if stones + logic.castles_remaining[player] != CASTLES_PER_PLAYER:
            raise ValueError("decoded inventory and board castle count disagree")

    roundtrip = encode_state(logic)
    if not np.allclose(roundtrip, state, rtol=0.0, atol=atol):
        difference = float(np.max(np.abs(roundtrip - state)))
        raise ValueError(f"V3 decode/encode mismatch (max difference {difference})")
    return logic


def select_audit_indices(total_states, maximum=10_000, seed=20260902):
    total_states = int(total_states)
    maximum = int(maximum)
    if total_states <= 0 or maximum <= 0:
        raise ValueError("state counts must be positive")
    if total_states <= maximum:
        return np.arange(total_states, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(
        rng.choice(total_states, size=maximum, replace=False).astype(np.int64)
    )


def _capture_candidates(logic, player):
    """Find last-liberty candidates, then let Rules V2 classify each one."""
    opponent = 3 - int(player)
    visited = set()
    candidates = set()
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if logic.board[y][x] != opponent or (x, y) in visited:
                continue
            group = logic.get_group(x, y)
            visited.update(group)
            liberties = set()
            for gx, gy in group:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = gx + dx, gy + dy
                    if logic.is_on_board(nx, ny) and logic.board[ny][nx] == 0:
                        liberties.add((nx, ny))
            if len(liberties) == 1:
                lx, ly = next(iter(liberties))
                candidates.add(lx + ly * BOARD_SIZE)
    return candidates


def immediate_winning_actions(logic, player=None):
    """Return exact one-action wins, with final legality decided by Rules V2."""
    if logic.game_over:
        return []
    if player is None:
        player = logic.turn
    player = int(player)
    wins = []
    for action in sorted(_capture_candidates(logic, player)):
        result = logic.classify_placement(
            player, action % BOARD_SIZE, action // BOARD_SIZE
        )
        if result == MoveResultV2.CAPTURE_WIN:
            wins.append(action)

    pass_state = logic.copy()
    pass_state.turn = player
    result = pass_state.apply_action(PASS_ACTION)
    if (
        result == MoveResultV2.PASS_SCORE_END
        and pass_state.winner == player
    ):
        wins.append(PASS_ACTION)
    return wins


def classify_oracle_state(logic, replay_index=-1):
    """Classify legal root actions without assigning labels to long outcomes."""
    if logic.game_over:
        raise ValueError("oracle classification requires an active state")
    player = logic.turn
    opponent = 3 - player
    original_threats = immediate_winning_actions(logic, opponent)
    legal_mask = action_mask_for_logic(logic, player)
    immediate_wins = []
    terminal_losses = []
    children = []
    for raw_action in np.flatnonzero(legal_mask):
        action = int(raw_action)
        child = logic.copy()
        result = child.apply_action(action)
        if result not in (
            MoveResultV2.NORMAL,
            MoveResultV2.CAPTURE_WIN,
            MoveResultV2.PASS,
            MoveResultV2.PASS_SCORE_END,
        ):
            raise RuntimeError(f"legal action {action} became {result.name}")
        if child.game_over:
            if child.winner == player:
                immediate_wins.append(action)
            else:
                terminal_losses.append(action)
            continue
        if child.turn != opponent:
            raise RuntimeError("nonterminal child did not switch player")
        opponent_wins = immediate_winning_actions(child, opponent)
        classification = EXACT_LOSS if opponent_wins else UNRESOLVED
        children.append(
            ChildAction(
                action=action,
                action_type="pass" if action == PASS_ACTION else "placement",
                classification=classification,
                opponent_winning_actions=opponent_wins,
                encoded_child=encode_state(child),
            )
        )
    return OracleState(
        replay_index=int(replay_index),
        player=int(player),
        immediate_win_actions=immediate_wins,
        terminal_loss_actions=terminal_losses,
        original_opponent_threat_actions=original_threats,
        children=children,
    )


def predict_values(checkpoint, encoded_states, batch_size=4096):
    encoded_states = np.asarray(encoded_states, dtype=np.float32)
    if encoded_states.ndim != 4 or encoded_states.shape[1:] != ENCODED_SHAPE:
        raise ValueError("child batch must have shape (N,9,9,9)")
    outputs = []
    was_training = checkpoint.network.training
    checkpoint.network.eval()
    with torch.no_grad():
        for start in range(0, len(encoded_states), int(batch_size)):
            inputs = torch.from_numpy(encoded_states[start : start + batch_size]).to(
                checkpoint.device
            )
            _, values = checkpoint.network(inputs)
            outputs.append(values.detach().cpu().numpy())
    if was_training:
        checkpoint.network.train()
    if not outputs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32, copy=False)


def distribution_statistics(values, target=None):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "mae": None,
            "mse": None,
            "positive_count": 0,
            "positive_fraction": None,
            "at_least_0_5_count": 0,
            "at_least_0_5_fraction": None,
        }
    result = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "positive_count": int(np.sum(values > 0.0)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "at_least_0_5_count": int(np.sum(values >= 0.5)),
        "at_least_0_5_fraction": float(np.mean(values >= 0.5)),
    }
    if target is None:
        result["mae"] = None
        result["mse"] = None
    else:
        errors = values - float(target)
        result["mae"] = float(np.mean(np.abs(errors)))
        result["mse"] = float(np.mean(np.square(errors)))
    return result


def value_histogram(values):
    values = np.asarray(values, dtype=np.float64)
    counts, _ = np.histogram(values, bins=VALUE_BUCKET_EDGES)
    total = int(values.size)
    return {
        "count": total,
        "buckets": {
            label: {
                "count": int(count),
                "fraction": (float(count / total) if total else None),
            }
            for label, count in zip(VALUE_BUCKET_LABELS, counts)
        },
    }


class NetworkAuditAccumulator:
    def __init__(self, iteration):
        self.iteration = int(iteration)
        self.exact_loss = []
        self.exact_loss_by_player = {BLUE: [], RED: []}
        self.exact_loss_by_action_type = {"placement": [], "pass": []}
        self.all_root_q = []
        self.all_raw_child_value = []
        self.all_root_q_by_player = {BLUE: [], RED: []}
        self.immediate_win_state_count = 0
        self.alternative_q = []
        self.alternative_state_maxima = []
        self.defense_opportunity_states = 0
        self.defense_rankable_states = 0
        self.defense_ranking_failures = 0
        self.defense_ranking_margins = []
        self.worst_exact_losses = []

    def add_state(self, oracle_state, raw_child_values):
        if len(raw_child_values) != len(oracle_state.children):
            raise ValueError("network child values and oracle actions disagree")
        root_qs = [root_q_from_child_value(value) for value in raw_child_values]
        child_q = {
            child.action: q for child, q in zip(oracle_state.children, root_qs)
        }
        self.all_raw_child_value.extend(float(value) for value in raw_child_values)
        self.all_root_q.extend(root_qs)
        self.all_root_q_by_player[oracle_state.player].extend(root_qs)

        for child, q_value in zip(oracle_state.children, root_qs):
            if child.classification != EXACT_LOSS:
                continue
            self.exact_loss.append(q_value)
            self.exact_loss_by_player[oracle_state.player].append(q_value)
            self.exact_loss_by_action_type[child.action_type].append(q_value)
            self.worst_exact_losses.append(
                {
                    "replay_index": oracle_state.replay_index,
                    "player": oracle_state.player,
                    "action": child.action,
                    "action_type": child.action_type,
                    "predicted_root_q": q_value,
                    "opponent_winning_actions": child.opponent_winning_actions,
                }
            )

        if oracle_state.immediate_win_actions:
            self.immediate_win_state_count += 1
            alternatives = list(root_qs)
            self.alternative_q.extend(alternatives)
            if alternatives:
                self.alternative_state_maxima.append(max(alternatives))

        if oracle_state.defense_opportunity:
            self.defense_opportunity_states += 1
            safe_qs = [
                child_q[child.action]
                for child in oracle_state.safe_nonterminal_children
            ]
            unsafe_qs = [
                child_q[child.action]
                for child in oracle_state.exact_loss_children
            ]
            if safe_qs and unsafe_qs:
                self.defense_rankable_states += 1
                margin = max(unsafe_qs) - max(safe_qs)
                self.defense_ranking_margins.append(margin)
                if margin > 0.0:
                    self.defense_ranking_failures += 1

    def finalize(self):
        alternative_q = np.asarray(self.alternative_q, dtype=np.float64)
        alternative = {
            "immediate_win_state_count": self.immediate_win_state_count,
            "nonterminal_alternative_count": int(alternative_q.size),
            "mean_alternative_predicted_q": (
                float(np.mean(alternative_q)) if alternative_q.size else None
            ),
            "max_alternative_predicted_q": (
                float(np.max(alternative_q)) if alternative_q.size else None
            ),
        }
        for threshold in (0.9, 0.95, 0.99):
            count = int(np.sum(alternative_q >= threshold))
            key = str(threshold).replace(".", "_")
            alternative[f"at_least_{key}_count"] = count
            alternative[f"at_least_{key}_fraction"] = (
                float(count / alternative_q.size) if alternative_q.size else None
            )
            state_count = sum(
                maximum >= threshold for maximum in self.alternative_state_maxima
            )
            alternative[f"states_with_max_at_least_{key}"] = int(state_count)

        self.worst_exact_losses.sort(
            key=lambda record: -record["predicted_root_q"]
        )
        defense_failure_fraction = (
            self.defense_ranking_failures / self.defense_rankable_states
            if self.defense_rankable_states
            else None
        )
        return {
            "iteration": self.iteration,
            "exact_loss": distribution_statistics(self.exact_loss, target=-1.0),
            "exact_loss_by_root_player": {
                "blue": distribution_statistics(
                    self.exact_loss_by_player[BLUE], target=-1.0
                ),
                "red": distribution_statistics(
                    self.exact_loss_by_player[RED], target=-1.0
                ),
            },
            "exact_loss_by_action_type": {
                name: distribution_statistics(values, target=-1.0)
                for name, values in self.exact_loss_by_action_type.items()
            },
            "defense": {
                "opportunity_states": self.defense_opportunity_states,
                "rankable_states": self.defense_rankable_states,
                "ranking_failures": self.defense_ranking_failures,
                "ranking_failure_fraction": defense_failure_fraction,
                "mean_unsafe_minus_safe_max_q": (
                    float(np.mean(self.defense_ranking_margins))
                    if self.defense_ranking_margins
                    else None
                ),
            },
            "immediate_win_alternatives": alternative,
            "saturation": {
                "predicted_root_q": value_histogram(self.all_root_q),
                "raw_child_current_player_value": value_histogram(
                    self.all_raw_child_value
                ),
                "predicted_root_q_by_root_player": {
                    "blue": value_histogram(self.all_root_q_by_player[BLUE]),
                    "red": value_histogram(self.all_root_q_by_player[RED]),
                },
            },
            "worst_exact_loss_examples": self.worst_exact_losses[:10],
        }


def validate_replay(states, values, players, *, atol=1e-6):
    """Validate every replay state/target and count exact contradictory targets."""
    states = np.asarray(states, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    players = np.asarray(players, dtype=np.int8)
    if states.ndim != 4 or states.shape[1:] != ENCODED_SHAPE:
        raise ValueError("replay states have the wrong V3 shape")
    if not (len(states) == len(values) == len(players)):
        raise ValueError("replay arrays have inconsistent lengths")

    outcomes = {}
    counts = {}
    blue = 0
    red = 0
    for index, (state, value, stored_player) in enumerate(
        zip(states, values, players)
    ):
        logic = decode_v3_state(state, atol=atol)
        if int(stored_player) != logic.turn:
            raise ValueError(
                f"replay sample {index} player {stored_player} != turn {logic.turn}"
            )
        if float(value) not in (-1.0, 1.0):
            raise ValueError(f"replay sample {index} target is not -1/+1")
        blue += logic.turn == BLUE
        red += logic.turn == RED
        digest = hashlib.sha256(np.ascontiguousarray(state).tobytes()).digest()
        bit = 1 if float(value) == -1.0 else 2
        outcomes[digest] = outcomes.get(digest, 0) | bit
        counts[digest] = counts.get(digest, 0) + 1
    contradictory = {digest for digest, mask in outcomes.items() if mask == 3}
    return {
        "total_states": len(states),
        "blue_turn_states": int(blue),
        "red_turn_states": int(red),
        "blue_turn_fraction": float(blue / len(states)),
        "red_turn_fraction": float(red / len(states)),
        "duplicate_unique_state_count": int(
            sum(count > 1 for count in counts.values())
        ),
        "duplicate_extra_sample_count": int(
            sum(count - 1 for count in counts.values() if count > 1)
        ),
        "contradictory_z_unique_state_count": len(contradictory),
        "contradictory_z_sample_count": int(
            sum(counts[digest] for digest in contradictory)
        ),
        "roundtrip_failures": 0,
        "player_mismatches": 0,
        "invalid_targets": 0,
    }


def classify_value_audit(iteration_10, iteration_50):
    ten = iteration_10["exact_loss"]
    fifty = iteration_50["exact_loss"]
    defense_ten = iteration_10["defense"]["ranking_failure_fraction"]
    defense_fifty = iteration_50["defense"]["ranking_failure_fraction"]
    regression_signals = sum(
        (
            fifty["mae"] > ten["mae"] + 0.05,
            fifty["positive_fraction"] > ten["positive_fraction"] + 0.05,
            defense_ten is not None
            and defense_fifty is not None
            and defense_fifty > defense_ten + 0.05,
        )
    )
    blue = iteration_50["exact_loss_by_root_player"]["blue"]
    red = iteration_50["exact_loss_by_root_player"]["red"]
    color_gap = max(
        abs(blue["mae"] - red["mae"]),
        abs(blue["positive_fraction"] - red["positive_fraction"]),
    )
    value_bad = (
        fifty["positive_fraction"] >= 0.25
        or (
            defense_fifty is not None
            and defense_fifty >= 0.25
        )
    )
    if regression_signals >= 2:
        primary = "CASE_TRAINING_REGRESSION"
    elif color_gap >= 0.20:
        primary = "CASE_COLOR"
    elif value_bad:
        primary = "CASE_VALUE_CALIBRATION"
    else:
        primary = "CASE_IMPROVED"
    return {
        "primary": primary,
        "regression_signal_count": int(regression_signals),
        "color_gap": float(color_gap),
        "value_miscalibration_flag": bool(value_bad),
        "color_asymmetry_flag": bool(color_gap >= 0.20),
    }


def audit_networks_on_states(
    checkpoints,
    states,
    indices,
    chunk_size=128,
    progress_callback=None,
):
    """Use one shared Rules-derived action set for every fixed checkpoint."""
    accumulators = {
        checkpoint.iteration: NetworkAuditAccumulator(checkpoint.iteration)
        for checkpoint in checkpoints
    }
    state_class_counts = {
        "immediate_win_states": 0,
        "defense_opportunity_states": 0,
        "terminal_loss_actions": 0,
        "nonterminal_child_actions": 0,
        "exact_loss_actions": 0,
        "unresolved_actions": 0,
    }
    audited_player_counts = {BLUE: 0, RED: 0}

    indices = np.asarray(indices, dtype=np.int64)
    for chunk_start in range(0, len(indices), int(chunk_size)):
        chunk_indices = indices[chunk_start : chunk_start + int(chunk_size)]
        oracle_states = []
        encoded_children = []
        slices = []
        for replay_index in chunk_indices:
            logic = decode_v3_state(states[int(replay_index)])
            oracle = classify_oracle_state(logic, int(replay_index))
            start = len(encoded_children)
            encoded_children.extend(
                child.encoded_child for child in oracle.children
            )
            slices.append((start, len(encoded_children)))
            oracle_states.append(oracle)
            audited_player_counts[oracle.player] += 1
            state_class_counts["immediate_win_states"] += bool(
                oracle.immediate_win_actions
            )
            state_class_counts["defense_opportunity_states"] += bool(
                oracle.defense_opportunity
            )
            state_class_counts["terminal_loss_actions"] += len(
                oracle.terminal_loss_actions
            )
            state_class_counts["nonterminal_child_actions"] += len(
                oracle.children
            )
            state_class_counts["exact_loss_actions"] += len(
                oracle.exact_loss_children
            )
            state_class_counts["unresolved_actions"] += len(
                oracle.safe_nonterminal_children
            )
        child_batch = np.asarray(encoded_children, dtype=np.float32)
        for checkpoint in checkpoints:
            predictions = predict_values(checkpoint, child_batch)
            accumulator = accumulators[checkpoint.iteration]
            for oracle, (start, end) in zip(oracle_states, slices):
                accumulator.add_state(oracle, predictions[start:end])
        if progress_callback is not None:
            progress_callback(min(chunk_start + len(chunk_indices), len(indices)))

    results = {
        str(iteration): accumulator.finalize()
        for iteration, accumulator in accumulators.items()
    }
    return {
        "state_class_counts": state_class_counts,
        "audited_player_counts": {
            "blue": audited_player_counts[BLUE],
            "red": audited_player_counts[RED],
            "blue_fraction": audited_player_counts[BLUE] / len(indices),
            "red_fraction": audited_player_counts[RED] / len(indices),
        },
        "networks": results,
    }
