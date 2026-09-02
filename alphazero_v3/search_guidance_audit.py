"""Diagnostic-only separation of policy and value guidance in V3 MCTS."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from alphazero_v2.config import AlphaZeroConfig
from alphazero_v2.evaluate import (
    analyze_safe_defense_actions,
    capture_tactical_position,
    defense_tactical_position,
    winning_pass_tactical_position,
)
from alphazero_v2.mcts import MCTS, masked_policy, terminal_value
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BOARD_SIZE,
    NUM_ACTIONS,
    PASS_ACTION,
    MoveResultV2,
)


@dataclass(frozen=True)
class SearchGuidanceMode:
    name: str
    learned_policy: bool
    learned_value: bool


SEARCH_GUIDANCE_MODES = (
    SearchGuidanceMode("learned_policy_learned_value", True, True),
    SearchGuidanceMode("uniform_policy_learned_value", False, True),
    SearchGuidanceMode("learned_policy_zero_value", True, False),
    SearchGuidanceMode("uniform_policy_zero_value", False, False),
)


def uniform_legal_policy(legal_mask):
    """Return a normalized uniform distribution over Rules V2 legal actions."""
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if legal_mask.shape != (NUM_ACTIONS,):
        raise ValueError("legal mask must have shape (82,)")
    legal_count = int(np.count_nonzero(legal_mask))
    if legal_count == 0:
        raise RuntimeError("active Rules V2 state has no legal action")
    priors = np.zeros(NUM_ACTIONS, dtype=np.float32)
    priors[legal_mask] = 1.0 / legal_count
    return priors


def value_from_player_to_root(value, value_player, root_player):
    """Normalize a player-perspective value to the root player's perspective."""
    return float(value) if int(value_player) == int(root_player) else -float(value)


class GuidanceAuditMCTS(MCTS):
    """MCTS variant for ablation only; production ``MCTS`` is unchanged."""

    def __init__(self, network, config, device, *, state_encoder, mode):
        super().__init__(network, config, device, state_encoder=state_encoder)
        if mode not in SEARCH_GUIDANCE_MODES:
            raise ValueError("unknown search guidance mode")
        self.mode = mode

    def _expand_and_evaluate(self, node, logic):
        if node.to_play != logic.turn:
            raise RuntimeError("MCTS node/player desynchronized from Rules V2 state")
        logits, learned_value = self._network_evaluate(logic)
        legal_mask = action_mask_for_logic(logic, node.to_play)
        priors = (
            masked_policy(logits, legal_mask)
            if self.mode.learned_policy
            else uniform_legal_policy(legal_mask)
        )
        self._expand(node, logic, priors)
        return learned_value if self.mode.learned_value else 0.0


def policy_statistics(logits, legal_mask, expected_action):
    """Describe the learned policy over legal actions with deterministic ranks."""
    logits = np.asarray(logits, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    expected_action = int(expected_action)
    if logits.shape != (NUM_ACTIONS,) or legal_mask.shape != (NUM_ACTIONS,):
        raise ValueError("policy logits and legal mask must both have shape (82,)")
    if not 0 <= expected_action < NUM_ACTIONS or not legal_mask[expected_action]:
        raise ValueError("expected action must be legal")
    probabilities = masked_policy(logits, legal_mask).astype(np.float64)
    legal_actions = [int(action) for action in np.flatnonzero(legal_mask)]
    ranked = sorted(legal_actions, key=lambda action: (-probabilities[action], action))
    entropy = -sum(
        probabilities[action] * math.log(probabilities[action])
        for action in legal_actions
        if probabilities[action] > 0.0
    )
    return {
        "expected_action_raw_logit": float(logits[expected_action]),
        "expected_action_legal_probability": float(probabilities[expected_action]),
        "expected_action_legal_rank": ranked.index(expected_action) + 1,
        "legal_policy_entropy": float(entropy),
        "legal_action_count": len(legal_actions),
    }


def _network_output(checkpoint, logic):
    encoded = (
        torch.from_numpy(checkpoint.state_encoder(logic))
        .unsqueeze(0)
        .to(checkpoint.device)
    )
    was_training = checkpoint.network.training
    checkpoint.network.eval()
    with torch.no_grad():
        logits, value = checkpoint.network(encoded)
    if was_training:
        checkpoint.network.train()
    return logits[0].detach().cpu().numpy(), float(value[0].item())


def _search_config(checkpoint, simulations, c_puct=None):
    if c_puct is None:
        c_puct = checkpoint.config.get("c_puct", 1.5)
    if float(c_puct) <= 0.0:
        raise ValueError("c_puct must be positive")
    return AlphaZeroConfig(
        channels=int(checkpoint.config["channels"]),
        residual_blocks=int(checkpoint.config["residual_blocks"]),
        mcts_simulations=int(simulations),
        c_puct=float(c_puct),
        dirichlet_fraction=0.0,
        temperature=0.0,
    )


def _child_record(root, logic, action, total_visits):
    child = root.children[int(action)]
    q_value = value_from_player_to_root(
        child.value(), child.to_play, root.to_play
    )
    simulated = logic.copy()
    result = simulated.apply_action(int(action))
    return {
        "action": int(action),
        "coordinate": (
            None
            if int(action) == PASS_ACTION
            else [int(action) % BOARD_SIZE, int(action) // BOARD_SIZE]
        ),
        "is_pass": int(action) == PASS_ACTION,
        "immediate_result": result.name,
        "immediate_winner": simulated.winner,
        "visit_count": int(child.visit_count),
        "visit_fraction": (
            float(child.visit_count / total_visits) if total_visits else 0.0
        ),
        "prior": float(child.prior),
        "q_value_root_player": q_value,
    }


def _expected_child_value(checkpoint, logic, expected_action):
    root_player = logic.turn
    child_logic = logic.copy()
    result = child_logic.apply_action(int(expected_action))
    if result not in (
        MoveResultV2.NORMAL,
        MoveResultV2.CAPTURE_WIN,
        MoveResultV2.PASS,
        MoveResultV2.PASS_SCORE_END,
    ):
        raise RuntimeError(f"expected action became illegal: {result.name}")
    record = {
        "move_result": result.name,
        "terminal": bool(child_logic.game_over),
        "child_to_play": int(child_logic.turn),
        "winner": child_logic.winner,
        "network_value_child_player": None,
        "network_value_root_player": None,
        "exact_terminal_value_root_player": None,
    }
    if child_logic.game_over:
        record["exact_terminal_value_root_player"] = terminal_value(
            child_logic, root_player
        )
    else:
        _, child_value = _network_output(checkpoint, child_logic)
        record["network_value_child_player"] = child_value
        record["network_value_root_player"] = value_from_player_to_root(
            child_value, child_logic.turn, root_player
        )
    return record


def analyze_guidance_mode(
    checkpoint,
    logic,
    expected_action,
    mode,
    *,
    simulations=256,
    c_puct=None,
    safe_actions=None,
    top_k=10,
):
    """Run one fixture/mode and return policy, value, and root diagnostics."""
    if int(simulations) <= 0:
        raise ValueError("simulations must be positive")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    expected_action = int(expected_action)
    legal_mask = action_mask_for_logic(logic, logic.turn)
    logits, root_network_value = _network_output(checkpoint, logic)
    policy = policy_statistics(logits, legal_mask, expected_action)
    search = GuidanceAuditMCTS(
        checkpoint.network,
        _search_config(checkpoint, simulations, c_puct),
        checkpoint.device,
        state_encoder=checkpoint.state_encoder,
        mode=mode,
    )
    root = search.run(logic, add_root_noise=False)
    total_visits = sum(child.visit_count for child in root.children.values())
    records = [
        _child_record(root, logic, action, total_visits)
        for action in root.children
    ]
    records.sort(key=lambda item: (-item["visit_count"], item["action"]))
    visited_records = [item for item in records if item["visit_count"] > 0]
    visit_probabilities = [
        item["visit_fraction"] for item in visited_records
    ]
    visit_entropy = -sum(
        probability * math.log(probability)
        for probability in visit_probabilities
        if probability > 0.0
    )
    selected = records[0]
    by_action = {item["action"]: item for item in records}
    expected = by_action[expected_action]
    if safe_actions is None:
        success = selected["action"] == expected_action
    else:
        success = selected["action"] in set(int(action) for action in safe_actions)
    return {
        "checkpoint_iteration": int(checkpoint.iteration),
        "mode": mode.name,
        "learned_policy": mode.learned_policy,
        "learned_value": mode.learned_value,
        "simulations": int(simulations),
        "c_puct": float(
            checkpoint.config.get("c_puct", 1.5)
            if c_puct is None
            else c_puct
        ),
        "root_player": int(root.to_play),
        "selected_action": selected["action"],
        "expected_action": expected_action,
        "safe_actions": (
            None if safe_actions is None else [int(action) for action in safe_actions]
        ),
        "success": bool(success),
        "selected_action_stats": selected,
        "expected_action_stats": expected,
        "policy_diagnostics": policy,
        "root_network_value": root_network_value,
        "root_search_leaf_value": (
            root_network_value if mode.learned_value else 0.0
        ),
        "expected_child_value": _expected_child_value(
            checkpoint, logic, expected_action
        ),
        "total_child_visits": int(total_visits),
        "legal_child_count": len(records),
        "visited_child_count": len(visited_records),
        "visited_child_fraction": len(visited_records) / len(records),
        "visit_entropy": float(visit_entropy),
        "top_actions": records[: int(top_k)],
    }


def fixture_definitions():
    defense = defense_tactical_position()
    defense_safety = analyze_safe_defense_actions(defense)
    capture_action = 1 + 2 * BOARD_SIZE
    winning_pass = winning_pass_tactical_position()
    original_player = winning_pass.turn
    terminal = winning_pass.copy()
    result = terminal.apply_action(PASS_ACTION)
    if result != MoveResultV2.PASS_SCORE_END or terminal.winner != original_player:
        raise RuntimeError("winning PASS fixture is not an immediate current-player win")
    return (
        {
            "name": "immediate_capture",
            "builder": capture_tactical_position,
            "expected_action": capture_action,
            "safe_actions": None,
            "fixture_metadata": {},
        },
        {
            "name": "immediate_capture_threat_defense",
            "builder": defense_tactical_position,
            "expected_action": capture_action,
            "safe_actions": defense_safety["safe_defense_actions"],
            "fixture_metadata": defense_safety,
        },
        {
            "name": "immediate_winning_pass",
            "builder": winning_pass_tactical_position,
            "expected_action": PASS_ACTION,
            "safe_actions": None,
            "fixture_metadata": {
                "terminal_result": result.name,
                "winner": terminal.winner,
                "original_player": original_player,
            },
        },
    )


def audit_checkpoint(checkpoint, simulations=256):
    records = []
    fixtures = {}
    for fixture in fixture_definitions():
        fixtures[fixture["name"]] = fixture["fixture_metadata"]
        for mode in SEARCH_GUIDANCE_MODES:
            record = analyze_guidance_mode(
                checkpoint,
                fixture["builder"](),
                fixture["expected_action"],
                mode,
                simulations=simulations,
                safe_actions=fixture["safe_actions"],
            )
            record["fixture"] = fixture["name"]
            records.append(record)
    return {"fixtures": fixtures, "records": records}


def classify_mode_outcomes(records, fixture):
    """Classify a failing production search using the requested audit rules."""
    outcomes = {
        record["mode"]: bool(record["success"])
        for record in records
        if record["fixture"] == fixture
    }
    expected_modes = {mode.name for mode in SEARCH_GUIDANCE_MODES}
    if set(outcomes) != expected_modes:
        raise ValueError("fixture does not contain all four search modes")
    current = outcomes["learned_policy_learned_value"]
    uniform_policy = outcomes["uniform_policy_learned_value"]
    zero_value = outcomes["learned_policy_zero_value"]
    uniform_zero = outcomes["uniform_policy_zero_value"]
    if current:
        return "BASELINE_PASS"
    if uniform_policy and zero_value:
        return "MIXED"
    if uniform_policy:
        return "POLICY"
    if zero_value:
        return "VALUE"
    if uniform_zero:
        return "BOTH"
    return "SEARCH"
