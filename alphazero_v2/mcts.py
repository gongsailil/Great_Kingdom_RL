"""Minimal PUCT search using Rules V2 as the only transition authority."""

import math

import numpy as np
import torch

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BOARD_SIZE,
    LEGAL_PLACEMENT_RESULTS,
    NUM_ACTIONS,
    PASS_ACTION,
    MoveResultV2,
)

from .encoder import encode_state


class Node:
    def __init__(self, prior, to_play):
        self.prior = float(prior)
        self.to_play = int(to_play)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def expanded(self):
        return bool(self.children)

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def terminal_value(logic, perspective_player):
    if not logic.game_over or logic.winner not in (1, 2):
        raise ValueError("terminal value requires a terminal Rules V2 winner")
    return 1.0 if logic.winner == perspective_player else -1.0


def backup(search_path, value, value_player):
    """Back up a leaf value while preserving each node's player perspective."""
    for node in reversed(search_path):
        node_value = value if node.to_play == value_player else -value
        node.value_sum += float(node_value)
        node.visit_count += 1


def masked_policy(logits, legal_mask):
    logits = np.asarray(logits, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if logits.shape != (NUM_ACTIONS,) or legal_mask.shape != (NUM_ACTIONS,):
        raise ValueError("policy logits and legal mask must both have shape (82,)")
    if not legal_mask.any():
        raise RuntimeError("active Rules V2 state has no legal action")
    masked = np.full(NUM_ACTIONS, -np.inf, dtype=np.float64)
    masked[legal_mask] = logits[legal_mask]
    maximum = np.max(masked[legal_mask])
    probabilities = np.zeros(NUM_ACTIONS, dtype=np.float64)
    probabilities[legal_mask] = np.exp(masked[legal_mask] - maximum)
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float32)


class MCTS:
    def __init__(self, network, config, device):
        self.network = network
        self.config = config
        self.device = torch.device(device)

    def _network_evaluate(self, logic):
        encoded = torch.from_numpy(encode_state(logic)).unsqueeze(0).to(self.device)
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(encoded)
        if was_training:
            self.network.train()
        return logits[0].detach().cpu().numpy(), float(value[0].item())

    def _expand_and_evaluate(self, node, logic):
        if node.to_play != logic.turn:
            raise RuntimeError("MCTS node/player desynchronized from Rules V2 state")
        logits, value = self._network_evaluate(logic)
        legal_mask = action_mask_for_logic(logic, node.to_play)
        priors = masked_policy(logits, legal_mask)
        self._expand(node, logic, priors)
        return value

    @staticmethod
    def _expand(node, logic, priors):
        for action in np.flatnonzero(priors > 0.0):
            action = int(action)
            if action == PASS_ACTION:
                child_to_play = 3 - node.to_play
            else:
                x = action % BOARD_SIZE
                y = action // BOARD_SIZE
                result = logic.classify_placement(node.to_play, x, y)
                if result not in LEGAL_PLACEMENT_RESULTS:
                    raise RuntimeError("masked policy included an illegal placement")
                child_to_play = (
                    node.to_play
                    if result == MoveResultV2.CAPTURE_WIN
                    else 3 - node.to_play
                )
            node.children[action] = Node(priors[action], child_to_play)

    def _add_root_noise(self, root, rng):
        actions = list(root.children)
        if not actions:
            return
        noise = rng.dirichlet(
            [self.config.dirichlet_alpha] * len(actions)
        )
        fraction = self.config.dirichlet_fraction
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (1.0 - fraction) * child.prior + fraction * sample

    def _select_child(self, parent):
        best_action = None
        best_child = None
        best_score = -float("inf")
        scale = math.sqrt(parent.visit_count + 1.0)
        for action, child in parent.children.items():
            child_value = child.value()
            q_value = (
                child_value
                if child.to_play == parent.to_play
                else -child_value
            )
            exploration = (
                self.config.c_puct
                * child.prior
                * scale
                / (1 + child.visit_count)
            )
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def run(self, logic, *, add_root_noise=False, rng=None):
        if logic.game_over:
            raise ValueError("cannot run MCTS from a terminal state")
        if rng is None:
            rng = np.random.default_rng()

        root = Node(prior=1.0, to_play=logic.turn)
        self._expand_and_evaluate(root, logic)
        if add_root_noise:
            self._add_root_noise(root, rng)

        for _ in range(self.config.mcts_simulations):
            simulation = logic.copy()
            node = root
            search_path = [root]

            while node.expanded() and not simulation.game_over:
                action, node = self._select_child(node)
                result = simulation.apply_action(action)
                if result not in (
                    MoveResultV2.NORMAL,
                    MoveResultV2.CAPTURE_WIN,
                    MoveResultV2.PASS,
                    MoveResultV2.PASS_SCORE_END,
                ):
                    raise RuntimeError(
                        f"MCTS selected illegal Rules V2 action: {result.name}"
                    )
                search_path.append(node)

            if simulation.game_over:
                value_player = simulation.turn
                value = terminal_value(simulation, value_player)
            else:
                value_player = simulation.turn
                value = self._expand_and_evaluate(node, simulation)
            backup(search_path, value, value_player)
        return root


def visit_count_policy(root, temperature=1.0):
    visits = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for action, child in root.children.items():
        visits[action] = child.visit_count
    if temperature <= 0.0:
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[int(np.argmax(visits))] = 1.0
        return policy
    adjusted = np.power(visits, 1.0 / temperature)
    if adjusted.sum() == 0.0:
        for action, child in root.children.items():
            adjusted[action] = child.prior
    adjusted /= adjusted.sum()
    return adjusted.astype(np.float32)
