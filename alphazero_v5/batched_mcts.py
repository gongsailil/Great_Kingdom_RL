"""PUCT over concurrent Rules V2 games with batched GPU leaf inference."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import BOARD_SIZE, PASS_ACTION, MoveResultV2
from .common import Node, backup, masked_policy, terminal_value


@dataclass
class SearchRequest:
    logic: object
    simulations: int
    root_actions: tuple | None = None
    add_root_noise: bool = False
    rng: object = None


@dataclass
class SearchResult:
    root: Node
    root_network_value: float


class BatchedNetworkEvaluator:
    def __init__(self, network, encoder, device, value_transform):
        self.network = network
        self.encoder = encoder
        self.device = torch.device(device)
        self.value_transform = value_transform
        self.inference_calls = 0
        self.inference_positions = 0
        self.batch_sizes = []

    def evaluate(self, logics):
        if not logics:
            return np.empty((0, 82), np.float32), np.empty((0,), np.float32)
        inputs = torch.from_numpy(np.stack([self.encoder(logic) for logic in logics])).to(self.device)
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(inputs)
            logits = outputs[0]
            values = self.value_transform(outputs[1])
        if was_training:
            self.network.train()
        self.inference_calls += 1
        self.inference_positions += len(logics)
        self.batch_sizes.append(len(logics))
        return logits.detach().cpu().numpy(), values.detach().cpu().numpy()

    def metrics(self):
        return {"network_inference_calls": self.inference_calls,
                "network_inference_positions": self.inference_positions,
                "mean_inference_batch_size": (float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0),
                "max_inference_batch_size": max(self.batch_sizes, default=0)}


class BatchedMCTS:
    def __init__(self, evaluator, c_puct=1.5, dirichlet_alpha=0.3,
                 dirichlet_fraction=0.25):
        self.evaluator = evaluator
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_fraction = float(dirichlet_fraction)

    @staticmethod
    def _expand(node, logic, priors):
        for raw_action in np.flatnonzero(priors > 0):
            action = int(raw_action)
            if action == PASS_ACTION:
                child_player = 3 - node.to_play
            else:
                result = logic.classify_placement(node.to_play, action % BOARD_SIZE, action // BOARD_SIZE)
                if result not in (MoveResultV2.NORMAL, MoveResultV2.CAPTURE_WIN):
                    raise RuntimeError("batched masked policy included illegal action")
                child_player = node.to_play if result == MoveResultV2.CAPTURE_WIN else 3 - node.to_play
            node.children[action] = Node(float(priors[action]), child_player)

    @staticmethod
    def _restrict_root(root, actions):
        allowed = set(int(action) for action in actions)
        if not allowed or not allowed.issubset(root.children):
            raise RuntimeError("invalid batched root candidate restriction")
        root.children = {a: c for a, c in root.children.items() if a in allowed}
        total = sum(c.prior for c in root.children.values())
        if total <= 0:
            raise RuntimeError("restricted root has no prior mass")
        for child in root.children.values():
            child.prior /= total

    def _add_noise(self, root, rng):
        actions = list(root.children)
        noise = rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - self.dirichlet_fraction) * child.prior + self.dirichlet_fraction * sample

    def _select_child(self, parent):
        best_action, best_child, best_score = None, None, -float("inf")
        scale = math.sqrt(parent.visit_count + 1.0)
        for action, child in parent.children.items():
            child_value = child.value()
            q_value = child_value if child.to_play == parent.to_play else -child_value
            score = q_value + self.c_puct * child.prior * scale / (1 + child.visit_count)
            if score > best_score:
                best_action, best_child, best_score = action, child, score
        return best_action, best_child

    def run_many(self, requests):
        requests = list(requests)
        if not requests:
            return []
        for request in requests:
            if request.logic.game_over or int(request.simulations) <= 0:
                raise ValueError("search requests must be active with positive simulations")
            if request.add_root_noise and request.rng is None:
                raise ValueError("noisy search requires an RNG")
        roots = [Node(1.0, request.logic.turn) for request in requests]
        logits, root_values = self.evaluator.evaluate([request.logic for request in requests])
        for request, root, policy_logits in zip(requests, roots, logits):
            legal = action_mask_for_logic(request.logic, root.to_play)
            self._expand(root, request.logic, masked_policy(policy_logits, legal))
            if request.root_actions is not None:
                self._restrict_root(root, request.root_actions)
            if request.add_root_noise:
                self._add_noise(root, request.rng)

        maximum = max(int(request.simulations) for request in requests)
        for simulation_index in range(maximum):
            pending = []
            for index, (request, root) in enumerate(zip(requests, roots)):
                if simulation_index >= int(request.simulations):
                    continue
                logic = request.logic.copy()
                node, path = root, [root]
                while node.expanded() and not logic.game_over:
                    action, node = self._select_child(node)
                    result = logic.apply_action(action)
                    if result not in (MoveResultV2.NORMAL, MoveResultV2.CAPTURE_WIN,
                                      MoveResultV2.PASS, MoveResultV2.PASS_SCORE_END):
                        raise RuntimeError(f"batched MCTS selected illegal action {result.name}")
                    path.append(node)
                if logic.game_over:
                    backup(path, terminal_value(logic, logic.turn), logic.turn)
                else:
                    pending.append((index, logic, node, path))
            if pending:
                leaf_logits, leaf_values = self.evaluator.evaluate([item[1] for item in pending])
                for (_, logic, node, path), policy_logits, value in zip(pending, leaf_logits, leaf_values):
                    legal = action_mask_for_logic(logic, node.to_play)
                    self._expand(node, logic, masked_policy(policy_logits, legal))
                    backup(path, float(value), logic.turn)
        return [SearchResult(root, float(value)) for root, value in zip(roots, root_values)]

    def run(self, logic, simulations, **kwargs):
        return self.run_many([SearchRequest(logic, simulations, **kwargs)])[0]
