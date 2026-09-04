"""V4 MCTS value adapter plus Rules-exact root candidate filtering."""

import numpy as np
import torch

from alphazero_v2.mcts import MCTS, Node, backup, terminal_value

from .network import value_logit_to_scalar


class V4MCTS(MCTS):
    def _network_evaluate(self, logic):
        encoded = (
            torch.from_numpy(self.state_encoder(logic))
            .unsqueeze(0)
            .to(self.device)
        )
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            logits, value_logit = self.network(encoded)
            value = value_logit_to_scalar(value_logit)
        if was_training:
            self.network.train()
        return logits[0].detach().cpu().numpy(), float(value[0].item())

    @staticmethod
    def _restrict_root(root, allowed_actions):
        allowed = {int(action) for action in allowed_actions}
        if not allowed:
            raise ValueError("root candidate set cannot be empty")
        missing = allowed - set(root.children)
        if missing:
            raise RuntimeError(f"tactical solver returned illegal actions: {missing}")
        root.children = {
            action: child
            for action, child in root.children.items()
            if action in allowed
        }
        total_prior = sum(child.prior for child in root.children.values())
        if total_prior <= 0.0:
            raise RuntimeError("restricted root has zero prior mass")
        for child in root.children.values():
            child.prior /= total_prior

    def run(self, logic, *, root_actions=None, add_root_noise=False, rng=None):
        if logic.game_over:
            raise ValueError("cannot run MCTS from a terminal state")
        if rng is None:
            rng = np.random.default_rng()
        root = Node(prior=1.0, to_play=logic.turn)
        self._expand_and_evaluate(root, logic)
        if root_actions is not None:
            self._restrict_root(root, root_actions)
        if add_root_noise:
            self._add_root_noise(root, rng)

        for _ in range(self.config.mcts_simulations):
            simulation = logic.copy()
            node = root
            search_path = [root]
            while node.expanded() and not simulation.game_over:
                action, node = self._select_child(node)
                result = simulation.apply_action(action)
                if result.name.startswith("IMPOSSIBLE"):
                    raise RuntimeError(f"V4 MCTS selected illegal action: {result.name}")
                search_path.append(node)
            if simulation.game_over:
                value_player = simulation.turn
                value = terminal_value(simulation, value_player)
            else:
                value_player = simulation.turn
                value = self._expand_and_evaluate(node, simulation)
            backup(search_path, value, value_player)
        return root
