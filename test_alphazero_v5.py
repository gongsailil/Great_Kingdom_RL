"""Focused correctness tests for the integrated AlphaZero V5 architecture."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from alphazero_v2.evaluate import apply_opening, capture_tactical_position, defense_tactical_position
from alphazero_v2.self_play import TrainingExample
from alphazero_v4.tactical import solve_tactical_root
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import BLUE, PASS_ACTION, GreatKingdomLogicV2

from alphazero_v5.batched_mcts import BatchedMCTS, BatchedNetworkEvaluator, SearchRequest
from alphazero_v5.calibration import fit_value_temperature
from alphazero_v5.config import V5Config
from alphazero_v5.diagnostics import run_fixed_tactical_diagnostics
from alphazero_v5.encoder import ENCODED_SHAPE, decode_state, encode_state, scoring_margin
from alphazero_v5.network import PolicyValueAuxNetwork, calibrated_probability, calibrated_scalar
from alphazero_v5.replay import CompactReplayBuffer, DuplicateAwareSplitView
from alphazero_v5.start_states import restore_logic, store_logic, validate_stored_logic
from alphazero_v5.symmetry import (
    inverse_policy, inverse_state, inverse_transform, transform_action,
    transform_policy, transform_state,
)
from alphazero_v5.training_runner import (
    clone_candidate, initialize_run, load_run, network_digest, should_promote,
)
from alphazero_v5.evaluation import load_v5_checkpoint, paired_bootstrap
from play_vs_alphazero_v5 import HumanVsAlphaZeroV5Controller


class TestEncoderAndSymmetry(unittest.TestCase):
    def test_twelve_planes_and_liberties(self):
        logic = apply_opening([0, 80, 1, 79])
        state = encode_state(logic)
        self.assertEqual(state.shape, (12, 9, 9))
        self.assertGreater(state[9, 0, 0], 0)
        self.assertEqual(state[9, 0, 0], state[9, 0, 1])
        self.assertGreater(state[10, 8, 8], 0)
        self.assertTrue(np.allclose(state[11], scoring_margin(logic)))

    def test_scoring_margin_uses_absolute_asymmetry_and_perspective(self):
        logic = GreatKingdomLogicV2()
        blue_value = scoring_margin(logic)
        logic.turn = 2
        self.assertAlmostEqual(scoring_margin(logic), -blue_value)
        self.assertLess(blue_value, 0)

    def test_d4_roundtrip_policy_pass_and_encoder_equivariance(self):
        logic = apply_opening([0, 80, 10, 70])
        state = encode_state(logic)
        policy = np.arange(82, dtype=np.float32)
        mask = action_mask_for_logic(logic, logic.turn)
        for transform in range(8):
            self.assertTrue(np.array_equal(inverse_state(transform_state(state, transform), transform), state))
            self.assertTrue(np.array_equal(inverse_policy(transform_policy(policy, transform), transform), policy))
            self.assertEqual(transform_action(PASS_ACTION, transform), PASS_ACTION)
            transformed_logic = decode_state(transform_state(state, transform))
            transformed_mask = action_mask_for_logic(transformed_logic, transformed_logic.turn)
            self.assertTrue(np.array_equal(transform_policy(mask, transform), transformed_mask))
            self.assertEqual(inverse_transform(inverse_transform(transform)), transform)


class TestNetworkAndReplay(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.network = PolicyValueAuxNetwork()

    def test_forward_and_value_conversion(self):
        outputs = self.network(torch.zeros(2, *ENCODED_SHAPE))
        self.assertEqual(outputs[0].shape, (2, 82))
        self.assertEqual(outputs[1].shape, (2,))
        self.assertEqual(outputs[2].shape, (2, 2, 9, 9))
        self.assertEqual(outputs[3].shape, (2,))
        logits = torch.tensor([-2.0, 0.0, 2.0])
        self.assertTrue(torch.allclose(calibrated_probability(logits, 1), torch.sigmoid(logits)))
        self.assertTrue(torch.allclose(calibrated_scalar(torch.zeros(1), 1), torch.zeros(1)))

    def _example(self, state, value=1):
        return TrainingExample(state.astype(np.float32), np.full(82, 1 / 82, np.float32), float(value), 1)

    def test_duplicate_targets_and_weighted_sampling(self):
        replay = CompactReplayBuffer(256)
        base = np.zeros(ENCODED_SHAPE, np.float32)
        # Contradictory exact state: empirical win probability 2/3.
        replay.extend([self._example(base, 1), self._example(base, 1), self._example(base, -1)])
        for index in range(80):
            state = base.copy(); state[0, index // 9, index % 9] = index / 80 + 0.01
            replay.extend([self._example(state, 1)])
        view = DuplicateAwareSplitView(replay, 0.05)
        all_samples = view.train + view.validation
        group = next(sample for sample in all_samples if sample.occurrence_count == 3)
        self.assertAlmostEqual(group.win_probability, 2 / 3)
        self.assertGreaterEqual(view.metrics()["contradictory_z_group_count"], 1)
        self.assertEqual(view.raw_size, 83)
        batch = view.sample_training_batch(16, np.random.default_rng(3), augment=True)
        self.assertEqual(batch[0].shape, (16, 12, 9, 9))
        self.assertEqual(batch[1].shape, (16, 82))
        self.assertEqual(batch[3].shape, (16, 2, 9, 9))

    def test_calibration_fits_and_does_not_mutate_network(self):
        replay = CompactReplayBuffer(256)
        rng = np.random.default_rng(2)
        for index in range(100):
            state = rng.random(ENCODED_SHAPE, dtype=np.float32)
            replay.extend([self._example(state, 1 if index % 2 else -1)])
        view = DuplicateAwareSplitView(replay, 0.2)
        before = network_digest(self.network)
        result = fit_value_temperature(self.network, view, torch.device("cpu"))
        self.assertGreater(result["temperature"], 0)
        self.assertLessEqual(result["after_nll"], result["before_nll"] + 1e-8)
        self.assertEqual(before, network_digest(self.network))


class TestSearchAndLifecycle(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.network = PolicyValueAuxNetwork()
        self.config = replace(
            V5Config(), concurrent_games=2, self_play_games_per_cycle=2,
            replay_max_positions=128, batch_size=2, training_updates_per_cycle=1,
            candidate_arena_openings=2, candidate_arena_games=4,
            opening_mcts_simulations=2, later_mcts_simulations=2,
            arena_mcts_simulations=2,
        )

    def test_batched_search_is_equivalent_for_identical_roots(self):
        evaluator = BatchedNetworkEvaluator(self.network, encode_state, "cpu", lambda x: calibrated_scalar(x, 1))
        search = BatchedMCTS(evaluator, 1.5, 0.3, 0)
        logic = apply_opening([0, 80])
        roots = search.run_many([SearchRequest(logic, 3), SearchRequest(logic.copy(), 3)])
        visits = [{a: c.visit_count for a, c in result.root.children.items()} for result in roots]
        self.assertEqual(visits[0], visits[1])
        self.assertGreater(evaluator.metrics()["max_inference_batch_size"], 1)

    def test_tactical_solver_and_fixed_diagnostics(self):
        capture = solve_tactical_root(capture_tactical_position())
        defense = solve_tactical_root(defense_tactical_position())
        self.assertEqual(capture.mode, "IMMEDIATE_WIN")
        self.assertEqual(defense.mode, "SAFE_DEFENSE")
        result = run_fixed_tactical_diagnostics(self.network, 1, self.config, "cpu")
        self.assertTrue(result["all_success"])

    def test_best_clone_freeze_and_promotion_gate(self):
        before = network_digest(self.network)
        candidate = clone_candidate(self.network, self.config, "cpu")
        with torch.no_grad():
            next(candidate.parameters()).add_(1)
        self.assertEqual(before, network_digest(self.network))
        tactical = {"all_success": True}
        arena = {"win_rate": 0.5625, "illegal_violations": 0, "tactical_failures": 0}
        self.assertTrue(should_promote(arena, tactical, 0.55))
        arena["win_rate"] = 0.546875
        self.assertFalse(should_promote(arena, tactical, 0.55))

    def test_checkpoint_resume_keeps_temperature_and_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            state = initialize_run(directory, self.config, torch.device("cpu"))
            state.calibration_temperature = 1.7
            from alphazero_v5.training_runner import save_run
            save_run(directory, state, self.config)
            loaded_config, loaded = load_run(directory, torch.device("cpu"))
            self.assertEqual(loaded_config, self.config)
            self.assertAlmostEqual(loaded.calibration_temperature, 1.7)
            self.assertEqual(network_digest(state.best_network), network_digest(loaded.best_network))
            checkpoint = load_v5_checkpoint(Path(directory) / "latest.pt", "cpu")
            self.assertEqual(checkpoint.best_version, 0)
            self.assertAlmostEqual(checkpoint.calibration_temperature, 1.7)

    def test_territory_start_roundtrip(self):
        logic = apply_opening([2, 80, 11, 79, 20, 78, 18, 77, 19, 76])
        stored = store_logic(logic)
        restored = validate_stored_logic(stored)
        self.assertEqual(stored, store_logic(restore_logic(stored)))
        self.assertTrue(np.array_equal(np.asarray(logic.board), np.asarray(restored.board)))

    def test_paired_bootstrap_requires_color_swap_and_reuse(self):
        games = [
            {"opening_id": 0, "opening_actions": [0, 80], "blue_agent": "v5_best",
             "red_agent": "old", "winner_agent": "v5_best"},
            {"opening_id": 0, "opening_actions": [0, 80], "blue_agent": "old",
             "red_agent": "v5_best", "winner_agent": "old"},
        ]
        result = paired_bootstrap(games, "v5_best", replicates=100, seed=1)
        self.assertEqual(result["pairs"], 1)
        self.assertEqual(result["win_rate"], 0.5)

    def test_human_controller_colors_ai_first_and_logging(self):
        class Checkpoint:
            path = Path(__file__)
            iteration = 0
            cycle = 0
            best_version = 0
            calibration_temperature = 1.0

        def selector(checkpoint, logic, simulations, top_k=5):
            action = int(np.flatnonzero(action_mask_for_logic(logic, logic.turn))[0])
            return {"action": action, "tactical_mode": "NORMAL", "top_actions": [],
                    "mcts_executed": True, "safe_defense_actions": [],
                    "unsafe_actions_filtered": 0}

        with tempfile.TemporaryDirectory() as directory:
            blue_human = HumanVsAlphaZeroV5Controller("blue", Checkpoint(), 2, directory, selector)
            self.assertTrue(blue_human.is_human_turn)
            blue_human.close_log()
            red_human = HumanVsAlphaZeroV5Controller("red", Checkpoint(), 2, directory, selector)
            self.assertTrue(red_human.is_ai_turn)
            action, _ = red_human.play_ai_move()
            self.assertEqual(action, 0)
            self.assertTrue(red_human.is_human_turn)
            red_human.close_log()
            payload = __import__("json").loads(red_human.log_path.read_text())
            self.assertEqual(payload["architecture"], "AlphaZero V5 strategy")
            self.assertEqual(len(payload["moves"]), 1)


if __name__ == "__main__":
    unittest.main()
