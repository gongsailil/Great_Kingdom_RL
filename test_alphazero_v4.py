"""Integrated tests for the AlphaZero V4 stability architecture."""

from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from alphazero_v2.evaluate import (
    capture_tactical_position,
    defense_tactical_position,
    winning_pass_tactical_position,
)
from alphazero_v2.self_play import TrainingExample
from alphazero_v3.encoder import ENCODED_SHAPE, encode_state
from alphazero_v3.temperature_audit import temperature_for_ply
from alphazero_v4.config import V4Config
from alphazero_v4.mcts import V4MCTS
from alphazero_v4.network import (
    PolicyValueLogitNetwork,
    value_logit_to_probability,
    value_logit_to_scalar,
)
from alphazero_v4.replay import DuplicateAwareTrainingView
from alphazero_v4.tactical import solve_tactical_root
from alphazero_v4.training_runner import (
    initialize_run,
    load_run,
    run_iteration,
    value_loss_components,
)
from great_kingdom_v2 import (
    BLUE,
    NUM_ACTIONS,
    PASS_ACTION,
    GreatKingdomLogicV2,
)


class ConstantLogitNetwork(nn.Module):
    def __init__(self, value_logit=0.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(float(value_logit)))

    def forward(self, inputs):
        logits = torch.zeros(
            (len(inputs), NUM_ACTIONS), dtype=inputs.dtype, device=inputs.device
        ) + self.anchor * 0.0
        values = torch.ones(
            len(inputs), dtype=inputs.dtype, device=inputs.device
        ) * self.anchor
        return logits, values


def tiny_config(**overrides):
    values = V4Config(
        channels=8,
        residual_blocks=1,
        mcts_simulations=2,
        self_play_games_per_iteration=1,
        replay_max_positions=16,
        batch_size=4,
        training_updates_per_iteration=1,
        checkpoint_milestone_interval=5,
        checkpoint_keep_recent=2,
        diagnostic_interval=10,
        value_oracle_max_states=2,
    ).to_dict()
    values.update(overrides)
    return V4Config.from_dict(values)


def replay_example(state, policy, value, player=BLUE):
    return TrainingExample(
        np.asarray(state, dtype=np.float32),
        np.asarray(policy, dtype=np.float32),
        float(value),
        int(player),
    )


def test_raw_value_logit_conversion_and_bce_loss():
    logits = torch.tensor((-2.0, 0.0, 2.0))
    probabilities = value_logit_to_probability(logits)
    scalars = value_logit_to_scalar(logits)
    assert torch.allclose(probabilities, torch.sigmoid(logits))
    assert torch.allclose(scalars, 2.0 * probabilities - 1.0)
    assert scalars[1] == 0.0

    network = ConstantLogitNetwork(0.0)
    states = torch.zeros((1, *ENCODED_SHAPE))
    policy = torch.zeros((1, NUM_ACTIONS))
    policy[0, 0] = 1.0
    losses = value_loss_components(
        network, states, policy, torch.ones(1)
    )
    assert torch.allclose(
        losses[1], torch.tensor(np.log(2.0), dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(losses[3], torch.tensor(0.25), atol=1e-6)
    assert torch.allclose(losses[2], losses[0] + losses[1])


def test_v4_network_shape_and_parameter_count():
    network = PolicyValueLogitNetwork()
    policy, value_logit = network(torch.zeros((2, *ENCODED_SHAPE)))
    assert policy.shape == (2, NUM_ACTIONS)
    assert value_logit.shape == (2,)
    assert network.parameter_count() == 246_141


def test_duplicate_aggregation_and_contradictory_mean():
    state = np.zeros(ENCODED_SHAPE, dtype=np.float32)
    policy_a = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy_b = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy_a[0] = 1.0
    policy_b[1] = 1.0
    view = DuplicateAwareTrainingView(
        [
            replay_example(state, policy_a, 1.0),
            replay_example(state, policy_b, -1.0),
            replay_example(state, policy_a, 1.0),
        ]
    )
    assert len(view) == 1
    sample = view.samples[0]
    assert sample.occurrence_count == 3
    assert np.isclose(sample.win_probability, 2.0 / 3.0)
    assert np.allclose(sample.policy, (2.0 * policy_a + policy_b) / 3.0)
    metrics = view.metrics()
    assert metrics["duplicate_group_count"] == 1
    assert metrics["contradictory_z_group_count"] == 1
    assert metrics["samples_in_contradictory_groups"] == 3


def test_weighted_group_sampling_preserves_occurrence_distribution():
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    first = np.zeros(ENCODED_SHAPE, dtype=np.float32)
    second = np.ones(ENCODED_SHAPE, dtype=np.float32)
    view = DuplicateAwareTrainingView(
        [replay_example(first, policy, 1.0)]
        + [replay_example(second, policy, -1.0) for _ in range(3)]
    )
    draws = view.sample(20_000, np.random.default_rng(7))
    second_fraction = np.mean([sample.state[0, 0, 0] == 1.0 for sample in draws])
    assert abs(second_fraction - 0.75) < 0.02


def test_exact_tactical_win_threat_and_safe_filtering():
    capture = solve_tactical_root(capture_tactical_position())
    assert capture.mode == "IMMEDIATE_WIN"
    assert capture.immediate_win_actions == (19,)
    assert capture.allowed_actions == (19,)

    winning_pass = solve_tactical_root(winning_pass_tactical_position())
    assert winning_pass.mode == "IMMEDIATE_WIN"
    assert PASS_ACTION in winning_pass.immediate_win_actions

    defense = solve_tactical_root(defense_tactical_position())
    assert defense.mode == "SAFE_DEFENSE"
    assert defense.opponent_threat_actions == (19,)
    assert defense.safe_defense_actions == (19,)
    assert set(defense.allowed_actions).isdisjoint(defense.exact_unsafe_actions)


def test_no_false_filter_without_safe_action():
    logic = defense_tactical_position()
    logic.castles_remaining[BLUE] = 0
    forced = solve_tactical_root(logic)
    assert forced.mode == "FORCED_LOSS"
    assert forced.forced_loss
    assert forced.safe_defense_actions == ()
    assert forced.allowed_actions == forced.legal_actions == (PASS_ACTION,)

    normal = solve_tactical_root(GreatKingdomLogicV2())
    assert normal.mode == "NORMAL"
    assert normal.allowed_actions == normal.legal_actions


def test_v4_mcts_converts_logit_value_in_node_perspective():
    network = ConstantLogitNetwork(float(np.log(3.0)))
    search = V4MCTS(
        network,
        tiny_config().self_play_config(),
        "cpu",
        state_encoder=encode_state,
    )
    _, scalar = search._network_evaluate(GreatKingdomLogicV2())
    assert np.isclose(scalar, 0.5, atol=1e-6)
    root = search.run(
        GreatKingdomLogicV2(), root_actions=(PASS_ACTION,), add_root_noise=False
    )
    assert tuple(root.children) == (PASS_ACTION,)


def test_early8_and_checkpoint_resume():
    config = tiny_config()
    assert config.temperature_schedule == "early8"
    assert temperature_for_ply(config.temperature_schedule, 7) == 1.0
    assert temperature_for_ply(config.temperature_schedule, 8) == 0.0
    with tempfile.TemporaryDirectory() as temp_name:
        run_dir = Path(temp_name) / "v4"
        state = initialize_run(run_dir, config, torch.device("cpu"))
        inputs = torch.from_numpy(encode_state(GreatKingdomLogicV2())).unsqueeze(0)
        state.network.eval()
        with torch.no_grad():
            expected = state.network(inputs)
        loaded_config, loaded = load_run(run_dir, torch.device("cpu"))
        loaded.network.eval()
        with torch.no_grad():
            actual = loaded.network(inputs)
        assert loaded_config == config
        assert loaded.iteration == 0
        assert torch.allclose(expected[0], actual[0])
        assert torch.allclose(expected[1], actual[1])


def test_tiny_iteration_is_legal_and_resumeable():
    config = tiny_config()
    with tempfile.TemporaryDirectory() as temp_name:
        run_dir = Path(temp_name) / "v4-iteration"
        state = initialize_run(run_dir, config, torch.device("cpu"))
        metric = run_iteration(
            run_dir,
            state,
            config,
            torch.device("cpu"),
            Path("unused-before-iteration-10.pt"),
        )
        assert metric["iteration"] == 1
        assert metric["new_games"] == 1
        assert metric["new_samples"] > 0
        assert all(
            game["illegal_probability_violations"] == 0
            for game in metric["games"]
        )
        assert metric["immediate_win_opportunities"] == (
            metric["immediate_win_taken"]
        )
        assert np.isfinite(metric["value_bce_loss"])
        assert np.isfinite(metric["brier_score"])
        loaded_config, loaded = load_run(run_dir, torch.device("cpu"))
        assert loaded_config == config
        assert loaded.iteration == 1
        assert len(loaded.replay) == config.replay_max_positions


if __name__ == "__main__":
    test_raw_value_logit_conversion_and_bce_loss()
    test_v4_network_shape_and_parameter_count()
    test_duplicate_aggregation_and_contradictory_mean()
    test_weighted_group_sampling_preserves_occurrence_distribution()
    test_exact_tactical_win_threat_and_safe_filtering()
    test_no_false_filter_without_safe_action()
    test_v4_mcts_converts_logit_value_in_node_perspective()
    test_early8_and_checkpoint_resume()
    test_tiny_iteration_is_legal_and_resumeable()
    print("AlphaZero V4 stability tests: PASS")
