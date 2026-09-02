"""Lightweight tests for the territory-aware AlphaZero V3 pilot."""

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch
from torch import nn

import alphazero_v3.training_runner as v3_runner
from alphazero_v2.config import AlphaZeroConfig
from alphazero_v2.encoder import ENCODED_SHAPE as V2_ENCODED_SHAPE
from alphazero_v2.encoder import encode_state as encode_v2_state
from alphazero_v2.mcts import MCTS
from alphazero_v2.network import PolicyValueNetwork
from alphazero_v3.config import TerritoryPilotConfig
from alphazero_v3.diagnostics import run_fixed_diagnostics
from alphazero_v3.encoder import ENCODED_SHAPE, encode_state
from alphazero_v3.training_runner import (
    _sample_signal_metrics,
    initialize_pilot,
    load_pilot,
    run_pilot,
)
from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    NEUTRAL,
    NUM_ACTIONS,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
)
from train_alphazero_v3 import DEFAULT_MAX_ITERATIONS, parse_args


class ShapeCheckingNetwork(nn.Module):
    def __init__(self, preferred_action=PASS_ACTION):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.preferred_action = int(preferred_action)
        self.seen_shape = None

    def forward(self, inputs):
        self.seen_shape = tuple(inputs.shape[1:])
        logits = torch.zeros(
            (inputs.shape[0], NUM_ACTIONS),
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        logits[:, self.preferred_action] = 20.0 + self.anchor * 0.0
        values = torch.zeros(
            inputs.shape[0],
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        return logits, values


def two_color_territory_position():
    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    for x, y in ((2, 0), (2, 1), (2, 2), (0, 2), (1, 2)):
        logic.board[y][x] = BLUE
    for x, y in ((6, 6), (7, 6), (8, 6), (6, 7), (6, 8)):
        logic.board[y][x] = RED
    return logic


def tiny_config(**overrides):
    values = TerritoryPilotConfig(
        channels=8,
        residual_blocks=1,
        replay_max_positions=8,
        batch_size=4,
        training_updates_per_iteration=1,
        self_play_games_per_iteration=1,
        mcts_simulations=4,
        diagnostic_simulations=4,
    ).to_dict()
    values.update(overrides)
    return TerritoryPilotConfig.from_dict(values)


def test_v3_encoder_shape_canonical_territory_and_occupied_zero():
    logic = two_color_territory_position()
    blue = encode_state(logic)
    assert blue.shape == ENCODED_SHAPE == (9, 9, 9)
    assert np.array_equal(blue[:7], encode_v2_state(logic))
    assert np.all(blue[7, :2, :2] == 1.0)
    assert np.all(blue[8, 7:, 7:] == 1.0)
    assert blue[7, 0, 2] == blue[8, 0, 2] == 0.0
    assert blue[7, 6, 6] == blue[8, 6, 6] == 0.0

    logic.turn = RED
    red = encode_state(logic)
    assert np.array_equal(red[7], blue[8])
    assert np.array_equal(red[8], blue[7])


def test_v2_encoder_and_network_defaults_remain_seven_planes():
    logic = GreatKingdomLogicV2()
    assert encode_v2_state(logic).shape == V2_ENCODED_SHAPE == (7, 9, 9)
    network = PolicyValueNetwork(channels=8, residual_blocks=1)
    assert network.input_planes == 7
    logits, values = network(torch.zeros((2, 7, 9, 9)))
    assert logits.shape == (2, 82)
    assert values.shape == (2,)


def test_v3_network_forward_and_mcts_use_nine_planes():
    network = PolicyValueNetwork(channels=8, residual_blocks=1, input_planes=9)
    logits, values = network(torch.zeros((2, *ENCODED_SHAPE)))
    assert logits.shape == (2, 82)
    assert values.shape == (2,)

    checking = ShapeCheckingNetwork()
    config = AlphaZeroConfig(mcts_simulations=2, dirichlet_fraction=0.0)
    root = MCTS(
        checking,
        config,
        "cpu",
        state_encoder=encode_state,
    ).run(GreatKingdomLogicV2())
    assert checking.seen_shape == ENCODED_SHAPE
    assert PASS_ACTION in root.children


def test_v3_checkpoint_resume_preserves_architecture_and_outputs():
    config = tiny_config()
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as temp_dir_name:
        run_dir = Path(temp_dir_name) / "v3"
        state = initialize_pilot(run_dir, config, device)
        assert state.network.input_planes == 9
        assert state.replay.encoded_shape == ENCODED_SHAPE
        inputs = torch.from_numpy(encode_state(GreatKingdomLogicV2())).unsqueeze(0)
        state.network.eval()
        with torch.no_grad():
            expected = state.network(inputs)
        loaded_config, loaded = load_pilot(run_dir, device)
        loaded.network.eval()
        with torch.no_grad():
            actual = loaded.network(inputs)
        assert loaded_config == config
        assert loaded.network.input_planes == 9
        assert loaded.replay.encoded_shape == ENCODED_SHAPE
        assert torch.allclose(expected[0], actual[0])
        assert torch.allclose(expected[1], actual[1])


def test_v3_pass_and_territory_signal_metrics():
    state = np.zeros(ENCODED_SHAPE, dtype=np.float32)
    state[7, 0, 0] = 1.0
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[PASS_ACTION] = 0.25
    policy[0] = 0.75
    example = SimpleNamespace(state=state, policy=policy)
    metrics = _sample_signal_metrics([example])
    assert metrics["states_with_current_territory"] == 1
    assert metrics["territory_state_fraction"] == 1.0
    assert metrics["mean_pass_target_probability"] == 0.25
    assert metrics["pass_target_nonzero_fraction"] == 1.0
    assert metrics["territory_pass_target_nonzero_fraction"] == 1.0


def test_v3_fixed_diagnostics_and_legality():
    config = tiny_config()
    state = SimpleNamespace(
        iteration=0,
        network=ShapeCheckingNetwork(preferred_action=1 + 2 * BOARD_SIZE),
    )
    diagnostic = run_fixed_diagnostics(state, config, torch.device("cpu"))
    assert diagnostic["iteration"] == 0
    assert diagnostic["mcts_simulations"] == 4
    assert diagnostic["immediate_capture"]["success"]
    assert diagnostic["defense"]["safe_defense_actions"] == [19]
    assert diagnostic["defense"]["success"]
    assert diagnostic["winning_pass"]["fixture_terminal_result"] == (
        "PASS_SCORE_END"
    )
    assert diagnostic["legality"] == {
        "own_territory_legal": True,
        "opponent_territory_excluded": True,
        "pure_suicide_excluded": True,
    }


def test_max_iterations_parser_and_exact_completed_boundary():
    defaults = parse_args(["--run-dir", "pilot"])
    assert defaults.max_iterations == DEFAULT_MAX_ITERATIONS == 50
    explicit = parse_args(
        ["--run-dir", "pilot", "--max-iterations", "3"]
    )
    assert explicit.max_iterations == 3
    with redirect_stderr(io.StringIO()):
        try:
            parse_args(["--run-dir", "pilot", "--max-iterations", "0"])
        except SystemExit as error:
            assert error.code != 0
        else:
            raise AssertionError("non-positive max iterations must be rejected")

    state = SimpleNamespace(iteration=1)
    original = v3_runner.run_iteration

    def synthetic_iteration(*args, **kwargs):
        state.iteration += 1
        return {
            "iteration": state.iteration,
            "iteration_seconds": 0.0,
            "total_self_play_games": 0,
            "new_samples": 0,
            "capture_endings": 0,
            "pass_score_endings": 0,
            "pass_action_count": 0,
            "territory_state_fraction": 0.0,
            "mean_pass_target_probability": 0.0,
            "pass_target_nonzero_fraction": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
        }

    v3_runner.run_iteration = synthetic_iteration
    try:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            with redirect_stdout(io.StringIO()):
                completed = run_pilot(
                    temp_dir_name,
                    state,
                    None,
                    None,
                    max_iterations=3,
                )
    finally:
        v3_runner.run_iteration = original
    assert state.iteration == 3
    assert len(completed) == 2


if __name__ == "__main__":
    test_v3_encoder_shape_canonical_territory_and_occupied_zero()
    test_v2_encoder_and_network_defaults_remain_seven_planes()
    test_v3_network_forward_and_mcts_use_nine_planes()
    test_v3_checkpoint_resume_preserves_architecture_and_outputs()
    test_v3_pass_and_territory_signal_metrics()
    test_v3_fixed_diagnostics_and_legality()
    test_max_iterations_parser_and_exact_completed_boundary()
    print("AlphaZero V3 territory pilot tests: PASS")
