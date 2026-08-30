"""Lightweight tests for the sustained AlphaZero V2 training runner."""

import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch

import alphazero_v2.training_runner as training_runner
import train_alphazero_v2 as training_cli
from alphazero_v2.encoder import ENCODED_SHAPE, encode_state
from alphazero_v2.replay_buffer import ReplayBuffer
from alphazero_v2.self_play import TrainingExample
from alphazero_v2.training_runner import (
    TrainingRunConfig,
    append_metric,
    initialize_run,
    load_run,
    retain_iteration_checkpoints,
    run_iteration,
    run_until_budget,
)
from great_kingdom_v2 import GreatKingdomLogicV2, NUM_ACTIONS, PASS_ACTION
from train_alphazero_v2 import parse_args


def example_with_action(action, value=1.0, player=1, marker=0.0):
    state = np.full(ENCODED_SHAPE, marker, dtype=np.float32)
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[action] = 1.0
    return TrainingExample(state, policy, value, player)


def tiny_config(**overrides):
    values = TrainingRunConfig(
        channels=8,
        residual_blocks=1,
        replay_max_positions=8,
        batch_size=4,
        training_updates_per_iteration=2,
        checkpoint_milestone_interval=5,
        checkpoint_keep_recent=2,
    ).to_dict()
    values.update(overrides)
    return TrainingRunConfig.from_dict(values)


def network_outputs(network):
    inputs = torch.from_numpy(encode_state(GreatKingdomLogicV2())).unsqueeze(0)
    network.eval()
    with torch.no_grad():
        return tuple(output.clone() for output in network(inputs))


def test_training_defaults_are_fixed_initial_config():
    config = TrainingRunConfig()
    assert config.mcts_simulations == 64
    assert config.self_play_games_per_iteration == 32
    assert config.replay_max_positions == 50_000
    assert config.batch_size == 256
    assert config.training_updates_per_iteration == 64
    assert config.channels == 64 and config.residual_blocks == 3


def test_replay_fifo_append_sample_and_round_trip():
    replay = ReplayBuffer(max_positions=3)
    replay.extend(
        example_with_action(i, marker=float(i))
        for i in range(5)
    )
    assert len(replay) == 3
    assert replay.total_samples_seen == 5
    assert [sample.state[0, 0, 0] for sample in replay.samples] == [2.0, 3.0, 4.0]
    batch = replay.sample(256, np.random.default_rng(7))
    assert len(batch) == 3

    replay.generation_metadata = {
        "iteration": 2,
        "total_self_play_games": 64,
        "total_samples_generated": 100,
    }
    restored = ReplayBuffer.from_state_dict(replay.state_dict())
    assert len(restored) == 3
    assert restored.total_samples_seen == 5
    assert restored.generation_metadata == replay.generation_metadata
    assert np.array_equal(restored.samples[-1].policy, replay.samples[-1].policy)


def test_run_directory_checkpoint_resume_metrics_and_finite_budget():
    device = torch.device("cpu")
    config = tiny_config(self_play_games_per_iteration=2)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        run_dir = Path(temp_dir_name) / "run"
        state = initialize_run(run_dir, config, device)
        assert (run_dir / "config.json").exists()
        assert (run_dir / "latest.pt").exists()
        assert (run_dir / "replay_buffer.pt").exists()
        assert (run_dir / "checkpoints").is_dir()
        state.elapsed_seconds = 3600.0
        assert run_until_budget(run_dir, state, config, device, hours=1) == []
        assert state.iteration == 0
        state.elapsed_seconds = 0.0

        fake_examples = [
                example_with_action(PASS_ACTION, 1.0, 1),
                example_with_action(0, -1.0, 2),
        ]
        fake_games = [
            {
                "winner": 1,
                "terminal_reason": "CAPTURE_WIN",
                "game_length": 10,
                "pass_usage": 0,
                "illegal_probability_violations": 0,
                "score_blue": None,
                "score_red": None,
                "game_index": 0,
            },
            {
                "winner": 2,
                "terminal_reason": "PASS_SCORE_END",
                "game_length": 12,
                "pass_usage": 2,
                "illegal_probability_violations": 0,
                "score_blue": 0,
                "score_red": 0,
                "game_index": 1,
            },
        ]
        original_generate = training_runner.generate_self_play
        training_runner.generate_self_play = (
            lambda network, self_play_config, selected_device, rng: (
                fake_examples,
                fake_games,
            )
        )
        try:
            metric = run_iteration(run_dir, state, config, device)
        finally:
            training_runner.generate_self_play = original_generate

        assert metric["iteration"] == 1
        assert metric["new_games"] == 2
        assert metric["new_samples"] == 2
        assert metric["blue_wins"] == metric["red_wins"] == 1
        assert metric["capture_endings"] == metric["pass_score_endings"] == 1
        assert metric["mean_game_length"] == 11.0
        assert metric["mean_pass_count"] == 1.0
        assert metric["training_updates"] == 2
        assert all(
            np.isfinite(metric[name])
            for name in ("policy_loss", "value_loss", "total_loss")
        )
        expected = network_outputs(state.network)

        payload = torch.load(run_dir / "latest.pt", weights_only=False)
        required = {
            "network_state_dict",
            "optimizer_state_dict",
            "iteration",
            "total_self_play_games",
            "total_samples_generated",
            "replay_metadata",
            "config",
        }
        assert required <= payload.keys()
        loaded_config, loaded = load_run(run_dir, device)
        actual = network_outputs(loaded.network)
        assert loaded_config == config
        assert loaded.iteration == 1
        assert loaded.total_self_play_games == 2
        assert loaded.total_samples_generated == 2
        assert len(loaded.replay) == 2
        assert torch.allclose(expected[0], actual[0])
        assert torch.allclose(expected[1], actual[1])

        metrics = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert metrics == [metric]
        append_metric(run_dir / "metrics.jsonl", {"iteration": 2})
        assert len(
            (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ) == 2


def test_checkpoint_retention_keeps_recent_and_milestones():
    config = tiny_config()
    with tempfile.TemporaryDirectory() as temp_dir_name:
        run_dir = Path(temp_dir_name)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        latest = run_dir / "latest.pt"
        latest.write_bytes(b"resume-critical")
        for iteration in range(1, 8):
            (checkpoint_dir / f"iteration_{iteration:06d}.pt").touch()
        retain_iteration_checkpoints(run_dir, config)
        remaining = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
        assert remaining == [
            "iteration_000005.pt",
            "iteration_000006.pt",
            "iteration_000007.pt",
        ]
        assert latest.read_bytes() == b"resume-critical"


def test_time_budget_cli_parser():
    omitted = parse_args(["--run-dir", "runs/alphazero_v2/unlimited"])
    assert omitted.hours is None
    zero = parse_args(
        ["--hours", "0", "--run-dir", "runs/alphazero_v2/unlimited"]
    )
    assert zero.hours is None
    positive = parse_args(
        ["--hours", "8", "--run-dir", "runs/alphazero_v2/test"]
    )
    assert positive.hours == 8.0
    assert positive.run_dir == Path("runs/alphazero_v2/test")
    assert positive.resume is None
    resumed = parse_args(["--hours", "1.5", "--resume", "saved-run"])
    assert resumed.hours == 1.5
    assert resumed.resume == Path("saved-run")
    unlimited_resume = parse_args(["--resume", "saved-run"])
    assert unlimited_resume.hours is None
    with redirect_stderr(io.StringIO()):
        try:
            parse_args(["--hours", "-1", "--run-dir", "invalid"])
        except SystemExit as error:
            assert error.code != 0
        else:
            raise AssertionError("negative --hours must be rejected")


def test_unlimited_budget_does_not_stop_on_elapsed_time():
    state = SimpleNamespace(elapsed_seconds=10_000_000.0)
    calls = []
    original_run_iteration = training_runner.run_iteration

    def interrupt_instead_of_training(*args):
        calls.append(True)
        raise KeyboardInterrupt

    training_runner.run_iteration = interrupt_instead_of_training
    try:
        try:
            run_until_budget("unused", state, None, None, hours=None)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("synthetic interrupt should stop the test")
    finally:
        training_runner.run_iteration = original_run_iteration
    assert calls == [True]


def test_ctrl_c_reports_last_atomic_checkpoint_and_resume_command():
    with tempfile.TemporaryDirectory() as temp_dir_name:
        run_dir = Path(temp_dir_name) / "run"
        run_dir.mkdir()
        torch.save({"iteration": 7}, run_dir / "latest.pt")
        original_choose_device = training_cli.choose_device

        def interrupt_before_resume_load(requested):
            raise KeyboardInterrupt

        training_cli.choose_device = interrupt_before_resume_load
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                exit_code = training_cli.main(["--resume", str(run_dir)])
        finally:
            training_cli.choose_device = original_choose_device
        rendered = output.getvalue()
        assert exit_code == 130
        assert "Last completed iteration: 7" in rendered
        assert f"Run directory: {run_dir}" in rendered
        assert f"--resume {run_dir}" in rendered


if __name__ == "__main__":
    test_training_defaults_are_fixed_initial_config()
    test_replay_fifo_append_sample_and_round_trip()
    test_run_directory_checkpoint_resume_metrics_and_finite_budget()
    test_checkpoint_retention_keeps_recent_and_milestones()
    test_time_budget_cli_parser()
    test_unlimited_budget_does_not_stop_on_elapsed_time()
    test_ctrl_c_reports_last_atomic_checkpoint_and_resume_command()
    print("AlphaZero V2 training runner tests: PASS")
