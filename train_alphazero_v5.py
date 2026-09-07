"""Run the integrated AlphaZero V5 BEST/candidate strategy training."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from alphazero_v2.training_runner import _atomic_json_save
from alphazero_v5.config import V5Config
from alphazero_v5.training_runner import (
    choose_device,
    initialize_run,
    load_run,
    prepare_run_assets,
    run_throughput_measurement,
    run_until_target,
)


DEFAULT_RUN_DIR = Path("runs/alphazero_v5/strategy_20260908")
DEFAULT_V4_REPLAY = Path("runs/alphazero_v4/stability_20260903/replay_buffer.pt")
DEFAULT_REPORT_DIR = Path("reports/alphazero_v5_strategy")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-dir", type=Path, default=None)
    group.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--v4-replay", type=Path, default=DEFAULT_V4_REPLAY)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--throughput-only", action="store_true")
    args = parser.parse_args(argv)
    if args.run_dir is None and args.resume is None:
        args.run_dir = DEFAULT_RUN_DIR
    return args


def main(argv=None):
    args = parse_args(argv)
    device = choose_device(args.device)
    run_dir = args.resume or args.run_dir
    if args.resume:
        config, state = load_run(run_dir, device)
    else:
        config = V5Config()
        state = initialize_run(run_dir, config, device)
    start_pool, openings = prepare_run_assets(run_dir, config, args.v4_replay)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(config.to_dict(), args.report_dir / "config.json")
    if args.throughput_only:
        # The gate must not advance the persisted training RNG or replay.
        gate_rng = np.random.default_rng()
        gate_rng.bit_generator.state = copy.deepcopy(state.rng.bit_generator.state)
        report = run_throughput_measurement(
            state.best_network, state.calibration_temperature, config, device,
            gate_rng, start_pool, games=64)
        _atomic_json_save(report, args.report_dir / "throughput.json")
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    try:
        run_until_target(run_dir, state, config, device, start_pool, openings)
    except KeyboardInterrupt:
        print(
            f"Interrupted. Last completed cycle={state.cycle}, best=v{state.best_version}, "
            f"run={run_dir}. Resume: python train_alphazero_v5.py --resume {run_dir}",
            flush=True,
        )
        return


if __name__ == "__main__":
    main()
