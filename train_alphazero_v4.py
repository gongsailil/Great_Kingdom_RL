"""Run or resume the bounded 50-iteration AlphaZero V4 stability experiment."""

import argparse
from pathlib import Path
import shlex

from alphazero_v4.config import V4Config
from alphazero_v4.training_runner import (
    choose_device,
    initialize_run,
    latest_completed_iteration,
    load_run,
    run_to_iteration,
)


DEFAULT_RUN_DIR = Path("runs/alphazero_v4/stability_20260903")
DEFAULT_ORACLE_REPLAY = Path(
    "runs/alphazero_v3/territory_pilot_20260901/replay_buffer.pt"
)
DEFAULT_MAX_ITERATIONS = 50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_mutually_exclusive_group()
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    parser.add_argument(
        "--oracle-replay", type=Path, default=DEFAULT_ORACLE_REPLAY
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.run_dir is None and args.resume is None:
        args.run_dir = DEFAULT_RUN_DIR
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.resume if args.resume is not None else args.run_dir
    try:
        device = choose_device(args.device)
        if args.resume is not None:
            config, state = load_run(run_dir, device)
        else:
            config = V4Config()
            state = initialize_run(run_dir, config, device)
        run_to_iteration(
            run_dir,
            state,
            config,
            device,
            args.max_iterations,
            args.oracle_replay,
        )
    except KeyboardInterrupt:
        completed = (
            latest_completed_iteration(run_dir)
            if (run_dir / "latest.pt").exists()
            else 0
        )
        print("\nAlphaZero V4 interrupted safely.", flush=True)
        print(f"Last completed iteration: {completed}", flush=True)
        print(f"Run directory: {run_dir}", flush=True)
        print(
            "Resume command: python train_alphazero_v4.py --resume "
            f"{shlex.quote(str(run_dir))} --max-iterations {args.max_iterations}",
            flush=True,
        )
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
