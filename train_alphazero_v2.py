"""Sustained single-network AlphaZero training on Great Kingdom Rules V2."""

import argparse
from pathlib import Path
import shlex

from alphazero_v2.training_runner import (
    TrainingRunConfig,
    choose_device,
    initialize_run,
    latest_completed_iteration,
    load_run,
    run_until_budget,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_mutually_exclusive_group(required=True)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--resume", type=Path)
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help=(
            "optional total completed-iteration time budget; omit or use 0 "
            "for unlimited training"
        ),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.hours is not None and args.hours < 0:
        parser.error("--hours must be non-negative")
    if args.hours == 0:
        args.hours = None
    return args


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.resume if args.resume is not None else args.run_dir
    try:
        device = choose_device(args.device)
        if args.resume is not None:
            config, state = load_run(run_dir, device)
        else:
            config = TrainingRunConfig()
            state = initialize_run(run_dir, config, device)
        run_until_budget(run_dir, state, config, device, args.hours)
    except KeyboardInterrupt:
        latest_path = run_dir / "latest.pt"
        completed = latest_completed_iteration(run_dir) if latest_path.exists() else 0
        resume = (
            "python train_alphazero_v2.py --resume "
            f"{shlex.quote(str(run_dir))}"
        )
        print("\nAlphaZero V2 training interrupted safely.", flush=True)
        print(f"Last completed iteration: {completed}", flush=True)
        print(f"Run directory: {run_dir}", flush=True)
        print(f"Resume command: {resume}", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
