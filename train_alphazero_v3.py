"""Run the fixed 50-iteration territory-aware AlphaZero V3 pilot."""

import argparse
from pathlib import Path
import shlex

from alphazero_v2.training_runner import choose_device, latest_completed_iteration
from alphazero_v3.config import TerritoryPilotConfig
from alphazero_v3.training_runner import (
    initialize_pilot,
    load_pilot,
    run_pilot,
)


DEFAULT_MAX_ITERATIONS = 50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_mutually_exclusive_group(required=True)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="stop after this exact completed iteration (default: 50)",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.resume if args.resume is not None else args.run_dir
    try:
        device = choose_device(args.device)
        if args.resume is not None:
            config, state = load_pilot(run_dir, device)
        else:
            config = TerritoryPilotConfig()
            state = initialize_pilot(run_dir, config, device)
        run_pilot(run_dir, state, config, device, args.max_iterations)
    except KeyboardInterrupt:
        latest_path = run_dir / "latest.pt"
        completed = latest_completed_iteration(run_dir) if latest_path.exists() else 0
        resume = (
            "python train_alphazero_v3.py --resume "
            f"{shlex.quote(str(run_dir))} --max-iterations "
            f"{args.max_iterations}"
        )
        print("\nAlphaZero V3 pilot interrupted safely.", flush=True)
        print(f"Last completed iteration: {completed}", flush=True)
        print(f"Run directory: {run_dir}", flush=True)
        print(f"Resume command: {resume}", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
