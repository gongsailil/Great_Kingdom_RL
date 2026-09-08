"""Validate one frozen AlphaZero V5 checkpoint and its strategic suite."""

import argparse
import json
from pathlib import Path

from alphazero_v5.common import atomic_json_save
from alphazero_v5.evaluation import (
    load_v5_checkpoint, run_strategic_suite, validate_checkpoint,
)


DEFAULT_CHECKPOINT = Path("runs/alphazero_v5/strategy_20260908/latest.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mcts-simulations", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional JSON output path; omitted results are printed only",
    )
    args = parser.parse_args(argv)
    if args.mcts_simulations <= 0:
        parser.error("--mcts-simulations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    validation = validate_checkpoint(args.checkpoint, args.device)
    checkpoint = load_v5_checkpoint(args.checkpoint, args.device)
    strategic = run_strategic_suite(checkpoint, args.mcts_simulations)
    payload = {
        "checkpoint": str(args.checkpoint),
        "validation": validation,
        "strategic_suite": strategic,
        "exact_tactical_pass": all(
            row["success"] for row in strategic if row["exact_assertion"]
        ),
        "note": (
            "Historical V2/V3/V4 arena results are frozen in docs/RESULTS.md; "
            "reproduce them from their annotated Git tags."
        ),
    }
    if args.output is not None:
        atomic_json_save(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
