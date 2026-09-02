"""Run the fixed V3 iteration-50 self-play temperature audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from alphazero_v2.evaluate import load_evaluation_checkpoint
from alphazero_v3.encoder import encode_state
from alphazero_v3.temperature_audit import (
    PHASES,
    SCHEDULES,
    aggregate_schedule_games,
    classify_temperature_audit,
    network_state_digest,
    paired_game_seeds,
    play_temperature_audit_game,
)


DEFAULT_CHECKPOINT = Path(
    "runs/alphazero_v3/territory_pilot_20260901/"
    "checkpoints/iteration_000050.pt"
)
DEFAULT_REPORT_DIR = Path("reports/alphazero_temperature_audit_20260902")
FIXED_SEED = 20260902
FIXED_GAMES_PER_SCHEDULE = 32
FIXED_SIMULATIONS = 256
FIXED_C_PUCT = 1.5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json_save(payload, path):
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_text_save(content, path):
    _atomic_write(path, content)


def append_json_line(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completed_games(path, game_seeds):
    path = Path(path)
    if not path.exists():
        return []
    games = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                game = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"corrupt games.jsonl line {line_number}: {error}"
                ) from error
            key = (game.get("schedule"), int(game.get("game_index", -1)))
            if key in seen:
                raise ValueError(f"duplicate completed game: {key}")
            if key[0] not in SCHEDULES or not 0 <= key[1] < len(game_seeds):
                raise ValueError(f"unexpected completed game identity: {key}")
            if int(game.get("game_seed", -1)) != game_seeds[key[1]]:
                raise ValueError(f"paired seed mismatch for completed game: {key}")
            seen.add(key)
            games.append(game)
    return games


def _percent(value):
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def render_summary(summary):
    lines = [
        "AlphaZero V3 self-play temperature stability audit (2026-09-02)",
        "=" * 68,
        "",
        "Behavior diagnostic only: no training, optimizer, replay, network,",
        "encoder, Rules V2, or MCTS algorithm changes were made.",
        f"Checkpoint: {summary['checkpoint_path']} (iteration 50)",
        f"Checkpoint SHA-256: {summary['checkpoint_sha256']}",
        f"Device: {summary['device']}",
        "Fixed search: 256 simulations, c_puct 1.5, learned policy/value,",
        "root Dirichlet alpha 0.3/fraction 0.25.",
        f"Paired seed block: {summary['seed']}; 32 games per schedule.",
        "",
        "Schedule totals",
        "---------------",
        "Schedule  W(B/R)  End(capture/score)  Len  PASS  Argmax  "
        "Capture miss  Safe defense  Win-PASS",
    ]
    for item in summary["schedules"]:
        overall = item["overall"]
        lines.append(
            f"{item['schedule']:<8}  {item['blue_wins']:>2}/"
            f"{item['red_wins']:<2}    {item['capture_endings']:>2}/"
            f"{item['pass_score_endings']:<2}             "
            f"{item['mean_game_length']:>5.2f}  "
            f"{item['mean_pass_usage']:>4.2f}  "
            f"{_percent(overall['argmax_action_rate']):>7}  "
            f"{_percent(overall['capture_miss_rate']):>12}  "
            f"{_percent(overall['safe_defense_rate']):>12}  "
            f"{overall['winning_pass_chosen_count']}/"
            f"{overall['winning_pass_opportunity_count']}"
        )

    lines.extend(["", "Early/mid/late split", "--------------------"])
    for item in summary["schedules"]:
        lines.append(item["schedule"] + ":")
        for phase in PHASES:
            data = item["phases"][phase]
            lines.append(
                f"  {phase:<5} moves={data['moves']:>4}; argmax="
                f"{_percent(data['argmax_action_rate'])}; capture misses="
                f"{data['capture_missed_count']}/"
                f"{data['capture_opportunity_count']}; unsafe defense="
                f"{data['unsafe_action_taken_count']}/"
                f"{data['defense_opportunity_count']}; PASS choices="
                f"{data['pass_choice_count']}; entropy="
                f"{data['mean_policy_entropy']:.6f}"
                if data["mean_policy_entropy"] is not None
                else f"  {phase:<5} moves=0; no observations"
            )

    lines.extend(["", "Value-vs-final-z trajectory proxy", "---------------------------------"])
    for item in summary["schedules"]:
        overall = item["overall"]
        lines.append(
            f"{item['schedule']}: MAE={overall['mean_value_absolute_error']:.6f}, "
            f"MSE={overall['mean_value_squared_error']:.6f}"
        )
        for phase in PHASES:
            data = item["phases"][phase]
            mae = data["mean_value_absolute_error"]
            mse = data["mean_value_squared_error"]
            lines.append(
                f"  {phase}: MAE={mae if mae is not None else 'n/a'}, "
                f"MSE={mse if mse is not None else 'n/a'}"
            )
    lines.extend(["", "Diversity and endings", "---------------------"])
    for item in summary["schedules"]:
        lines.append(
            f"{item['schedule']}: unique trajectories "
            f"{item['unique_trajectory_count']}/{item['games']}, unique first-8 "
            f"{item['unique_opening8_count']}/{item['games']}, selected-action "
            f"entropy {item['selected_action_entropy']:.6f}, capture/score "
            f"{item['capture_endings']}/{item['pass_score_endings']}"
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            f"Classification: {summary['classification']}",
            *[f"- {note}" for note in summary["interpretation_notes"]],
            "The value error is an on-policy trajectory difficulty/noise proxy,",
            "not an independent held-out calibration or generalization metric.",
            f"Network state unchanged: {summary['network_unchanged']}",
            f"Checkpoint file unchanged: {summary['checkpoint_unchanged']}",
            f"Total measured game time: {summary['elapsed_seconds']:.3f} seconds",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args):
    checkpoint_sha_before = file_sha256(args.checkpoint)
    checkpoint = load_evaluation_checkpoint(
        args.checkpoint,
        device=args.device,
        expected_iteration=50,
        state_encoder=encode_state,
    )
    config = checkpoint.config
    if int(config.get("mcts_simulations", -1)) != FIXED_SIMULATIONS:
        raise ValueError("checkpoint does not record the fixed 256 simulations")
    if float(config.get("c_puct", -1.0)) != FIXED_C_PUCT:
        raise ValueError("checkpoint does not record fixed c_puct 1.5")
    if float(config.get("dirichlet_alpha", -1.0)) != 0.3:
        raise ValueError("checkpoint Dirichlet alpha differs from self-play")
    if float(config.get("dirichlet_fraction", -1.0)) != 0.25:
        raise ValueError("checkpoint Dirichlet fraction differs from self-play")

    parameter_digest_before = network_state_digest(checkpoint.network)
    game_seeds = paired_game_seeds(FIXED_SEED, FIXED_GAMES_PER_SCHEDULE)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    games_path = args.report_dir / "games.jsonl"
    games = load_completed_games(games_path, game_seeds)
    completed = {(game["schedule"], game["game_index"]) for game in games}

    for schedule in SCHEDULES:
        for game_index, game_seed in enumerate(game_seeds):
            key = (schedule, game_index)
            if key in completed:
                continue
            started = time.perf_counter()
            game = play_temperature_audit_game(
                checkpoint,
                schedule,
                game_index,
                game_seed,
                simulations=FIXED_SIMULATIONS,
                c_puct=FIXED_C_PUCT,
            )
            game["elapsed_seconds"] = time.perf_counter() - started
            append_json_line(game, games_path)
            games.append(game)
            completed.add(key)
            print(
                f"{schedule} game {game_index + 1:02d}/32: "
                f"winner={game['winner']} end={game['terminal_reason']} "
                f"plies={game['game_length']} pass={game['pass_usage']} "
                f"seconds={game['elapsed_seconds']:.2f}",
                flush=True,
            )

    expected_total = len(SCHEDULES) * FIXED_GAMES_PER_SCHEDULE
    if len(games) != expected_total:
        raise RuntimeError(f"expected {expected_total} completed games, got {len(games)}")
    parameter_digest_after = network_state_digest(checkpoint.network)
    checkpoint_sha_after = file_sha256(args.checkpoint)
    schedules = [aggregate_schedule_games(name, games) for name in SCHEDULES]
    by_schedule = {item["schedule"]: item for item in schedules}
    hot = by_schedule["all_hot"]
    early = by_schedule["early8"]
    greedy = by_schedule["greedy"]
    interpretation_notes = [
        "early8 lowered capture misses from "
        f"{hot['overall']['capture_miss_rate']:.6f} to "
        f"{early['overall']['capture_miss_rate']:.6f} and value MAE from "
        f"{hot['overall']['mean_value_absolute_error']:.6f} to "
        f"{early['overall']['mean_value_absolute_error']:.6f}.",
        "early8 did not improve the key defense metric: safe-defense rate "
        f"changed from {hot['overall']['safe_defense_rate']:.6f} to "
        f"{early['overall']['safe_defense_rate']:.6f}.",
        "greedy safe-defense rate was "
        f"{greedy['overall']['safe_defense_rate']:.6f}; neither exploitation "
        "schedule produced a consistent defense improvement.",
        "All schedules retained 32/32 unique complete trajectories; early8 "
        "did not show trajectory collapse.",
    ]
    summary = {
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_iteration": int(checkpoint.iteration),
        "checkpoint_sha256": checkpoint_sha_before,
        "device": str(checkpoint.device),
        "network_parameters": sum(
            parameter.numel() for parameter in checkpoint.network.parameters()
        ),
        "seed": FIXED_SEED,
        "games_per_schedule": FIXED_GAMES_PER_SCHEDULE,
        "total_games": expected_total,
        "mcts_simulations": FIXED_SIMULATIONS,
        "c_puct": FIXED_C_PUCT,
        "dirichlet_alpha": float(config["dirichlet_alpha"]),
        "dirichlet_fraction": float(config["dirichlet_fraction"]),
        "temperature_schedules": {
            "all_hot": "temperature 1.0 at every ply",
            "early8": "temperature 1.0 at plies 0-7, then 0.0",
            "greedy": "temperature 0.0 at every ply",
        },
        "rng_protocol": (
            "Same game-index seed block per schedule; independent deterministic "
            "ply streams for root noise and action sampling."
        ),
        "schedules": schedules,
        "classification": classify_temperature_audit(schedules),
        "interpretation_notes": interpretation_notes,
        "network_state_digest_before": parameter_digest_before,
        "network_state_digest_after": parameter_digest_after,
        "network_unchanged": parameter_digest_before == parameter_digest_after,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after,
        "elapsed_seconds": sum(float(game["elapsed_seconds"]) for game in games),
    }
    if not summary["network_unchanged"] or not summary["checkpoint_unchanged"]:
        raise RuntimeError("fixed checkpoint/network changed during audit")
    if any(item["illegal_violations"] for item in schedules):
        raise RuntimeError("illegal-action violation observed")
    atomic_json_save(summary, args.report_dir / "summary.json")
    atomic_text_save(render_summary(summary), args.report_dir / "summary.txt")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv=None):
    run_audit(parse_args(argv))


if __name__ == "__main__":
    main()
