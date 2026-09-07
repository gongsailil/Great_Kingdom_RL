"""Expanded acceptance arena; fixed checkpoints, no training or value ablation."""

import argparse
import json
import os
from pathlib import Path
import time

from alphazero_v2.evaluate import load_evaluation_checkpoint
from alphazero_v2.encoder import encode_state as encode_v2
from alphazero_v2.training_runner import _atomic_json_save
from alphazero_v3.encoder import encode_state as encode_v3
from alphazero_v3.temperature_audit import network_state_digest
from alphazero_v4.acceptance import (
    CLASSIFICATION_POLICY, SEED, classify_acceptance, file_digest,
    generate_acceptance_openings, json_digest, play_acceptance_game,
    run_strategic_suite, summarize_matchup, validate_acceptance_openings,
)
from alphazero_v4.evaluation import load_v4_evaluation_checkpoint


V4_PATH = Path("runs/alphazero_v4/stability_20260903/latest.pt")
V3_PATH = Path("runs/alphazero_v3/territory_pilot_20260901/checkpoints/iteration_000050.pt")
V2_PATHS = (
    Path("runs/alphazero_v2/main_20260830/checkpoints/iteration_000375.pt"),
    Path("runs/alphazero_v2/main_20260830/latest.pt"),
)
REPORT_DIR = Path("reports/alphazero_v4_acceptance_20260906")


def render_summary(summary):
    lines = [
        "AlphaZero V4 acceptance, seed 20260906",
        f"Automatic result: {summary['classification']}: {summary['reason']}",
        "Final project ACCEPT/REJECT: PENDING Human Blue 3 + Human Red 3 completed games.",
        "V4 MCTS256/c_puct1.5/no noise/temp0/exact tactical solver ON.",
        "V3 MCTS256; V2 MCTS64 (saved historical budget); historical solvers unchanged.",
        "50 unique legal opening positions, 2..6 placements, paired color swap.",
        "Bootstrap: 10,000 opening-level resamples, seed 20260906.",
        "Difference = V4 win rate minus opponent win rate; CI describes this opening suite.",
        "V2 comparison is a historical-agent comparison, not an equal-compute ablation.",
        "Operational classification thresholds were fixed before arena; see summary.json.",
        "",
    ]
    for name, m in summary["matchups"].items():
        boot = m["paired_bootstrap"]
        b = m["v4_behavior"]
        lines += [
            f"V4 vs {name}: {m['v4_wins']}-{m['opponent_wins']} / {m['games']} games",
            f"Win rate {boot['v4_win_rate']:.1%}, paired 95% CI {boot['v4_win_rate_ci95']}",
            f"Win-rate difference {boot['win_rate_difference_v4_minus_opponent']:+.3f}, paired 95% CI {boot['difference_ci95']}",
            f"V4 Blue: {m['v4_by_color']['1']}; Red: {m['v4_by_color']['2']}",
            f"Color Blue/Red wins: {m['blue_wins']}/{m['red_wins']}",
            f"Capture/score endings {m['capture_endings']}/{m['pass_score_endings']}; PASS actions {m['pass_actions']}; mean length {m['mean_game_length']:.2f}",
            f"V4 territory states {b['states_with_nonzero_territory']}/{b['states']} ({b['nonzero_territory_fraction']:.1%}); games reaching territory {b['games_reaching_nonzero_territory']}; V4 PASS actions {b['pass_actions']}",
            f"V4 tactical modes: {b['tactical_modes']}",
            f"Scoring distributions: {m['territory_score_distribution']}", "",
        ]
    lines += ["Strategic suite (D/E/F/G are diagnostic only):"]
    for s in summary["strategic_suite"]:
        lines.append(f"{s['scenario']}: action={s['selection']['action']} mode={s['selection']['tactical_mode']} success={s['success']} territory delta={s['territory_delta']} allows immediate loss={s['allows_immediate_tactical_loss']}")
    lines += [
        "", "Capture endings alone are not a failure criterion. Territory counts show exposure, not proof of strategic understanding.",
        f"Network/checkpoint unchanged: {summary['network_unchanged']}/{summary['checkpoint_files_unchanged']}",
        f"Elapsed seconds: {summary['elapsed_seconds']:.3f}",
        "Human instructions: docs/V4_HUMAN_ACCEPTANCE.md", "",
    ]
    return "\n".join(lines)


def run(args):
    started = time.perf_counter()
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise FileExistsError("acceptance report directory is not empty; preserving existing results")
    v4 = load_v4_evaluation_checkpoint(V4_PATH, args.device, expected_iteration=50)
    v3 = load_evaluation_checkpoint(V3_PATH, v4.device, expected_iteration=50, state_encoder=encode_v3)
    opponents = {"v3_iter50": v3}
    v2_status = "missing: neither historical candidate path exists"
    for path in V2_PATHS:
        if path.is_file():
            try:
                opponents["v2_iter375"] = load_evaluation_checkpoint(path, v4.device, expected_iteration=375, state_encoder=encode_v2)
            except ValueError as error:
                v2_status = str(error)
                continue
            v2_status = f"loaded {path}"
            break
    agents = {"v4_iter50": v4, **opponents}
    if v4.config.mcts_simulations != 256 or v4.config.c_puct != 1.5:
        raise ValueError("unexpected V4 acceptance checkpoint search config")
    if any(c.config["c_puct"] != 1.5 for c in opponents.values()):
        raise ValueError("unexpected historical c_puct; preserving experiment config")
    network_before = {name: network_state_digest(c.network) for name, c in agents.items()}
    files_before = {name: file_digest(c.path) for name, c in agents.items()}
    suite = generate_acceptance_openings()
    validate_acceptance_openings(suite)
    opening_hash = json_digest(suite)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(suite, args.report_dir / "openings.json")
    strategic = run_strategic_suite(v4)
    _atomic_json_save(strategic, args.report_dir / "strategic_suite.json")
    print("strategic suite complete; starting paired arena", flush=True)
    all_games = []
    try:
        with (args.report_dir / "arena_games.jsonl").open("x", encoding="utf-8") as handle:
            for name, opponent in opponents.items():
                for opening in suite["openings"]:
                    for color in (1, 2):
                        game = play_acceptance_game(v4, opponent, opening, color, name)
                        game["opening_suite_sha256"] = opening_hash
                        all_games.append(game)
                        handle.write(json.dumps(game, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        played = [g for g in all_games if g["matchup"] == name]
                        wins = sum(g["winner_agent"] == "v4_iter50" for g in played)
                        print(f"{name}: {len(played)}/100 V4 wins={wins} latest={game['winner_agent']} {game['terminal_reason']}", flush=True)
        if json_digest(suite) != opening_hash:
            raise RuntimeError("opening suite mutated")
        if any(network_state_digest(c.network) != network_before[n] for n, c in agents.items()):
            raise RuntimeError("network parameters/buffers mutated")
        if any(file_digest(c.path) != files_before[n] for n, c in agents.items()):
            raise RuntimeError("checkpoint file changed")
    except Exception as error:
        _atomic_json_save({"classification": "AUTO_FAIL", "completed": False,
                           "reason": repr(error), "games_completed": len(all_games)},
                          args.report_dir / "summary.json")
        raise
    matchups = {name: summarize_matchup([g for g in all_games if g["matchup"] == name]) for name in opponents}
    classification, reason = classify_acceptance(matchups, strategic)
    summary = {
        "classification": classification, "reason": reason,
        "final_project_acceptance": "PENDING_HUMAN_6_GAMES",
        "classification_policy": CLASSIFICATION_POLICY,
        "checkpoint_metadata": {
            name: {"path": str(c.path), "iteration": c.iteration,
                   "file_sha256": files_before[name], "network_sha256": network_before[name],
                   "mcts_simulations": c.config.mcts_simulations if name == "v4_iter50" else c.config["mcts_simulations"]}
            for name, c in agents.items()
        },
        "v2_status": v2_status, "device": str(v4.device),
        "opening_suite_sha256": opening_hash, "seed": SEED,
        "network_unchanged": True, "checkpoint_files_unchanged": True,
        "root_noise": False, "temperature": 0.0, "c_puct": 1.5,
        "matchups": matchups, "strategic_suite": strategic,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json_save(summary, args.report_dir / "summary.json")
    (args.report_dir / "summary.txt").write_text(render_summary(summary), encoding="utf-8")
    print(f"{classification}: {reason}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
