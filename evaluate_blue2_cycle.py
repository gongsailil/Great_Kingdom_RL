import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from evaluate_frozen_br import paired_bootstrap_delta
from evaluate_terminal_fix_audit import (
    BLUE0_MODEL,
    BLUE1_MODEL,
    CROSSPLAY_MASTER_SEED,
    RANDOM_MASTER_SEED,
    RED0_MODEL,
    RED1_MODEL,
    evaluate_blue_crossplay,
    evaluate_vs_random,
    sha256_file,
    terminal_text,
)


RED2_MODEL = Path("models/MaskablePPO_CNN/red2_ft_vs_blue1_plus10k.zip")
BLUE2_MODEL = Path("models/MaskablePPO_CNN/blue2_ft_vs_red2_plus10k.zip")
DEFAULT_REPORT_DIR = Path("reports/blue2_finetune_vs_red2_20260822")
CHECKPOINT_COMMIT = "4b3edf7a1470d15bbf9d81b578b722f084b976ab"
BLUE0_RANDOM_REFERENCE = 0.768
BLUE1_RANDOM_REFERENCE = 0.814
EXPECTED_BASE_HASHES = {
    "red0": "e28340c33406a333940df1fe94eee39b9f78494c4b2b2886cd565f27de27c944",
    "blue0": "17f990f29d0f2f6ae09c386561cb700210dc267820eed5f828565d7ffd9992ba",
    "red1": "5dcc2d8b016f55ddc4e2c9abd7c7e860a485e8fd105399b0be7e6ef037fe8883",
    "blue1": "4b1bdc51904d8b471c576da57df2019e1ba46dd31a069e3db84bd879d18c3e44",
    "red2": "f99ce74b54a45039a3fb63be8400795ea5820b247b0eed197976eb0ba2fc66ac",
}
BASE_MODEL_PATHS = {
    "red0": RED0_MODEL,
    "blue0": BLUE0_MODEL,
    "red1": RED1_MODEL,
    "blue1": BLUE1_MODEL,
    "red2": RED2_MODEL,
}


def validate_base_hashes():
    hashes = {name: sha256_file(path) for name, path in BASE_MODEL_PATHS.items()}
    for name, expected in EXPECTED_BASE_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(
                f"{name} hash mismatch: expected {expected}, got {hashes[name]}"
            )
    return hashes


def match_row(label, result):
    return (
        f"{label} | {result['blue_wins']} | {result['red_wins']} | "
        f"{result['draws']} | {result['blue_win_rate']:.3f} | "
        f"{result['blue_suicides']} | {result['red_suicides']} | "
        f"{result['mask_violations']} | {result['mean_game_length']:.3f} | "
        f"{terminal_text(result)}"
    )


def render_summary(summary):
    pre = summary["pre_blue1_vs_red2"]
    post = summary["post_blue2_vs_red2"]
    retention_old = summary["blue1_vs_red1"]
    retention_new = summary["blue2_vs_red1"]
    random_red = summary["blue2_vs_random_red"]
    latest_ci = summary["latest_delta_bootstrap_95_ci"]
    retention_ci = summary["retention_delta_bootstrap_95_ci"]
    matrix = summary["crossplay"]["red_win_rate_matrix"]
    lines = [
        "# Blue2 continuation vs Frozen Red2",
        "",
        "Blue1 was continued for one additional 10k request against stochastic frozen Red2.",
        "Policy-vs-policy evaluations used stochastic masked sampling and paired seeds.",
        "Generation numbers are not treated as an absolute policy ranking.",
        "",
        "## [Infrastructure]",
        "",
        f"Checkpoint commit: {summary['checkpoint_commit']}",
        f"Infrastructure pass: {str(summary['infrastructure_pass']).lower()}",
        (
            "Timestep counter: "
            f"{summary['training']['start_num_timesteps']} -> "
            f"{summary['training']['end_num_timesteps']}"
        ),
        (
            "Update counter: "
            f"{summary['training']['start_n_updates']} -> "
            f"{summary['training']['end_n_updates']}"
        ),
        (
            "Optimizer state entries: "
            f"{summary['training']['source_optimizer_state_entries']} -> "
            f"{summary['training']['blue2_optimizer_state_entries']}"
        ),
        "",
        "## [Confirmed evaluations]",
        "",
        "Condition | Blue wins | Red wins | Draws | Blue win rate | Blue suicide | Red suicide | Mask | Mean length | Terminal reasons",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---",
        match_row("PRE Blue1 vs Red2", pre),
        match_row("POST Blue2 vs Red2", post),
        match_row("Blue1 vs Red1", retention_old),
        match_row("Blue2 vs Red1", retention_new),
        match_row("Blue2 vs Random Red", random_red),
        "",
        f"Latest-opponent delta: {summary['latest_delta']:+.3f}",
        f"Paired bootstrap 95% CI: [{latest_ci['low']:+.3f}, {latest_ci['high']:+.3f}]",
        f"Previous-opponent retention delta: {summary['retention_delta']:+.3f}",
        f"Retention paired bootstrap 95% CI: [{retention_ci['low']:+.3f}, {retention_ci['high']:+.3f}]",
        "",
        "## 2x2 generation cross-play (Red win rate)",
        "",
        " | Blue1 | Blue2",
        "--- | ---: | ---:",
        f"Red1 | {matrix['Red1']['Blue1']:.3f} | {matrix['Red1']['Blue2']:.3f}",
        f"Red2 | {matrix['Red2']['Blue1']:.3f} | {matrix['Red2']['Blue2']:.3f}",
        "",
        "## [Interpretation]",
        "",
        f"Latest-opponent adaptation: {summary['latest_opponent_adaptation']}",
        f"Forgetting/cycling evidence: {summary['forgetting_cycling_evidence']}",
        f"Next recommendation: {summary['next_recommendation']}",
        "",
        "## [Unconfirmed]",
        "",
        "Absolute policy ranking, automated alternating behavior, opponent-pool benefit,",
        "self-play convergence, human-play strength, and Nash behavior remain unconfirmed.",
        "",
    ]
    return "\n".join(lines)


def run_baseline(args):
    report_dir = Path(args.report_dir)
    if report_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing report directory: {report_dir}"
        )
    hashes = validate_base_hashes()
    pre = evaluate_blue_crossplay(
        blue_model_path=BLUE1_MODEL,
        red_model_path=RED2_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="PRE blue1_vs_red2",
    )
    summary = {
        "status": "baseline_complete",
        "experiment": "Blue1 continuation vs frozen Red2",
        "checkpoint_commit": CHECKPOINT_COMMIT,
        "configuration": {
            "episodes_per_condition": args.episodes,
            "policy_master_seed": args.master_seed,
            "random_master_seed": args.random_seed,
            "policy_evaluation_deterministic": False,
            "frozen_training_opponent_deterministic": False,
            "action_masks_both_policies": True,
        },
        "model_paths": {
            **{name: str(path) for name, path in BASE_MODEL_PATHS.items()},
            "blue2": str(BLUE2_MODEL),
        },
        "base_model_sha256_before": hashes,
        "pre_blue1_vs_red2": pre,
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved PRE checkpoint {report_dir / 'summary.json'}", flush=True)


def run_post_train(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    matrix_path = report_dir / "crossplay_matrix.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"PRE checkpoint not found: {json_path}")
    if text_path.exists() or matrix_path.exists():
        raise FileExistsError("refusing to overwrite completed Blue2 reports")
    if not BLUE2_MODEL.is_file():
        raise FileNotFoundError(BLUE2_MODEL)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "baseline_complete":
        raise ValueError("summary.json is not a PRE-only checkpoint")
    config = summary["configuration"]
    if (
        config["episodes_per_condition"] != args.episodes
        or config["policy_master_seed"] != args.master_seed
        or config["random_master_seed"] != args.random_seed
    ):
        raise ValueError("POST settings do not match PRE")

    hashes_before_post = validate_base_hashes()
    if summary["base_model_sha256_before"] != hashes_before_post:
        raise RuntimeError("a base model changed after PRE")

    post = evaluate_blue_crossplay(
        blue_model_path=BLUE2_MODEL,
        red_model_path=RED2_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="POST blue2_vs_red2",
    )
    retention_old = evaluate_blue_crossplay(
        blue_model_path=BLUE1_MODEL,
        red_model_path=RED1_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="retention baseline blue1_vs_red1",
    )
    retention_new = evaluate_blue_crossplay(
        blue_model_path=BLUE2_MODEL,
        red_model_path=RED1_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="retention blue2_vs_red1",
    )
    random_red = evaluate_vs_random(
        model_path=BLUE2_MODEL,
        agent_player=1,
        episodes=args.episodes,
        master_seed=args.random_seed,
        label="blue2_vs_random_red",
    )

    pre = summary["pre_blue1_vs_red2"]
    latest_delta = post["blue_win_rate"] - pre["blue_win_rate"]
    latest_ci = paired_bootstrap_delta(
        pre["blue_win_indicators"], post["blue_win_indicators"]
    )
    retention_delta = (
        retention_new["blue_win_rate"] - retention_old["blue_win_rate"]
    )
    retention_ci = paired_bootstrap_delta(
        retention_old["blue_win_indicators"],
        retention_new["blue_win_indicators"],
    )

    blue1_model = MaskablePPO.load(BLUE1_MODEL, device="cpu")
    blue2_model = MaskablePPO.load(BLUE2_MODEL, device="cpu")
    source_optimizer_entries = len(
        blue1_model.policy.optimizer.state_dict()["state"]
    )
    blue2_optimizer_entries = len(
        blue2_model.policy.optimizer.state_dict()["state"]
    )
    blue2_hash = sha256_file(BLUE2_MODEL)
    hashes_after = validate_base_hashes()
    all_results = [pre, post, retention_old, retention_new, random_red]
    all_masks_clean = all(
        result["mask_violations"] == 0
        and result["nonterminal_all_false_masks"] == 0
        for result in all_results
    )
    infrastructure_pass = (
        source_optimizer_entries > 0
        and blue2_optimizer_entries > 0
        and blue2_model.num_timesteps > blue1_model.num_timesteps
        and blue2_hash != hashes_after["blue1"]
        and hashes_before_post == hashes_after
        and all_masks_clean
    )

    if latest_delta > 0 and latest_ci["low"] > 0:
        latest_adaptation = "PASS: clear positive improvement against Red2"
        latest_pass = True
    elif latest_delta > 0:
        latest_adaptation = "INCONCLUSIVE: positive estimate, CI includes zero"
        latest_pass = False
    else:
        latest_adaptation = "FAIL: no point-estimate improvement against Red2"
        latest_pass = False

    if retention_delta < 0 and retention_ci["high"] < 0:
        forgetting_evidence = (
            "CLEAR: Blue2 lost performance against Red1; latest-only overfitting/cycling evidence"
        )
        clear_forgetting = True
    elif retention_delta < 0:
        forgetting_evidence = (
            "INCONCLUSIVE: lower Red1 point estimate, but retention CI includes zero"
        )
        clear_forgetting = False
    else:
        forgetting_evidence = (
            "NONE in this matrix: Red1 retention did not decrease in point estimate"
        )
        clear_forgetting = False

    if latest_pass and not clear_forgetting:
        recommendation = (
            "Design a minimal alternating self-play runner; do not launch Red3 manually."
        )
    elif latest_pass and clear_forgetting:
        recommendation = (
            "Do not automate latest-only alternation; evaluate opponent-pool training next."
        )
    else:
        recommendation = (
            "Stop continuation rounds and review the latest-only training distribution."
        )

    crossplay = {
        "episodes_per_cell": args.episodes,
        "master_seed": args.master_seed,
        "red_win_rate_matrix": {
            "Red1": {
                "Blue1": retention_old["red_win_rate"],
                "Blue2": retention_new["red_win_rate"],
            },
            "Red2": {
                "Blue1": pre["red_win_rate"],
                "Blue2": post["red_win_rate"],
            },
        },
        "blue_win_rate_matrix": {
            "Red1": {
                "Blue1": retention_old["blue_win_rate"],
                "Blue2": retention_new["blue_win_rate"],
            },
            "Red2": {
                "Blue1": pre["blue_win_rate"],
                "Blue2": post["blue_win_rate"],
            },
        },
        "cells": {
            "red1_vs_blue1": retention_old,
            "red1_vs_blue2": retention_new,
            "red2_vs_blue1": pre,
            "red2_vs_blue2": post,
        },
    }
    summary.update(
        {
            "status": "complete",
            "post_blue2_vs_red2": post,
            "blue1_vs_red1": retention_old,
            "blue2_vs_red1": retention_new,
            "blue2_vs_random_red": random_red,
            "latest_delta": latest_delta,
            "latest_delta_bootstrap_95_ci": latest_ci,
            "retention_delta": retention_delta,
            "retention_delta_bootstrap_95_ci": retention_ci,
            "random_references": {
                "blue0_vs_random_red": BLUE0_RANDOM_REFERENCE,
                "blue1_vs_random_red": BLUE1_RANDOM_REFERENCE,
                "blue2_minus_blue1": (
                    random_red["blue_win_rate"] - BLUE1_RANDOM_REFERENCE
                ),
            },
            "training": {
                "continuation": True,
                "reset_num_timesteps": False,
                "requested_additional_timesteps": 10_000,
                "start_num_timesteps": blue1_model.num_timesteps,
                "end_num_timesteps": blue2_model.num_timesteps,
                "actual_counter_increase": (
                    blue2_model.num_timesteps - blue1_model.num_timesteps
                ),
                "start_n_updates": blue1_model._n_updates,
                "end_n_updates": blue2_model._n_updates,
                "source_optimizer_state_entries": source_optimizer_entries,
                "blue2_optimizer_state_entries": blue2_optimizer_entries,
            },
            "model_sha256": {
                "base_before": summary["base_model_sha256_before"],
                "base_after": hashes_after,
                "base_unchanged": hashes_before_post == hashes_after,
                "blue2": blue2_hash,
                "blue1_and_blue2_differ": blue2_hash != hashes_after["blue1"],
            },
            "crossplay": crossplay,
            "all_masks_clean": all_masks_clean,
            "infrastructure_pass": infrastructure_pass,
            "latest_opponent_adaptation": latest_adaptation,
            "latest_opponent_adaptation_pass": latest_pass,
            "forgetting_cycling_evidence": forgetting_evidence,
            "clear_forgetting_cycling_evidence": clear_forgetting,
            "next_recommendation": recommendation,
            "unconfirmed": [
                "absolute policy ranking",
                "automated alternating behavior",
                "opponent-pool benefit",
                "self-play convergence",
                "human-play strength",
                "Nash behavior",
            ],
        }
    )
    matrix_path.write_text(
        json.dumps(crossplay, indent=2) + "\n", encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    text_path.write_text(render_summary(summary), encoding="utf-8")
    print(f"latest_delta={latest_delta:+.3f}", flush=True)
    print(
        f"latest_bootstrap_95_ci=[{latest_ci['low']:+.3f}, {latest_ci['high']:+.3f}]",
        flush=True,
    )
    print(f"retention_delta={retention_delta:+.3f}", flush=True)
    print(
        f"retention_bootstrap_95_ci=[{retention_ci['low']:+.3f}, {retention_ci['high']:+.3f}]",
        flush=True,
    )
    print(f"latest_adaptation={latest_adaptation}", flush=True)
    print(f"forgetting_cycling={forgetting_evidence}", flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {text_path}", flush=True)
    print(f"Saved {matrix_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("baseline", "post-train"):
        subparser = subparsers.add_parser(phase)
        subparser.add_argument("--episodes", type=int, default=500)
        subparser.add_argument("--master-seed", type=int, default=CROSSPLAY_MASTER_SEED)
        subparser.add_argument("--random-seed", type=int, default=RANDOM_MASTER_SEED)
        subparser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.phase == "baseline":
        run_baseline(args)
    else:
        run_post_train(args)


if __name__ == "__main__":
    main()
