import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from evaluate_frozen_br import paired_bootstrap_delta
from evaluate_terminal_fix_audit import (
    BLUE0_MODEL,
    BLUE1_MODEL,
    CROSSPLAY_MASTER_SEED,
    EXPECTED_HASHES,
    RANDOM_MASTER_SEED,
    RED0_MODEL,
    RED1_MODEL,
    evaluate_crossplay,
    evaluate_vs_random,
    sha256_file,
    terminal_text,
)


RED2_MODEL = Path("models/MaskablePPO_CNN/red2_ft_vs_blue1_plus10k.zip")
DEFAULT_REPORT_DIR = Path("reports/red2_finetune_vs_blue1_20260822")
CHECKPOINT_COMMIT = "571dc8c0238a629a4255bce9f53191116c878ea6"
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_SAMPLES = 10_000
RED0_RANDOM_REFERENCE = 0.692
RED1_RANDOM_REFERENCE = 0.782
BASE_MODEL_PATHS = {
    "red0": RED0_MODEL,
    "blue0": BLUE0_MODEL,
    "red1": RED1_MODEL,
    "blue1": BLUE1_MODEL,
}


def validate_base_hashes():
    hashes = {name: sha256_file(path) for name, path in BASE_MODEL_PATHS.items()}
    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(
                f"{name} hash mismatch: expected {expected}, got {hashes[name]}"
            )
    return hashes


def match_row(label, result):
    return (
        f"{label} | {result['red_wins']} | {result['blue_wins']} | "
        f"{result['draws']} | {result['red_win_rate']:.3f} | "
        f"{result['red_suicides']} | {result['blue_suicides']} | "
        f"{result['mask_violations']} | {result['mean_game_length']:.3f} | "
        f"{terminal_text(result)}"
    )


def render_summary(summary):
    pre = summary["pre_red1_vs_blue1"]
    post = summary["post_red2_vs_blue1"]
    retention_old = summary["red1_vs_blue0"]
    retention_new = summary["red2_vs_blue0"]
    random_blue = summary["red2_vs_random_blue"]
    latest_ci = summary["latest_delta_bootstrap_95_ci"]
    retention_ci = summary["retention_delta_bootstrap_95_ci"]
    matrix = summary["crossplay"]["red_win_rate_matrix"]
    lines = [
        "# Red2 continuation vs Frozen Blue1",
        "",
        "Red1 was continued for one additional 10k request against stochastic frozen Blue1.",
        "Policy-vs-policy evaluations used stochastic masked sampling and paired seeds.",
        "No absolute policy ranking is inferred from generation numbers.",
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
            f"{summary['training']['red2_optimizer_state_entries']}"
        ),
        "",
        "## [Confirmed evaluations]",
        "",
        "Condition | Red wins | Blue wins | Draws | Red win rate | Red suicide | Blue suicide | Mask | Mean length | Terminal reasons",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---",
        match_row("PRE Red1 vs Blue1", pre),
        match_row("POST Red2 vs Blue1", post),
        match_row("Red1 vs Blue0", retention_old),
        match_row("Red2 vs Blue0", retention_new),
        match_row("Red2 vs Random Blue", random_blue),
        "",
        f"Latest-opponent delta: {summary['latest_delta']:+.3f}",
        f"Paired bootstrap 95% CI: [{latest_ci['low']:+.3f}, {latest_ci['high']:+.3f}]",
        f"Previous-opponent retention delta: {summary['retention_delta']:+.3f}",
        f"Retention paired bootstrap 95% CI: [{retention_ci['low']:+.3f}, {retention_ci['high']:+.3f}]",
        "",
        "## 2x2 generation cross-play (Red win rate)",
        "",
        " | Blue0 | Blue1",
        "--- | ---: | ---:",
        f"Red1 | {matrix['Red1']['Blue0']:.3f} | {matrix['Red1']['Blue1']:.3f}",
        f"Red2 | {matrix['Red2']['Blue0']:.3f} | {matrix['Red2']['Blue1']:.3f}",
        "",
        "## [Interpretation]",
        "",
        f"Latest-opponent adaptation: {summary['latest_opponent_adaptation']}",
        f"Forgetting/cycling evidence: {summary['forgetting_cycling_evidence']}",
        f"Next recommendation: {summary['next_recommendation']}",
        "",
        "## [Unconfirmed]",
        "",
        "Later alternating behavior, opponent-pool benefit, absolute policy ranking,",
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
    pre = evaluate_crossplay(
        red_model_path=RED1_MODEL,
        blue_model_path=BLUE1_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="PRE red1_vs_blue1",
    )
    summary = {
        "status": "baseline_complete",
        "experiment": "Red1 continuation vs frozen Blue1",
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
            "red2": str(RED2_MODEL),
        },
        "base_model_sha256_before": hashes,
        "pre_red1_vs_blue1": pre,
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
        raise FileExistsError("refusing to overwrite completed Red2 reports")
    if not RED2_MODEL.is_file():
        raise FileNotFoundError(RED2_MODEL)

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

    post = evaluate_crossplay(
        red_model_path=RED2_MODEL,
        blue_model_path=BLUE1_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="POST red2_vs_blue1",
    )
    retention_old = evaluate_crossplay(
        red_model_path=RED1_MODEL,
        blue_model_path=BLUE0_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="retention baseline red1_vs_blue0",
    )
    retention_new = evaluate_crossplay(
        red_model_path=RED2_MODEL,
        blue_model_path=BLUE0_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
        label="retention red2_vs_blue0",
    )
    random_blue = evaluate_vs_random(
        model_path=RED2_MODEL,
        agent_player=2,
        episodes=args.episodes,
        master_seed=args.random_seed,
        label="red2_vs_random_blue",
    )

    pre = summary["pre_red1_vs_blue1"]
    latest_delta = post["red_win_rate"] - pre["red_win_rate"]
    latest_ci = paired_bootstrap_delta(
        pre["red_win_indicators"], post["red_win_indicators"]
    )
    retention_delta = (
        retention_new["red_win_rate"] - retention_old["red_win_rate"]
    )
    retention_ci = paired_bootstrap_delta(
        retention_old["red_win_indicators"],
        retention_new["red_win_indicators"],
    )

    red1_model = MaskablePPO.load(RED1_MODEL, device="cpu")
    red2_model = MaskablePPO.load(RED2_MODEL, device="cpu")
    source_optimizer_entries = len(
        red1_model.policy.optimizer.state_dict()["state"]
    )
    red2_optimizer_entries = len(
        red2_model.policy.optimizer.state_dict()["state"]
    )
    red2_hash = sha256_file(RED2_MODEL)
    hashes_after = validate_base_hashes()
    all_results = [pre, post, retention_old, retention_new, random_blue]
    all_masks_clean = all(
        result["mask_violations"] == 0
        and result["nonterminal_all_false_masks"] == 0
        for result in all_results
    )
    infrastructure_pass = (
        source_optimizer_entries > 0
        and red2_optimizer_entries > 0
        and red2_model.num_timesteps > red1_model.num_timesteps
        and red2_hash != hashes_after["red1"]
        and hashes_before_post == hashes_after
        and all_masks_clean
    )

    if latest_delta > 0 and latest_ci["low"] > 0:
        latest_adaptation = "PASS: clear positive improvement against Blue1"
        latest_pass = True
    elif latest_delta > 0:
        latest_adaptation = "INCONCLUSIVE: positive estimate, CI includes zero"
        latest_pass = False
    else:
        latest_adaptation = "FAIL: no point-estimate improvement against Blue1"
        latest_pass = False

    if retention_delta < 0 and retention_ci["high"] < 0:
        forgetting_evidence = (
            "CLEAR: Red2 lost performance against Blue0; latest-only overfitting/cycling evidence"
        )
        clear_forgetting = True
    elif retention_delta < 0:
        forgetting_evidence = (
            "INCONCLUSIVE: lower Blue0 point estimate, but retention CI includes zero"
        )
        clear_forgetting = False
    else:
        forgetting_evidence = (
            "NONE in this matrix: Blue0 retention did not decrease in point estimate"
        )
        clear_forgetting = False

    if latest_pass and not clear_forgetting:
        recommendation = (
            "Run exactly one Blue2 latest-only continuation cycle before considering automation."
        )
    elif latest_pass and clear_forgetting:
        recommendation = (
            "Stop latest-only alternation and make opponent-pool training the next experiment."
        )
    else:
        recommendation = (
            "Stop continuation rounds and review the opponent training distribution."
        )

    crossplay = {
        "episodes_per_cell": args.episodes,
        "master_seed": args.master_seed,
        "red_win_rate_matrix": {
            "Red1": {
                "Blue0": retention_old["red_win_rate"],
                "Blue1": pre["red_win_rate"],
            },
            "Red2": {
                "Blue0": retention_new["red_win_rate"],
                "Blue1": post["red_win_rate"],
            },
        },
        "cells": {
            "red1_vs_blue0": retention_old,
            "red1_vs_blue1": pre,
            "red2_vs_blue0": retention_new,
            "red2_vs_blue1": post,
        },
    }
    summary.update(
        {
            "status": "complete",
            "post_red2_vs_blue1": post,
            "red1_vs_blue0": retention_old,
            "red2_vs_blue0": retention_new,
            "red2_vs_random_blue": random_blue,
            "latest_delta": latest_delta,
            "latest_delta_bootstrap_95_ci": latest_ci,
            "retention_delta": retention_delta,
            "retention_delta_bootstrap_95_ci": retention_ci,
            "random_references": {
                "red0_vs_random_blue": RED0_RANDOM_REFERENCE,
                "red1_vs_random_blue": RED1_RANDOM_REFERENCE,
                "red2_minus_red1": (
                    random_blue["red_win_rate"] - RED1_RANDOM_REFERENCE
                ),
            },
            "training": {
                "continuation": True,
                "reset_num_timesteps": False,
                "requested_additional_timesteps": 10_000,
                "start_num_timesteps": red1_model.num_timesteps,
                "end_num_timesteps": red2_model.num_timesteps,
                "actual_counter_increase": (
                    red2_model.num_timesteps - red1_model.num_timesteps
                ),
                "start_n_updates": red1_model._n_updates,
                "end_n_updates": red2_model._n_updates,
                "source_optimizer_state_entries": source_optimizer_entries,
                "red2_optimizer_state_entries": red2_optimizer_entries,
            },
            "model_sha256": {
                "base_before": summary["base_model_sha256_before"],
                "base_after": hashes_after,
                "base_unchanged": hashes_before_post == hashes_after,
                "red2": red2_hash,
                "red1_and_red2_differ": red2_hash != hashes_after["red1"],
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
                "later alternating behavior",
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
