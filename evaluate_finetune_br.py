import argparse
import hashlib
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from evaluate_frozen_br import (
    BLUE_MODEL,
    OLD_RED_MODEL,
    POLICY_MASTER_SEED,
    RANDOM_BLUE_SEED,
    paired_bootstrap_delta,
    read_old_random_reference,
    run_new_red_vs_random_blue,
    run_policy_match,
)


FT_RED_MODEL = Path(
    "models/MaskablePPO_CNN/red10k_ft_vs_blue10k_plus10k.zip"
)
DEFAULT_REPORT_DIR = Path("reports/red10k_finetune_vs_blue10k_20260822")
EXPECTED_OLD_RED_SHA256 = (
    "e28340c33406a333940df1fe94eee39b9f78494c4b2b2886cd565f27de27c944"
)
EXPECTED_BLUE_SHA256 = (
    "17f990f29d0f2f6ae09c386561cb700210dc267820eed5f828565d7ffd9992ba"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_match_row(label, result):
    return (
        f"{label} | {result['episodes']} | {result['red_wins']} | "
        f"{result['blue_wins']} | {result['draws']} | "
        f"{result['red_win_rate']:.3f} | {result['red_suicide_rate']:.3f} | "
        f"{result['blue_suicide_rate']:.3f} | {result['mask_violations']} | "
        f"{result['mean_game_length']:.2f}"
    )


def render_summary(summary):
    old = summary["old_red10k_vs_frozen_blue10k"]
    fine_tuned = summary["ft_red_vs_frozen_blue10k"]
    random_blue = summary["ft_red_vs_random_blue"]
    interval = summary["win_rate_delta_bootstrap_95_ci"]
    learning_result = summary["learning_result"]
    lines = [
        "# Red10k continuation fine-tuning vs Frozen Blue10k",
        "",
        "OLD Red10k was loaded with its optimizer/checkpoint state and fine-tuned.",
        "Frozen Blue sampled with deterministic=False and received action masks.",
        "This is one continuation experiment, not alternating self-play.",
        "",
        "## [Infrastructure PASS]",
        "",
        f"Infrastructure pass: {str(summary['infrastructure_pass']).lower()}",
        (
            "Training timestep counter: "
            f"{summary['training']['start_num_timesteps']} -> "
            f"{summary['training']['end_num_timesteps']}"
        ),
        (
            "Training update counter: "
            f"{summary['training']['start_n_updates']} -> "
            f"{summary['training']['end_n_updates']}"
        ),
        (
            "Restored optimizer state entries: "
            f"{summary['training']['optimizer_state_entries']}"
        ),
        "",
        "## [Learning result]",
        "",
        (
            "condition | episodes | Red wins | Blue wins | draws | Red win_rate | "
            "Red suicide_rate | Blue suicide_rate | mask violations | mean game length"
        ),
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
        format_match_row("OLD Red10k vs Frozen Blue10k", old),
        format_match_row("FT Red vs Frozen Blue10k", fine_tuned),
        format_match_row("FT Red vs Random Blue", random_blue),
        "",
        f"Frozen-Blue win-rate delta: {summary['win_rate_delta']:+.3f}",
        (
            "Paired bootstrap 95% CI: "
            f"[{interval['low']:+.3f}, {interval['high']:+.3f}]"
        ),
        f"Learning result: {learning_result}",
        (
            "FT vs Random Blue change from OLD reference: "
            f"{summary['random_blue_reference_delta']:+.3f}"
        ),
        "",
        "## [미확정]",
        "",
        "- Alternating self-play performance",
        "- Full self-play convergence",
        "- Human-opponent strength",
        "- Nash equilibrium behavior",
        "- Whether corner openings are optimal",
        "",
    ]
    return "\n".join(lines)


def validate_original_hashes():
    old_hash = sha256_file(OLD_RED_MODEL)
    blue_hash = sha256_file(BLUE_MODEL)
    if old_hash != EXPECTED_OLD_RED_SHA256:
        raise RuntimeError(f"OLD Red hash mismatch: {old_hash}")
    if blue_hash != EXPECTED_BLUE_SHA256:
        raise RuntimeError(f"Frozen Blue hash mismatch: {blue_hash}")
    return old_hash, blue_hash


def run_baseline(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    existing = [path for path in (json_path, text_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing reports: {existing}")
    old_hash, blue_hash = validate_original_hashes()

    old_result, _ = run_policy_match(
        red_model_path=OLD_RED_MODEL,
        blue_model_path=BLUE_MODEL,
        episodes=args.episodes,
        deterministic=False,
        master_seed=args.master_seed,
    )
    partial_summary = {
        "status": "baseline_complete",
        "experiment": "Red10k continuation fine-tuning vs Frozen Blue10k",
        "scope": "One fine-tuning experiment; not alternating self-play.",
        "configuration": {
            "episodes_per_condition": args.episodes,
            "policy_master_seed": args.master_seed,
            "old_red_model_path": str(OLD_RED_MODEL),
            "frozen_blue_model_path": str(BLUE_MODEL),
            "ft_red_model_path": str(FT_RED_MODEL),
            "red_evaluation_deterministic": False,
            "blue_evaluation_deterministic": False,
            "frozen_training_opponent_deterministic": False,
            "action_masks_both_policies": True,
        },
        "model_sha256": {
            "old_red_before": old_hash,
            "frozen_blue_before": blue_hash,
        },
        "old_red10k_vs_frozen_blue10k": old_result,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(partial_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved baseline checkpoint {json_path}", flush=True)


def run_post_train(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    if not json_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {json_path}")
    if text_path.exists():
        raise FileExistsError(f"refusing to overwrite completed report: {text_path}")
    if not FT_RED_MODEL.is_file():
        raise FileNotFoundError(FT_RED_MODEL)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "baseline_complete":
        raise ValueError("summary.json is not a baseline-only checkpoint")
    configuration = summary["configuration"]
    if (
        configuration["episodes_per_condition"] != args.episodes
        or configuration["policy_master_seed"] != args.master_seed
    ):
        raise ValueError("post-training settings do not match the baseline")

    old_hash, blue_hash = validate_original_hashes()
    if summary["model_sha256"]["old_red_before"] != old_hash:
        raise RuntimeError("OLD Red changed after baseline evaluation")
    if summary["model_sha256"]["frozen_blue_before"] != blue_hash:
        raise RuntimeError("Frozen Blue changed after baseline evaluation")

    fine_tuned_result, _ = run_policy_match(
        red_model_path=FT_RED_MODEL,
        blue_model_path=BLUE_MODEL,
        episodes=args.episodes,
        deterministic=False,
        master_seed=args.master_seed,
    )
    random_blue_result = run_new_red_vs_random_blue(
        red_model_path=FT_RED_MODEL,
        episodes=args.episodes,
        seed=RANDOM_BLUE_SEED,
    )

    old_result = summary["old_red10k_vs_frozen_blue10k"]
    delta = fine_tuned_result["red_win_rate"] - old_result["red_win_rate"]
    interval = paired_bootstrap_delta(
        old_result["red_win_indicators"],
        fine_tuned_result["red_win_indicators"],
    )
    old_random_reference, reference_path = read_old_random_reference()

    old_model = MaskablePPO.load(OLD_RED_MODEL, device="cpu")
    fine_tuned_model = MaskablePPO.load(FT_RED_MODEL, device="cpu")
    old_optimizer_entries = len(old_model.policy.optimizer.state_dict()["state"])
    ft_optimizer_entries = len(
        fine_tuned_model.policy.optimizer.state_dict()["state"]
    )
    ft_hash = sha256_file(FT_RED_MODEL)
    hashes_differ = ft_hash != old_hash
    timestep_increased = fine_tuned_model.num_timesteps > old_model.num_timesteps
    all_mask_violations = sum(
        result["mask_violations"]
        for result in (old_result, fine_tuned_result, random_blue_result)
    )
    infrastructure_pass = (
        old_optimizer_entries > 0
        and ft_optimizer_entries > 0
        and timestep_increased
        and hashes_differ
        and all_mask_violations == 0
        and sha256_file(BLUE_MODEL) == blue_hash
    )
    if delta > 0 and interval["low"] > 0:
        learning_result = "PASS: clear improvement"
    elif delta > 0:
        learning_result = "INCONCLUSIVE: positive point estimate, CI includes zero"
    else:
        learning_result = "FAIL: no point-estimate improvement"

    summary.update(
        {
            "status": "complete",
            "ft_red_vs_frozen_blue10k": fine_tuned_result,
            "ft_red_vs_random_blue": random_blue_result,
            "win_rate_delta": delta,
            "win_rate_delta_bootstrap_95_ci": interval,
            "old_red10k_vs_random_blue_reference": old_random_reference,
            "old_random_blue_reference_path": reference_path,
            "random_blue_reference_delta": (
                random_blue_result["red_win_rate"] - old_random_reference
            ),
            "training": {
                "continuation": True,
                "reset_num_timesteps": False,
                "requested_additional_timesteps": 10_000,
                "start_num_timesteps": old_model.num_timesteps,
                "end_num_timesteps": fine_tuned_model.num_timesteps,
                "actual_counter_increase": (
                    fine_tuned_model.num_timesteps - old_model.num_timesteps
                ),
                "start_n_updates": old_model._n_updates,
                "end_n_updates": fine_tuned_model._n_updates,
                "optimizer_state_entries": old_optimizer_entries,
                "ft_optimizer_state_entries": ft_optimizer_entries,
            },
            "model_sha256": {
                "old_red_before": old_hash,
                "old_red_after": sha256_file(OLD_RED_MODEL),
                "frozen_blue_before": blue_hash,
                "frozen_blue_after": sha256_file(BLUE_MODEL),
                "fine_tuned_red": ft_hash,
                "old_and_ft_differ": hashes_differ,
            },
            "all_mask_violations": all_mask_violations,
            "infrastructure_pass": infrastructure_pass,
            "learning_result": learning_result,
        }
    )
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(render_summary(summary), encoding="utf-8")
    print(f"OLD_to_FT_delta={delta:+.3f}", flush=True)
    print(
        f"bootstrap_95_ci=[{interval['low']:+.3f}, {interval['high']:+.3f}]",
        flush=True,
    )
    print(f"learning_result={learning_result}", flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {text_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("baseline", "post-train"):
        subparser = subparsers.add_parser(phase)
        subparser.add_argument("--episodes", type=int, default=500)
        subparser.add_argument("--master-seed", type=int, default=POLICY_MASTER_SEED)
        subparser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.phase == "baseline":
        run_baseline(args)
    else:
        run_post_train(args)


if __name__ == "__main__":
    main()
