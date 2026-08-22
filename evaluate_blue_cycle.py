import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from evaluate_frozen_br import paired_bootstrap_delta, run_policy_match
from frozen_policy_env import FrozenPolicyOpponentEnv, predict_with_local_rng
from gk_env import GreatKingdomEnv


RED0_MODEL = Path("models/MaskablePPO_CNN/masked_ppo_10000.zip")
BLUE0_MODEL = Path("models/MaskablePPO_CNN/blue_masked_ppo_10000.zip")
RED1_MODEL = Path(
    "models/MaskablePPO_CNN/red10k_ft_vs_blue10k_plus10k.zip"
)
BLUE1_MODEL = Path(
    "models/MaskablePPO_CNN/blue10k_ft_vs_red1_plus10k.zip"
)
DEFAULT_REPORT_DIR = Path("reports/blue10k_finetune_vs_red1_20260822")
MASTER_SEED = 20260822
RANDOM_RED_SEED = 30000
EXPECTED_HASHES = {
    "red0": "e28340c33406a333940df1fe94eee39b9f78494c4b2b2886cd565f27de27c944",
    "blue0": "17f990f29d0f2f6ae09c386561cb700210dc267820eed5f828565d7ffd9992ba",
    "red1": "5dcc2d8b016f55ddc4e2c9abd7c7e860a485e8fd105399b0be7e6ef037fe8883",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_base_hashes():
    paths = {"red0": RED0_MODEL, "blue0": BLUE0_MODEL, "red1": RED1_MODEL}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(f"{name} hash mismatch: {hashes[name]}")
    return hashes


def run_blue_match(
    *,
    blue_model_path,
    red_model_path,
    episodes,
    master_seed,
):
    blue_model = MaskablePPO.load(blue_model_path, device="cpu")
    env = FrozenPolicyOpponentEnv(
        agent_player=1,
        opponent_model_path=red_model_path,
        opponent_deterministic=False,
        opponent_seed=master_seed + 100_000,
    )
    blue_wins = 0
    red_wins = 0
    draws = 0
    blue_suicides = 0
    red_suicides = 0
    mask_violations = 0
    game_lengths = []
    blue_openings = Counter()
    blue_win_indicators = []

    print(
        f"\n=== Blue policy vs frozen Red policy: stochastic, {episodes} games ===",
        flush=True,
    )
    for episode in range(episodes):
        obs, _ = env.reset(seed=master_seed + episode)
        if env.logic.turn != 1 or env.move_trace:
            raise AssertionError("Blue learner did not receive the opening turn")
        blue_rng = np.random.default_rng(
            np.random.SeedSequence([master_seed, episode, 1])
        )
        terminated = False
        truncated = False
        info = {}
        first_blue_action = None

        while not (terminated or truncated):
            action_mask = env.action_masks()
            action = predict_with_local_rng(
                blue_model,
                obs,
                action_mask,
                False,
                blue_rng,
            )
            if first_blue_action is None:
                first_blue_action = action
            if (
                action < 0
                or action >= action_mask.size
                or not bool(action_mask[action])
            ):
                mask_violations += 1
            obs, _, terminated, truncated, info = env.step(action)

        blue_openings[first_blue_action] += 1
        mask_violations += env.opponent_mask_violations
        winner = info.get("winner", env.logic.winner)
        if winner == 1:
            blue_wins += 1
            blue_win_indicators.append(1)
        elif winner == 2:
            red_wins += 1
            blue_win_indicators.append(0)
        else:
            draws += 1
            blue_win_indicators.append(0)
        if info.get("outcome") == "agent_suicide":
            blue_suicides += 1
        elif info.get("outcome") == "opponent_suicide":
            red_suicides += 1
        game_lengths.append(len(env.move_trace))

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            print(
                f"  {completed}/{episodes}: blue_wins={blue_wins} "
                f"blue_win_rate={blue_wins / completed:.3f}",
                flush=True,
            )

    result = {
        "episodes": episodes,
        "blue_wins": blue_wins,
        "red_wins": red_wins,
        "draws": draws,
        "blue_win_rate": blue_wins / episodes,
        "blue_suicides": blue_suicides,
        "blue_suicide_rate": blue_suicides / episodes,
        "red_suicides": red_suicides,
        "red_suicide_rate": red_suicides / episodes,
        "mask_violations": mask_violations,
        "mean_game_length": float(np.mean(game_lengths)),
        "unique_blue_opening_count": len(blue_openings),
        "blue_opening_histogram": {
            f"{action % env.board_size},{action // env.board_size}": count
            for action, count in sorted(blue_openings.items())
        },
        "blue_win_indicators": blue_win_indicators,
        "blue_deterministic": False,
        "red_deterministic": False,
    }
    env.close()
    return result


def run_blue_vs_random_red(*, blue_model_path, episodes, seed):
    blue_model = MaskablePPO.load(blue_model_path, device="cpu")
    env = GreatKingdomEnv(agent_player=1)
    blue_wins = 0
    red_wins = 0
    draws = 0
    blue_suicides = 0
    red_suicides = 0
    mask_violations = 0
    game_lengths = []

    print(f"\n=== Blue1 vs Random Red: {episodes} games ===", flush=True)
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        info = {}
        while not (terminated or truncated):
            action_mask = env.action_masks()
            action, _ = blue_model.predict(
                obs,
                action_masks=action_mask,
                deterministic=True,
            )
            action = int(np.asarray(action).item())
            if (
                action < 0
                or action >= action_mask.size
                or not bool(action_mask[action])
            ):
                mask_violations += 1
            obs, _, terminated, truncated, info = env.step(action)

        winner = info.get("winner", env.logic.winner)
        if winner == 1:
            blue_wins += 1
        elif winner == 2:
            red_wins += 1
        else:
            draws += 1
        if info.get("outcome") == "agent_suicide":
            blue_suicides += 1
        elif info.get("outcome") == "opponent_suicide":
            red_suicides += 1
        board = np.asarray(env.logic.board)
        game_lengths.append(int(np.count_nonzero((board == 1) | (board == 2))))

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            print(
                f"  {completed}/{episodes}: blue_wins={blue_wins} "
                f"blue_win_rate={blue_wins / completed:.3f}",
                flush=True,
            )

    env.close()
    return {
        "episodes": episodes,
        "blue_wins": blue_wins,
        "red_wins": red_wins,
        "draws": draws,
        "blue_win_rate": blue_wins / episodes,
        "blue_suicides": blue_suicides,
        "blue_suicide_rate": blue_suicides / episodes,
        "red_suicides": red_suicides,
        "red_suicide_rate": red_suicides / episodes,
        "mask_violations": mask_violations,
        "mean_game_length": float(np.mean(game_lengths)),
        "blue_deterministic": True,
        "red_opponent": "uniformly random",
    }


def read_blue0_random_reference():
    path = Path("reports/blue_10k_random_baseline_20260822/summary.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["blue_ppo_10k"]["win_rate"], str(path)


def strip_crossplay_indicators(result):
    result = dict(result)
    result.pop("red_win_indicators", None)
    return result


def run_crossplay(episodes, master_seed):
    specs = {
        "A_red0_vs_blue0": (RED0_MODEL, BLUE0_MODEL),
        "B_red0_vs_blue1": (RED0_MODEL, BLUE1_MODEL),
        "C_red1_vs_blue0": (RED1_MODEL, BLUE0_MODEL),
        "D_red1_vs_blue1": (RED1_MODEL, BLUE1_MODEL),
    }
    cells = {}
    for label, (red_path, blue_path) in specs.items():
        print(f"\n##### Cross-play {label} #####", flush=True)
        result, _ = run_policy_match(
            red_model_path=red_path,
            blue_model_path=blue_path,
            episodes=episodes,
            deterministic=False,
            master_seed=master_seed,
        )
        cells[label] = strip_crossplay_indicators(result)

    a = cells["A_red0_vs_blue0"]
    b = cells["B_red0_vs_blue1"]
    c = cells["C_red1_vs_blue0"]
    d = cells["D_red1_vs_blue1"]
    red1_vs_blue0_delta = c["red_win_rate"] - a["red_win_rate"]
    blue1_vs_red1_delta = (
        d["blue_wins"] / d["episodes"] - c["blue_wins"] / c["episodes"]
    )
    blue1_vs_red0_delta = (
        b["blue_wins"] / b["episodes"] - a["blue_wins"] / a["episodes"]
    )
    if blue1_vs_red1_delta > 0 and blue1_vs_red0_delta < 0:
        observation = (
            "Blue1 improved against Red1 but weakened against Red0 in point estimates; "
            "latest-opponent overfitting/cycling is a concern."
        )
    elif blue1_vs_red1_delta > 0 and blue1_vs_red0_delta >= 0:
        observation = (
            "Blue1 improved against Red1 without a point-estimate drop against Red0; "
            "no cross-play forgetting signal was observed in this matrix."
        )
    else:
        observation = (
            "Blue1 did not improve against Red1 in the cross-play point estimate."
        )
    return {
        "episodes_per_cell": episodes,
        "master_seed": master_seed,
        "red_win_rate_matrix": {
            "Red0": {
                "Blue0": a["red_win_rate"],
                "Blue1": b["red_win_rate"],
            },
            "Red1": {
                "Blue0": c["red_win_rate"],
                "Blue1": d["red_win_rate"],
            },
        },
        "cells": cells,
        "derived": {
            "red1_minus_red0_vs_blue0": red1_vs_blue0_delta,
            "blue1_minus_blue0_vs_red1": blue1_vs_red1_delta,
            "blue1_minus_blue0_vs_red0": blue1_vs_red0_delta,
        },
        "cycling_forgetting_observation": observation,
    }


def blue_match_row(label, result):
    return (
        f"{label} | {result['episodes']} | {result['blue_wins']} | "
        f"{result['red_wins']} | {result['draws']} | "
        f"{result['blue_win_rate']:.3f} | {result['blue_suicide_rate']:.3f} | "
        f"{result['red_suicide_rate']:.3f} | {result['mask_violations']} | "
        f"{result['mean_game_length']:.2f}"
    )


def red_match_row(label, result):
    return (
        f"{label} | {result['red_wins']} | {result['blue_wins']} | "
        f"{result['draws']} | {result['red_win_rate']:.3f} | "
        f"{result['red_suicide_rate']:.3f} | {result['blue_suicide_rate']:.3f} | "
        f"{result['mask_violations']} | {result['mean_game_length']:.2f}"
    )


def render_summary(summary):
    pre = summary["pre_blue0_vs_frozen_red1"]
    post = summary["post_blue1_vs_frozen_red1"]
    random_red = summary["blue1_vs_random_red"]
    interval = summary["blue_win_rate_delta_bootstrap_95_ci"]
    crossplay = summary["crossplay"]
    lines = [
        "# Blue0 continuation vs Frozen Red1",
        "",
        "This is the reverse-direction validation of one alternating cycle.",
        "Blue0 checkpoint state was continued; Frozen Red1 was not updated.",
        "Both policies used stochastic masked sampling in policy-vs-policy games.",
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
            "condition | episodes | Blue wins | Red wins | draws | Blue win_rate | "
            "Blue suicide_rate | Red suicide_rate | mask violations | mean game length"
        ),
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
        blue_match_row("PRE Blue0 vs Frozen Red1", pre),
        blue_match_row("POST Blue1 vs Frozen Red1", post),
        blue_match_row("Blue1 vs Random Red", random_red),
        "",
        f"Blue0 -> Blue1 delta: {summary['blue_win_rate_delta']:+.3f}",
        (
            "Paired bootstrap 95% CI: "
            f"[{interval['low']:+.3f}, {interval['high']:+.3f}]"
        ),
        f"Learning result: {summary['learning_result']}",
        (
            "Blue1 vs Random Red change from Blue0 reference: "
            f"{summary['random_red_reference_delta']:+.3f}"
        ),
        "",
        "## 2x2 cross-play (Red win rate)",
        "",
        " | Blue0 | Blue1",
        "--- | ---: | ---:",
        (
            f"Red0 | {crossplay['red_win_rate_matrix']['Red0']['Blue0']:.3f} | "
            f"{crossplay['red_win_rate_matrix']['Red0']['Blue1']:.3f}"
        ),
        (
            f"Red1 | {crossplay['red_win_rate_matrix']['Red1']['Blue0']:.3f} | "
            f"{crossplay['red_win_rate_matrix']['Red1']['Blue1']:.3f}"
        ),
        "",
        (
            "cell | Red wins | Blue wins | draws | Red win_rate | Red suicide_rate | "
            "Blue suicide_rate | mask violations | mean game length"
        ),
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for label, result in crossplay["cells"].items():
        lines.append(red_match_row(label, result))
    lines.extend(
        [
            "",
            crossplay["cycling_forgetting_observation"],
            "",
            "## [미확정]",
            "",
            "- Long-run alternating self-play stability",
            "- Nash equilibrium or absolute policy ranking",
            "- Human-opponent strength",
            "- Whether corner openings are optimal",
            "",
        ]
    )
    return "\n".join(lines)


def run_baseline(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    matrix_path = report_dir / "crossplay_matrix.json"
    existing = [
        path for path in (json_path, text_path, matrix_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing reports: {existing}")
    hashes = validate_base_hashes()

    pre = run_blue_match(
        blue_model_path=BLUE0_MODEL,
        red_model_path=RED1_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
    )
    partial = {
        "status": "baseline_complete",
        "experiment": "Blue0 continuation vs Frozen Red1",
        "scope": "Reverse direction of one alternating cycle only.",
        "configuration": {
            "episodes_per_condition": args.episodes,
            "master_seed": args.master_seed,
            "blue0_model_path": str(BLUE0_MODEL),
            "red1_model_path": str(RED1_MODEL),
            "blue1_model_path": str(BLUE1_MODEL),
            "policy_evaluation_deterministic": False,
            "frozen_training_opponent_deterministic": False,
            "action_masks_both_policies": True,
        },
        "base_model_sha256_before": hashes,
        "pre_blue0_vs_frozen_red1": pre,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
    print(f"Saved PRE checkpoint {json_path}", flush=True)


def run_post_train(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    matrix_path = report_dir / "crossplay_matrix.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"PRE checkpoint not found: {json_path}")
    if text_path.exists() or matrix_path.exists():
        raise FileExistsError("refusing to overwrite completed cycle reports")
    if not BLUE1_MODEL.is_file():
        raise FileNotFoundError(BLUE1_MODEL)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "baseline_complete":
        raise ValueError("summary.json is not a PRE-only checkpoint")
    if (
        summary["configuration"]["episodes_per_condition"] != args.episodes
        or summary["configuration"]["master_seed"] != args.master_seed
    ):
        raise ValueError("POST settings do not match PRE")
    hashes_after = validate_base_hashes()
    if summary["base_model_sha256_before"] != hashes_after:
        raise RuntimeError("Red0, Blue0, or Red1 changed after PRE")

    post = run_blue_match(
        blue_model_path=BLUE1_MODEL,
        red_model_path=RED1_MODEL,
        episodes=args.episodes,
        master_seed=args.master_seed,
    )
    random_red = run_blue_vs_random_red(
        blue_model_path=BLUE1_MODEL,
        episodes=args.episodes,
        seed=RANDOM_RED_SEED,
    )
    crossplay = run_crossplay(args.episodes, args.master_seed)

    pre = summary["pre_blue0_vs_frozen_red1"]
    delta = post["blue_win_rate"] - pre["blue_win_rate"]
    interval = paired_bootstrap_delta(
        pre["blue_win_indicators"],
        post["blue_win_indicators"],
    )
    blue0_random_reference, reference_path = read_blue0_random_reference()

    blue0 = MaskablePPO.load(BLUE0_MODEL, device="cpu")
    blue1 = MaskablePPO.load(BLUE1_MODEL, device="cpu")
    blue0_optimizer_entries = len(blue0.policy.optimizer.state_dict()["state"])
    blue1_optimizer_entries = len(blue1.policy.optimizer.state_dict()["state"])
    blue1_hash = sha256_file(BLUE1_MODEL)
    all_mask_violations = (
        pre["mask_violations"]
        + post["mask_violations"]
        + random_red["mask_violations"]
        + sum(cell["mask_violations"] for cell in crossplay["cells"].values())
    )
    infrastructure_pass = (
        blue0_optimizer_entries > 0
        and blue1_optimizer_entries > 0
        and blue1.num_timesteps > blue0.num_timesteps
        and blue1_hash != hashes_after["blue0"]
        and all_mask_violations == 0
        and validate_base_hashes() == hashes_after
    )
    if delta > 0 and interval["low"] > 0:
        learning_result = "PASS: clear improvement"
        alternating_cycle_pass = True
    elif delta > 0:
        learning_result = "INCONCLUSIVE: positive point estimate, CI includes zero"
        alternating_cycle_pass = False
    else:
        learning_result = "FAIL: no point-estimate improvement"
        alternating_cycle_pass = False

    summary.update(
        {
            "status": "complete",
            "post_blue1_vs_frozen_red1": post,
            "blue1_vs_random_red": random_red,
            "blue_win_rate_delta": delta,
            "blue_win_rate_delta_bootstrap_95_ci": interval,
            "blue0_vs_random_red_reference": blue0_random_reference,
            "blue0_random_reference_path": reference_path,
            "random_red_reference_delta": (
                random_red["blue_win_rate"] - blue0_random_reference
            ),
            "training": {
                "continuation": True,
                "reset_num_timesteps": False,
                "requested_additional_timesteps": 10_000,
                "start_num_timesteps": blue0.num_timesteps,
                "end_num_timesteps": blue1.num_timesteps,
                "actual_counter_increase": blue1.num_timesteps - blue0.num_timesteps,
                "start_n_updates": blue0._n_updates,
                "end_n_updates": blue1._n_updates,
                "optimizer_state_entries": blue0_optimizer_entries,
                "blue1_optimizer_state_entries": blue1_optimizer_entries,
            },
            "model_sha256": {
                "red0_before": summary["base_model_sha256_before"]["red0"],
                "red0_after": hashes_after["red0"],
                "blue0_before": summary["base_model_sha256_before"]["blue0"],
                "blue0_after": hashes_after["blue0"],
                "red1_before": summary["base_model_sha256_before"]["red1"],
                "red1_after": hashes_after["red1"],
                "blue1": blue1_hash,
                "blue0_and_blue1_differ": blue1_hash != hashes_after["blue0"],
            },
            "crossplay": crossplay,
            "all_mask_violations": all_mask_violations,
            "infrastructure_pass": infrastructure_pass,
            "learning_result": learning_result,
            "alternating_cycle_pass": alternating_cycle_pass,
        }
    )
    matrix_path.write_text(
        json.dumps(crossplay, indent=2) + "\n",
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render_summary(summary), encoding="utf-8")
    print(f"Blue0_to_Blue1_delta={delta:+.3f}", flush=True)
    print(
        f"bootstrap_95_ci=[{interval['low']:+.3f}, {interval['high']:+.3f}]",
        flush=True,
    )
    print(f"learning_result={learning_result}", flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {text_path}", flush=True)
    print(f"Saved {matrix_path}", flush=True)


def refresh_random_red(args):
    """Re-evaluate only Random Red after a transition-semantics fix."""
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    matrix_path = report_dir / "crossplay_matrix.json"
    for path in (json_path, text_path, matrix_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("summary.json is not a completed cycle report")
    if (
        summary["configuration"]["episodes_per_condition"] != args.episodes
        or summary["configuration"]["master_seed"] != args.master_seed
    ):
        raise ValueError("refresh settings do not match the completed report")
    validate_base_hashes()
    if not BLUE1_MODEL.is_file():
        raise FileNotFoundError(BLUE1_MODEL)

    random_red = run_blue_vs_random_red(
        blue_model_path=BLUE1_MODEL,
        episodes=args.episodes,
        seed=RANDOM_RED_SEED,
    )
    blue0_random_reference = summary["blue0_vs_random_red_reference"]
    summary["blue1_vs_random_red"] = random_red
    summary["random_red_reference_delta"] = (
        random_red["blue_win_rate"] - blue0_random_reference
    )
    summary["all_mask_violations"] = (
        summary["pre_blue0_vs_frozen_red1"]["mask_violations"]
        + summary["post_blue1_vs_frozen_red1"]["mask_violations"]
        + random_red["mask_violations"]
        + sum(
            cell["mask_violations"]
            for cell in summary["crossplay"]["cells"].values()
        )
    )
    summary["infrastructure_pass"] = (
        summary["all_mask_violations"] == 0
        and summary["model_sha256"]["blue0_and_blue1_differ"]
        and validate_base_hashes() == summary["base_model_sha256_before"]
    )
    summary["random_red_transition_fix_verified"] = True
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render_summary(summary), encoding="utf-8")
    print(f"Refreshed Random Red result in {json_path}", flush=True)
    print(f"Refreshed {text_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("baseline", "post-train", "refresh-random"):
        subparser = subparsers.add_parser(phase)
        subparser.add_argument("--episodes", type=int, default=500)
        subparser.add_argument("--master-seed", type=int, default=MASTER_SEED)
        subparser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.phase == "baseline":
        run_baseline(args)
    elif args.phase == "post-train":
        run_post_train(args)
    else:
        refresh_random_red(args)


if __name__ == "__main__":
    main()
