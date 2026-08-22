import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from frozen_policy_env import FrozenPolicyOpponentEnv, predict_with_local_rng
from gk_env import GreatKingdomEnv


OLD_RED_MODEL = Path("models/MaskablePPO_CNN/masked_ppo_10000.zip")
BLUE_MODEL = Path("models/MaskablePPO_CNN/blue_masked_ppo_10000.zip")
NEW_RED_MODEL = Path(
    "models/MaskablePPO_CNN/red_br_vs_blue10k_10000.zip"
)
DEFAULT_REPORT_DIR = Path(
    "reports/frozen_policy_br_red_vs_blue10k_20260822"
)
POLICY_MASTER_SEED = 20260822
RANDOM_BLUE_SEED = 30000
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_SAMPLES = 10_000


def run_policy_match(
    *,
    red_model_path,
    blue_model_path,
    episodes,
    deterministic,
    master_seed,
    keep_last_trace=False,
):
    red_model = MaskablePPO.load(red_model_path, device="cpu")
    env = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=blue_model_path,
        opponent_deterministic=deterministic,
        opponent_seed=master_seed + 100_000,
    )

    red_wins = 0
    blue_wins = 0
    draws = 0
    red_suicides = 0
    blue_suicides = 0
    mask_violations = 0
    game_lengths = []
    blue_openings = Counter()
    red_win_indicators = []
    last_trace = None
    last_info = None
    final_board = None

    mode = "deterministic" if deterministic else "stochastic"
    print(
        f"\n=== Red policy vs frozen Blue policy: {mode}, {episodes} game(s) ===",
        flush=True,
    )

    for episode in range(episodes):
        obs, _ = env.reset(seed=master_seed + episode)
        red_rng = np.random.default_rng(
            np.random.SeedSequence([master_seed, episode, 2])
        )
        terminated = False
        truncated = False
        info = {}

        if not env.move_trace or env.move_trace[0]["player"] != 1:
            raise AssertionError("frozen Blue did not make the opening move")
        blue_openings[env.move_trace[0]["action"]] += 1

        while not (terminated or truncated):
            action_mask = env.action_masks()
            action = predict_with_local_rng(
                red_model,
                obs,
                action_mask,
                deterministic,
                red_rng,
            )
            if (
                action < 0
                or action >= action_mask.size
                or not bool(action_mask[action])
            ):
                mask_violations += 1
            obs, _, terminated, truncated, info = env.step(action)

        mask_violations += env.opponent_mask_violations
        winner = info.get("winner", env.logic.winner)
        if winner == 2:
            red_wins += 1
            red_win_indicators.append(1)
        elif winner == 1:
            blue_wins += 1
            red_win_indicators.append(0)
        else:
            draws += 1
            red_win_indicators.append(0)

        if info.get("outcome") == "agent_suicide":
            red_suicides += 1
        elif info.get("outcome") == "opponent_suicide":
            blue_suicides += 1
        game_lengths.append(len(env.move_trace))

        if keep_last_trace:
            last_trace = [dict(move) for move in env.move_trace]
            last_info = dict(info)
            final_board = [row[:] for row in env.logic.board]

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            print(
                f"  {completed}/{episodes}: red_wins={red_wins} "
                f"red_win_rate={red_wins / completed:.3f}",
                flush=True,
            )

    result = {
        "episodes": episodes,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "red_win_rate": red_wins / episodes,
        "red_suicides": red_suicides,
        "red_suicide_rate": red_suicides / episodes,
        "blue_suicides": blue_suicides,
        "blue_suicide_rate": blue_suicides / episodes,
        "mask_violations": mask_violations,
        "mean_game_length": float(np.mean(game_lengths)),
        "unique_blue_opening_count": len(blue_openings),
        "blue_opening_histogram": {
            f"{action % env.board_size},{action // env.board_size}": count
            for action, count in sorted(blue_openings.items())
        },
        "red_win_indicators": red_win_indicators,
        "red_deterministic": deterministic,
        "blue_deterministic": deterministic,
    }
    trace = None
    if keep_last_trace:
        trace = {
            "red_model_path": str(red_model_path),
            "blue_model_path": str(blue_model_path),
            "deterministic": deterministic,
            "master_seed": master_seed,
            "winner": last_info.get("winner"),
            "outcome": last_info.get("outcome"),
            "win_reason": env.logic.win_reason,
            "game_length": len(last_trace),
            "mask_violations": mask_violations,
            "moves": last_trace,
            "final_board": final_board,
        }
    env.close()
    return result, trace


def run_new_red_vs_random_blue(
    *,
    red_model_path,
    episodes,
    seed,
):
    red_model = MaskablePPO.load(red_model_path, device="cpu")
    env = GreatKingdomEnv(agent_player=2)
    red_wins = 0
    blue_wins = 0
    draws = 0
    red_suicides = 0
    blue_suicides = 0
    mask_violations = 0
    game_lengths = []

    print(
        f"\n=== New Red best-response vs Random Blue: {episodes} games ===",
        flush=True,
    )
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            action_mask = env.action_masks()
            action, _ = red_model.predict(
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
        if winner == 2:
            red_wins += 1
        elif winner == 1:
            blue_wins += 1
        else:
            draws += 1
        if info.get("outcome") == "agent_suicide":
            red_suicides += 1
        elif info.get("outcome") == "opponent_suicide":
            blue_suicides += 1
        board = np.asarray(env.logic.board)
        game_lengths.append(int(np.count_nonzero((board == 1) | (board == 2))))

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            print(
                f"  {completed}/{episodes}: red_wins={red_wins} "
                f"red_win_rate={red_wins / completed:.3f}",
                flush=True,
            )

    env.close()
    return {
        "episodes": episodes,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "red_win_rate": red_wins / episodes,
        "red_suicides": red_suicides,
        "red_suicide_rate": red_suicides / episodes,
        "blue_suicides": blue_suicides,
        "blue_suicide_rate": blue_suicides / episodes,
        "mask_violations": mask_violations,
        "mean_game_length": float(np.mean(game_lengths)),
        "red_deterministic": True,
        "blue_opponent": "uniformly random",
    }


def paired_bootstrap_delta(old_indicators, new_indicators):
    old = np.asarray(old_indicators, dtype=np.float64)
    new = np.asarray(new_indicators, dtype=np.float64)
    if old.shape != new.shape:
        raise ValueError("OLD and NEW result arrays must have the same shape")

    paired_differences = new - old
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(
        0,
        paired_differences.size,
        size=(BOOTSTRAP_SAMPLES, paired_differences.size),
    )
    bootstrap_deltas = paired_differences[sample_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_deltas, [0.025, 0.975])
    return {
        "method": "paired episode bootstrap percentile interval",
        "confidence_level": 0.95,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "low": float(low),
        "high": float(high),
    }


def read_old_random_reference():
    path = Path(
        "reports/generalization_10k_corner_opening_20260822/summary.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    for condition in data["conditions"]:
        if (
            condition["red_agent"] == "ppo_10k"
            and condition["blue_opponent"] == "random"
        ):
            return condition["win_rate"], str(path)
    raise ValueError("OLD Red vs Random Blue reference not found")


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
    new = summary["new_red_br_vs_frozen_blue10k"]
    random_blue = summary["new_red_br_vs_random_blue"]
    interval = summary["win_rate_delta_bootstrap_95_ci"]
    passed = summary["pass"]
    interpretation = (
        "The NEW Red point estimate improved against frozen Blue."
        if new["red_win_rate"] > old["red_win_rate"]
        else "The NEW Red point estimate did not improve against frozen Blue."
    )
    lines = [
        "# Frozen-policy Red best-response prerequisite",
        "",
        "This is one Red learner against one frozen Blue policy, not full self-play.",
        "Frozen Blue samples its policy with deterministic=False during training.",
        "Both policies receive canonical observations and action masks.",
        "",
        "## [확정] Executed results",
        "",
        (
            "condition | episodes | Red wins | Blue wins | draws | Red win_rate | "
            "Red suicide_rate | Blue suicide_rate | mask violations | mean game length"
        ),
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
        format_match_row("OLD Red10k vs Frozen Blue10k", old),
        format_match_row("NEW Red BR vs Frozen Blue10k", new),
        format_match_row("NEW Red BR vs Random Blue", random_blue),
        "",
        (
            "OLD -> NEW frozen-Blue win-rate delta: "
            f"{summary['old_to_new_win_rate_delta']:+.3f}"
        ),
        (
            "Paired bootstrap 95% CI: "
            f"[{interval['low']:+.3f}, {interval['high']:+.3f}]"
        ),
        (
            "NEW vs Random Blue change from OLD reference: "
            f"{summary['random_blue_reference_delta']:+.3f}"
        ),
        f"Overall prerequisite result: {'PASS' if passed else 'FAIL'}",
        "",
        "## [해석]",
        "",
        interpretation,
        (
            "The uncertainty interval is entirely above zero."
            if interval["low"] > 0
            else "The uncertainty interval includes zero, so the positive delta is not conclusive."
        ),
        "",
        "## [미확정]",
        "",
        "- Full self-play convergence",
        "- Human-opponent strength",
        "- Nash equilibrium behavior",
        "- Whether corner openings are optimal",
        "",
    ]
    return "\n".join(lines)


def run_baseline(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    trace_path = report_dir / "deterministic_trace.json"
    existing = [
        path for path in (json_path, text_path, trace_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing reports: {existing}")
    for model_path in (OLD_RED_MODEL, BLUE_MODEL):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

    deterministic, trace = run_policy_match(
        red_model_path=OLD_RED_MODEL,
        blue_model_path=BLUE_MODEL,
        episodes=1,
        deterministic=True,
        master_seed=args.master_seed,
        keep_last_trace=True,
    )
    if deterministic["mask_violations"] != 0:
        raise AssertionError("deterministic sanity game had a mask violation")
    if trace["winner"] not in (1, 2):
        raise AssertionError("deterministic sanity game had no winner")

    old_baseline, _ = run_policy_match(
        red_model_path=OLD_RED_MODEL,
        blue_model_path=BLUE_MODEL,
        episodes=args.episodes,
        deterministic=False,
        master_seed=args.master_seed,
    )
    partial_summary = {
        "status": "baseline_complete",
        "experiment": "Red best-response against frozen Blue10k prerequisite",
        "scope": "One frozen-policy opponent iteration; not full self-play.",
        "configuration": {
            "episodes_per_stochastic_policy_match": args.episodes,
            "policy_master_seed": args.master_seed,
            "old_red_model_path": str(OLD_RED_MODEL),
            "frozen_blue_model_path": str(BLUE_MODEL),
            "policy_evaluation_deterministic": False,
            "frozen_training_opponent_deterministic": False,
            "action_masks_both_policies": True,
        },
        "deterministic_sanity": deterministic,
        "old_red10k_vs_frozen_blue10k": old_baseline,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(trace, indent=2) + "\n",
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(partial_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {trace_path}", flush=True)
    print(f"Saved baseline checkpoint {json_path}", flush=True)


def run_post_train(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    if not json_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {json_path}")
    if text_path.exists():
        raise FileExistsError(f"refusing to overwrite completed report: {text_path}")
    if not NEW_RED_MODEL.is_file():
        raise FileNotFoundError(NEW_RED_MODEL)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "baseline_complete":
        raise ValueError("summary.json is not a baseline-only checkpoint")
    configuration = summary["configuration"]
    if (
        configuration["episodes_per_stochastic_policy_match"] != args.episodes
        or configuration["policy_master_seed"] != args.master_seed
    ):
        raise ValueError("post-training settings do not match the baseline")

    new_result, _ = run_policy_match(
        red_model_path=NEW_RED_MODEL,
        blue_model_path=BLUE_MODEL,
        episodes=args.episodes,
        deterministic=False,
        master_seed=args.master_seed,
    )
    random_blue_result = run_new_red_vs_random_blue(
        red_model_path=NEW_RED_MODEL,
        episodes=args.episodes,
        seed=RANDOM_BLUE_SEED,
    )

    old_result = summary["old_red10k_vs_frozen_blue10k"]
    delta = new_result["red_win_rate"] - old_result["red_win_rate"]
    interval = paired_bootstrap_delta(
        old_result["red_win_indicators"],
        new_result["red_win_indicators"],
    )
    old_random_reference, reference_path = read_old_random_reference()
    all_mask_violations = sum(
        result["mask_violations"]
        for result in (
            summary["deterministic_sanity"],
            old_result,
            new_result,
            random_blue_result,
        )
    )
    model = MaskablePPO.load(NEW_RED_MODEL, device="cpu")
    training_completed = model.num_timesteps >= 10_000
    passed = (
        summary["deterministic_sanity"]["episodes"] == 1
        and summary["deterministic_sanity"]["red_wins"]
        + summary["deterministic_sanity"]["blue_wins"]
        == 1
        and all_mask_violations == 0
        and training_completed
        and delta > 0
    )

    configuration.update(
        {
            "new_red_model_path": str(NEW_RED_MODEL),
            "training_initialization": "from scratch",
            "training_requested_timesteps": 10_000,
            "training_stored_num_timesteps": model.num_timesteps,
            "random_blue_evaluation_seed": RANDOM_BLUE_SEED,
            "random_blue_evaluation_red_deterministic": True,
            "old_random_blue_reference_path": reference_path,
        }
    )
    summary.update(
        {
            "status": "complete",
            "new_red_br_vs_frozen_blue10k": new_result,
            "new_red_br_vs_random_blue": random_blue_result,
            "old_to_new_win_rate_delta": delta,
            "win_rate_delta_bootstrap_95_ci": interval,
            "old_red10k_vs_random_blue_reference": old_random_reference,
            "random_blue_reference_delta": (
                random_blue_result["red_win_rate"] - old_random_reference
            ),
            "all_mask_violations": all_mask_violations,
            "training_completed": training_completed,
            "pass_criteria": {
                "frozen_policy_transition_sanity": True,
                "mask_violations_zero": all_mask_violations == 0,
                "policy_games_terminate": True,
                "red_best_response_10k_completed": training_completed,
                "new_frozen_blue_win_rate_above_old": delta > 0,
            },
            "pass": passed,
        }
    )
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(render_summary(summary), encoding="utf-8")
    print(f"OLD_to_NEW_delta={delta:+.3f}", flush=True)
    print(
        f"bootstrap_95_ci=[{interval['low']:+.3f}, {interval['high']:+.3f}]",
        flush=True,
    )
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
