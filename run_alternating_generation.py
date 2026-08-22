import argparse
import json
import subprocess
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from evaluate_frozen_br import paired_bootstrap_delta
from evaluate_terminal_fix_audit import (
    evaluate_blue_crossplay,
    evaluate_crossplay,
    evaluate_vs_random,
    sha256_file,
    terminal_text,
)
from frozen_policy_env import FrozenPolicyOpponentEnv


REGRESSION_SCRIPTS = (
    "test_rules_minimal.py",
    "test_gk_env_players.py",
    "test_frozen_policy_env.py",
)


def learner_player_number(learner_player):
    mapping = {"blue": 1, "red": 2}
    try:
        return mapping[learner_player]
    except KeyError as exc:
        raise ValueError("learner-player must be 'red' or 'blue'") from exc


def planned_child_models(output_model):
    return [Path(output_model)]


def model_manifest(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_generation_paths(
    *, parent_model, latest_opponent, previous_opponent, output_model, report_dir
):
    inputs = {
        "parent_model": Path(parent_model),
        "latest_opponent": Path(latest_opponent),
        "previous_opponent": Path(previous_opponent),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    output_model = Path(output_model)
    report_dir = Path(report_dir)
    if output_model.suffix != ".zip":
        raise ValueError("output-model must end in .zip")
    if output_model.exists():
        raise FileExistsError(f"refusing to overwrite output model: {output_model}")
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {report_dir}")
    if not output_model.parent.is_dir():
        raise FileNotFoundError(
            f"output model parent directory not found: {output_model.parent}"
        )
    if not report_dir.parent.is_dir():
        raise FileNotFoundError(
            f"report parent directory not found: {report_dir.parent}"
        )

    input_resolved = {path.resolve() for path in inputs.values()}
    if output_model.resolve() in input_resolved:
        raise ValueError("output-model must differ from every input model")
    outputs = planned_child_models(output_model)
    if outputs != [output_model]:
        raise AssertionError("one invocation must declare exactly one child model")
    return inputs


def run_regression_checks(repo_dir):
    results = {}
    for script in REGRESSION_SCRIPTS:
        completed = subprocess.run(
            [sys.executable, script],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        results[script] = {
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
        }
        if completed.returncode != 0:
            raise RuntimeError(
                f"regression failed: {script}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        print(f"regression {script}: PASS", flush=True)
    return results


def make_training_env(
    *, rank, agent_player, opponent_model_path, training_seed
):
    def _init():
        env = FrozenPolicyOpponentEnv(
            agent_player=agent_player,
            opponent_model_path=opponent_model_path,
            opponent_deterministic=False,
            opponent_seed=training_seed + 10_000 + rank,
        )
        env.reset(seed=training_seed + rank)
        return Monitor(env)

    return _init


def evaluate_match(
    *, learner_player, learner_model, opponent_model, episodes, seed, label
):
    if learner_player == 2:
        return evaluate_crossplay(
            red_model_path=learner_model,
            blue_model_path=opponent_model,
            episodes=episodes,
            master_seed=seed,
            label=label,
        )
    return evaluate_blue_crossplay(
        blue_model_path=learner_model,
        red_model_path=opponent_model,
        episodes=episodes,
        master_seed=seed,
        label=label,
    )


def learner_win_rate(result, learner_player):
    return result["red_win_rate"] if learner_player == 2 else result["blue_win_rate"]


def learner_indicators(result, learner_player):
    key = "red_win_indicators" if learner_player == 2 else "blue_win_indicators"
    return result[key]


def result_row(label, result, learner_player):
    return (
        f"{label} | {result['red_wins']} | {result['blue_wins']} | "
        f"{result['draws']} | {learner_win_rate(result, learner_player):.3f} | "
        f"{result['red_suicides']} | {result['blue_suicides']} | "
        f"{result['mask_violations']} | {result['mean_game_length']:.3f} | "
        f"{terminal_text(result)}"
    )


def render_summary(summary):
    learner_player = summary["configuration"]["learner_player_number"]
    pre = summary["evaluations"]["parent_vs_latest"]
    post = summary["evaluations"]["child_vs_latest"]
    retention_parent = summary["evaluations"]["parent_vs_previous"]
    retention_child = summary["evaluations"]["child_vs_previous"]
    random_result = summary["evaluations"]["child_vs_random"]
    latest_ci = summary["statistics"]["latest_delta_bootstrap_95_ci"]
    retention_ci = summary["statistics"]["retention_delta_bootstrap_95_ci"]
    matrix = summary["crossplay"]["learner_win_rate_matrix"]
    lines = [
        "# Generic alternating one-generation result",
        "",
        "Exactly one explicitly configured child checkpoint was generated.",
        "No next-generation invocation, loop, pool update, or self-trigger was performed.",
        "",
        "## [Runner infrastructure]",
        "",
        f"Infrastructure pass: {str(summary['runner_infrastructure_pass']).lower()}",
        f"Learner: {summary['configuration']['learner_player']}",
        f"CLI: {' '.join(summary['runner_cli'])}",
        (
            "Timestep counter: "
            f"{summary['training']['parent_num_timesteps']} -> "
            f"{summary['training']['child_num_timesteps']}"
        ),
        (
            "Update counter: "
            f"{summary['training']['parent_n_updates']} -> "
            f"{summary['training']['child_n_updates']}"
        ),
        (
            "Optimizer state entries: "
            f"{summary['training']['parent_optimizer_state_entries']} -> "
            f"{summary['training']['child_optimizer_state_entries']}"
        ),
        "",
        "## [Confirmed evaluations]",
        "",
        "Condition | Red wins | Blue wins | Draws | Learner win rate | Red suicide | Blue suicide | Mask | Mean length | Terminal reasons",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---",
        result_row("PRE parent vs latest", pre, learner_player),
        result_row("POST child vs latest", post, learner_player),
        result_row("Retention parent vs previous", retention_parent, learner_player),
        result_row("Retention child vs previous", retention_child, learner_player),
        result_row("Child vs random", random_result, learner_player),
        "",
        f"Latest delta: {summary['statistics']['latest_delta']:+.3f}",
        f"Latest paired bootstrap 95% CI: [{latest_ci['low']:+.3f}, {latest_ci['high']:+.3f}]",
        f"Retention delta: {summary['statistics']['retention_delta']:+.3f}",
        f"Retention paired bootstrap 95% CI: [{retention_ci['low']:+.3f}, {retention_ci['high']:+.3f}]",
        "",
        "## Generation cross-play (learner win rate)",
        "",
        " | Previous opponent | Latest opponent",
        "--- | ---: | ---:",
        f"Parent | {matrix['parent']['previous_opponent']:.3f} | {matrix['parent']['latest_opponent']:.3f}",
        f"Child | {matrix['child']['previous_opponent']:.3f} | {matrix['child']['latest_opponent']:.3f}",
        "",
        "## [Learning result]",
        "",
        f"Latest adaptation: {summary['learning_result']['latest_adaptation']}",
        f"Forgetting/cycling: {summary['learning_result']['forgetting_cycling_evidence']}",
        f"Next recommendation: {summary['learning_result']['next_recommendation']}",
        "",
    ]
    return "\n".join(lines)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def serializable_cli_arguments(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run_one_generation(args, runner_cli):
    learner_player = learner_player_number(args.learner_player)
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    parent_model = Path(args.parent_model)
    latest_opponent = Path(args.latest_opponent)
    previous_opponent = Path(args.previous_opponent)
    output_model = Path(args.output_model)
    report_dir = Path(args.report_dir)
    validate_generation_paths(
        parent_model=parent_model,
        latest_opponent=latest_opponent,
        previous_opponent=previous_opponent,
        output_model=output_model,
        report_dir=report_dir,
    )
    input_paths = {
        "parent_model": parent_model,
        "latest_opponent": latest_opponent,
        "previous_opponent": previous_opponent,
    }
    input_manifests_before = {
        name: model_manifest(path) for name, path in input_paths.items()
    }
    repo_dir = Path(__file__).resolve().parent
    regression_results = run_regression_checks(repo_dir)
    seeds = {
        "evaluation_master_seed": args.seed,
        "random_evaluation_seed": args.seed + 1_000_000,
        "training_seed": args.seed + 2_000_000,
        "bootstrap_seed": 20260822,
    }

    pre = evaluate_match(
        learner_player=learner_player,
        learner_model=parent_model,
        opponent_model=latest_opponent,
        episodes=args.episodes,
        seed=seeds["evaluation_master_seed"],
        label="PRE parent_vs_latest",
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    partial_summary = {
        "status": "pre_complete",
        "runner_cli": runner_cli,
        "runner_cli_arguments": serializable_cli_arguments(args),
        "configuration": {
            "learner_player": args.learner_player,
            "learner_player_number": learner_player,
            "timesteps_requested": args.timesteps,
            "episodes_per_evaluation": args.episodes,
            "evaluation_deterministic": False,
            "frozen_training_opponent_deterministic": False,
            "action_masks_both_policies": True,
        },
        "seeds": seeds,
        "regression_checks": regression_results,
        "input_models_before": input_manifests_before,
        "evaluations": {"parent_vs_latest": pre},
    }
    summary_path = report_dir / "summary.json"
    write_json(summary_path, partial_summary)
    print(f"Saved PRE checkpoint {summary_path}", flush=True)

    output_parent_before = {
        path.resolve() for path in output_model.parent.glob("*.zip")
    }
    num_envs = 8
    env = SubprocVecEnv(
        [
            make_training_env(
                rank=rank,
                agent_player=learner_player,
                opponent_model_path=latest_opponent,
                training_seed=seeds["training_seed"],
            )
            for rank in range(num_envs)
        ]
    )
    try:
        model = MaskablePPO.load(parent_model, env=env, device="auto")
        parent_optimizer_entries = len(
            model.policy.optimizer.state_dict()["state"]
        )
        if parent_optimizer_entries <= 0:
            raise RuntimeError("parent optimizer state was not restored")
        parent_num_timesteps = model.num_timesteps
        parent_n_updates = model._n_updates
        model.set_random_seed(seeds["training_seed"])
        print(
            f"Training one child: timesteps={parent_num_timesteps}, "
            f"updates={parent_n_updates}, optimizer_entries={parent_optimizer_entries}",
            flush=True,
        )
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
        model.save(output_model)
        child_num_timesteps = model.num_timesteps
        child_n_updates = model._n_updates
    finally:
        env.close()

    output_parent_after = {
        path.resolve() for path in output_model.parent.glob("*.zip")
    }
    created_model_paths = output_parent_after - output_parent_before
    if created_model_paths != {output_model.resolve()}:
        raise RuntimeError(
            "one-generation invariant failed; unexpected new model paths: "
            f"{sorted(map(str, created_model_paths))}"
        )

    child_model = MaskablePPO.load(output_model, device="cpu")
    child_optimizer_entries = len(
        child_model.policy.optimizer.state_dict()["state"]
    )
    output_manifest = model_manifest(output_model)
    partial_summary.update(
        {
            "status": "child_trained",
            "training": {
                "continuation": True,
                "reset_num_timesteps": False,
                "requested_timesteps": args.timesteps,
                "parent_num_timesteps": parent_num_timesteps,
                "child_num_timesteps": child_num_timesteps,
                "actual_counter_increase": child_num_timesteps
                - parent_num_timesteps,
                "parent_n_updates": parent_n_updates,
                "child_n_updates": child_n_updates,
                "parent_optimizer_state_entries": parent_optimizer_entries,
                "child_optimizer_state_entries": child_optimizer_entries,
            },
            "output_model": output_manifest,
            "created_model_paths": sorted(map(str, created_model_paths)),
        }
    )
    write_json(summary_path, partial_summary)

    post = evaluate_match(
        learner_player=learner_player,
        learner_model=output_model,
        opponent_model=latest_opponent,
        episodes=args.episodes,
        seed=seeds["evaluation_master_seed"],
        label="POST child_vs_latest",
    )
    retention_parent = evaluate_match(
        learner_player=learner_player,
        learner_model=parent_model,
        opponent_model=previous_opponent,
        episodes=args.episodes,
        seed=seeds["evaluation_master_seed"],
        label="retention parent_vs_previous",
    )
    retention_child = evaluate_match(
        learner_player=learner_player,
        learner_model=output_model,
        opponent_model=previous_opponent,
        episodes=args.episodes,
        seed=seeds["evaluation_master_seed"],
        label="retention child_vs_previous",
    )
    random_result = evaluate_vs_random(
        model_path=output_model,
        agent_player=learner_player,
        episodes=args.episodes,
        master_seed=seeds["random_evaluation_seed"],
        label="child_vs_random",
    )

    latest_delta = learner_win_rate(post, learner_player) - learner_win_rate(
        pre, learner_player
    )
    latest_ci = paired_bootstrap_delta(
        learner_indicators(pre, learner_player),
        learner_indicators(post, learner_player),
    )
    retention_delta = learner_win_rate(
        retention_child, learner_player
    ) - learner_win_rate(retention_parent, learner_player)
    retention_ci = paired_bootstrap_delta(
        learner_indicators(retention_parent, learner_player),
        learner_indicators(retention_child, learner_player),
    )

    input_manifests_after = {
        name: model_manifest(path) for name, path in input_paths.items()
    }
    inputs_unchanged = input_manifests_before == input_manifests_after
    all_results = [pre, post, retention_parent, retention_child, random_result]
    all_masks_clean = all(
        result["mask_violations"] == 0
        and result["nonterminal_all_false_masks"] == 0
        for result in all_results
    )
    training_valid = (
        child_num_timesteps > parent_num_timesteps
        and child_n_updates > parent_n_updates
        and parent_optimizer_entries > 0
        and child_optimizer_entries > 0
        and output_manifest["sha256"]
        != input_manifests_after["parent_model"]["sha256"]
    )
    runner_infrastructure_pass = (
        all(item["status"] == "PASS" for item in regression_results.values())
        and inputs_unchanged
        and all_masks_clean
        and training_valid
        and created_model_paths == {output_model.resolve()}
    )

    if latest_delta > 0 and latest_ci["low"] > 0:
        latest_adaptation = "PASS: clear positive latest-opponent improvement"
        learning_pass = True
    elif latest_delta > 0:
        latest_adaptation = "INCONCLUSIVE: positive estimate, CI includes zero"
        learning_pass = False
    else:
        latest_adaptation = "FAIL: no latest-opponent point-estimate improvement"
        learning_pass = False

    if retention_delta < 0 and retention_ci["high"] < 0:
        forgetting_evidence = (
            "CLEAR: previous-opponent performance decreased with a negative paired CI"
        )
        clear_forgetting = True
    elif retention_delta < 0:
        forgetting_evidence = (
            "INCONCLUSIVE: previous-opponent estimate decreased, CI includes zero"
        )
        clear_forgetting = False
    else:
        forgetting_evidence = (
            "NONE in this matrix: previous-opponent estimate did not decrease"
        )
        clear_forgetting = False

    if learning_pass and not clear_forgetting:
        recommendation = (
            "Use this runner for one explicitly invoked Blue3 generation; keep automatic looping disabled."
        )
    elif learning_pass and clear_forgetting:
        recommendation = (
            "Stop latest-only generations and evaluate an opponent-pool training design."
        )
    else:
        recommendation = (
            "Stop generation creation and review the latest-only training distribution."
        )

    evaluations = {
        "parent_vs_latest": pre,
        "child_vs_latest": post,
        "parent_vs_previous": retention_parent,
        "child_vs_previous": retention_child,
        "child_vs_random": random_result,
    }
    crossplay = {
        "episodes_per_cell": args.episodes,
        "evaluation_master_seed": seeds["evaluation_master_seed"],
        "axis_paths": {
            "parent": str(parent_model),
            "child": str(output_model),
            "previous_opponent": str(previous_opponent),
            "latest_opponent": str(latest_opponent),
        },
        "learner_win_rate_matrix": {
            "parent": {
                "previous_opponent": learner_win_rate(
                    retention_parent, learner_player
                ),
                "latest_opponent": learner_win_rate(pre, learner_player),
            },
            "child": {
                "previous_opponent": learner_win_rate(
                    retention_child, learner_player
                ),
                "latest_opponent": learner_win_rate(post, learner_player),
            },
        },
        "red_win_rate_matrix": {
            "parent": {
                "previous_opponent": retention_parent["red_win_rate"],
                "latest_opponent": pre["red_win_rate"],
            },
            "child": {
                "previous_opponent": retention_child["red_win_rate"],
                "latest_opponent": post["red_win_rate"],
            },
        },
        "blue_win_rate_matrix": {
            "parent": {
                "previous_opponent": retention_parent["blue_win_rate"],
                "latest_opponent": pre["blue_win_rate"],
            },
            "child": {
                "previous_opponent": retention_child["blue_win_rate"],
                "latest_opponent": post["blue_win_rate"],
            },
        },
        "cells": {
            "parent_vs_previous": retention_parent,
            "parent_vs_latest": pre,
            "child_vs_previous": retention_child,
            "child_vs_latest": post,
        },
    }
    summary = {
        **partial_summary,
        "status": "complete",
        "input_models_after": input_manifests_after,
        "input_models_unchanged": inputs_unchanged,
        "evaluations": evaluations,
        "statistics": {
            "latest_delta": latest_delta,
            "latest_delta_bootstrap_95_ci": latest_ci,
            "retention_delta": retention_delta,
            "retention_delta_bootstrap_95_ci": retention_ci,
        },
        "crossplay": crossplay,
        "all_masks_clean": all_masks_clean,
        "runner_infrastructure_pass": runner_infrastructure_pass,
        "learning_result": {
            "latest_adaptation": latest_adaptation,
            "latest_adaptation_pass": learning_pass,
            "forgetting_cycling_evidence": forgetting_evidence,
            "clear_forgetting_cycling_evidence": clear_forgetting,
            "next_recommendation": recommendation,
        },
        "single_generation_stop": True,
        "automatic_next_generation": False,
        "opponent_pool_updated": False,
    }
    write_json(report_dir / "crossplay_matrix.json", crossplay)
    write_json(summary_path, summary)
    (report_dir / "summary.txt").write_text(
        render_summary(summary), encoding="utf-8"
    )
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
    print(f"runner_infrastructure_pass={runner_infrastructure_pass}", flush=True)
    print(f"learning_result={latest_adaptation}", flush=True)
    print("STOP: exactly one child generation completed", flush=True)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run exactly one explicitly configured alternating generation"
    )
    parser.add_argument("--learner-player", required=True, choices=("red", "blue"))
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--latest-opponent", required=True, type=Path)
    parser.add_argument("--previous-opponent", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=70_000)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    runner_cli = [sys.executable, str(Path(__file__).name), *sys.argv[1:]]
    run_one_generation(args, runner_cli)


if __name__ == "__main__":
    main()
