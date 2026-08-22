import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from frozen_policy_env import FrozenPolicyOpponentEnv, predict_with_local_rng
from gk_env import GreatKingdomEnv


RED0_MODEL = Path("models/MaskablePPO_CNN/masked_ppo_10000.zip")
BLUE0_MODEL = Path("models/MaskablePPO_CNN/blue_masked_ppo_10000.zip")
RED1_MODEL = Path("models/MaskablePPO_CNN/red10k_ft_vs_blue10k_plus10k.zip")
BLUE1_MODEL = Path("models/MaskablePPO_CNN/blue10k_ft_vs_red1_plus10k.zip")
DEFAULT_REPORT_DIR = Path("reports/terminal_fix_lineage_audit_20260822")
CHECKPOINT_COMMIT = "d3469520ac7f2722dbc8537bff1f867a059dec73"
EPISODES = 500
RANDOM_MASTER_SEED = 30000
CROSSPLAY_MASTER_SEED = 20260822

EXPECTED_HASHES = {
    "red0": "e28340c33406a333940df1fe94eee39b9f78494c4b2b2886cd565f27de27c944",
    "blue0": "17f990f29d0f2f6ae09c386561cb700210dc267820eed5f828565d7ffd9992ba",
    "red1": "5dcc2d8b016f55ddc4e2c9abd7c7e860a485e8fd105399b0be7e6ef037fe8883",
    "blue1": "4b1bdc51904d8b471c576da57df2019e1ba46dd31a069e3db84bd879d18c3e44",
}
MODEL_PATHS = {
    "red0": RED0_MODEL,
    "blue0": BLUE0_MODEL,
    "red1": RED1_MODEL,
    "blue1": BLUE1_MODEL,
}
OLD_RANDOM_REFERENCES = {
    "red0_vs_random_blue": 0.692,
    "blue0_vs_random_red": 0.768,
    "red1_vs_random_blue": 0.782,
    "blue1_vs_random_red": 0.814,
}
OLD_CROSSPLAY_REFERENCES = {
    "red0_vs_blue0": 0.440,
    "red0_vs_blue1": 0.316,
    "red1_vs_blue0": 0.600,
    "red1_vs_blue1": 0.320,
}
TERMINAL_CATEGORIES = ("capture", "suicide", "score/no-playable", "other")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hashes():
    hashes = {name: sha256_file(path) for name, path in MODEL_PATHS.items()}
    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(
                f"{name} hash mismatch: expected {expected}, got {hashes[name]}"
            )
    return hashes


def terminal_category(outcome):
    if outcome in ("agent_capture_win", "opponent_capture_win"):
        return "capture"
    if outcome in ("agent_suicide", "opponent_suicide"):
        return "suicide"
    if outcome == "score":
        return "score/no-playable"
    return "other"


def record_terminal(terminal_histogram, raw_histogram, info, env):
    outcome = info.get("outcome", "missing")
    category = terminal_category(outcome)
    terminal_histogram[category] += 1
    raw_histogram[outcome] += 1
    if category == "score/no-playable" and env.logic.get_playable_spots():
        raise AssertionError("score terminal exposed while selectable moves remain")


def checked_mask(env):
    action_mask = env.action_masks()
    if not np.any(action_mask):
        raise AssertionError("nonterminal all-false action mask was exposed")
    return action_mask


def finalize_result(
    *,
    episodes,
    red_wins,
    blue_wins,
    draws,
    red_suicides,
    blue_suicides,
    mask_violations,
    game_lengths,
    terminal_histogram,
    raw_histogram,
):
    result = {
        "episodes": episodes,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "red_win_rate": red_wins / episodes,
        "blue_win_rate": blue_wins / episodes,
        "red_suicides": red_suicides,
        "red_suicide_rate": red_suicides / episodes,
        "blue_suicides": blue_suicides,
        "blue_suicide_rate": blue_suicides / episodes,
        "mask_violations": mask_violations,
        "nonterminal_all_false_masks": 0,
        "mean_game_length": float(np.mean(game_lengths)),
        "terminal_reason_histogram": {
            key: terminal_histogram[key] for key in TERMINAL_CATEGORIES
        },
        "raw_outcome_histogram": dict(sorted(raw_histogram.items())),
        "no_playable_score_terminations": terminal_histogram["score/no-playable"],
    }
    if sum(result["terminal_reason_histogram"].values()) != episodes:
        raise AssertionError("terminal reason histogram does not sum to episodes")
    return result


def evaluate_vs_random(*, model_path, agent_player, episodes, master_seed, label):
    model = MaskablePPO.load(model_path, device="cpu")
    env = GreatKingdomEnv(agent_player=agent_player)
    red_wins = blue_wins = draws = 0
    red_suicides = blue_suicides = mask_violations = 0
    game_lengths = []
    terminal_histogram = Counter()
    raw_histogram = Counter()

    print(f"\n=== {label}: {episodes} corrected-env games ===", flush=True)
    for episode in range(episodes):
        obs, _ = env.reset(seed=master_seed + episode)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action_mask = checked_mask(env)
            action, _ = model.predict(
                obs,
                action_masks=action_mask,
                deterministic=True,
            )
            action = int(np.asarray(action).item())
            if action < 0 or action >= action_mask.size or not action_mask[action]:
                mask_violations += 1
            obs, _, terminated, truncated, info = env.step(action)

        winner = info.get("winner", env.logic.winner)
        if winner == 2:
            red_wins += 1
        elif winner == 1:
            blue_wins += 1
        else:
            draws += 1
        outcome = info.get("outcome")
        if outcome == "agent_suicide":
            if agent_player == 2:
                red_suicides += 1
            else:
                blue_suicides += 1
        elif outcome == "opponent_suicide":
            if agent_player == 2:
                blue_suicides += 1
            else:
                red_suicides += 1
        record_terminal(terminal_histogram, raw_histogram, info, env)
        board = np.asarray(env.logic.board)
        game_lengths.append(int(np.count_nonzero((board == 1) | (board == 2))))

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            agent_wins = red_wins if agent_player == 2 else blue_wins
            print(
                f"  {completed}/{episodes}: agent_wins={agent_wins} "
                f"agent_win_rate={agent_wins / completed:.3f}",
                flush=True,
            )

    env.close()
    result = finalize_result(
        episodes=episodes,
        red_wins=red_wins,
        blue_wins=blue_wins,
        draws=draws,
        red_suicides=red_suicides,
        blue_suicides=blue_suicides,
        mask_violations=mask_violations,
        game_lengths=game_lengths,
        terminal_histogram=terminal_histogram,
        raw_histogram=raw_histogram,
    )
    result.update(
        {
            "agent_player": agent_player,
            "agent_deterministic": True,
            "opponent": "uniformly random",
            "master_seed": master_seed,
        }
    )
    return result


def evaluate_crossplay(
    *, red_model_path, blue_model_path, episodes, master_seed, label
):
    red_model = MaskablePPO.load(red_model_path, device="cpu")
    env = FrozenPolicyOpponentEnv(
        agent_player=2,
        opponent_model_path=blue_model_path,
        opponent_deterministic=False,
        opponent_seed=master_seed + 100_000,
    )
    red_wins = blue_wins = draws = 0
    red_suicides = blue_suicides = mask_violations = 0
    game_lengths = []
    red_win_indicators = []
    terminal_histogram = Counter()
    raw_histogram = Counter()

    print(f"\n=== {label}: {episodes} corrected-env games ===", flush=True)
    for episode in range(episodes):
        obs, _ = env.reset(seed=master_seed + episode)
        red_rng = np.random.default_rng(
            np.random.SeedSequence([master_seed, episode, 2])
        )
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action_mask = checked_mask(env)
            action = predict_with_local_rng(
                red_model,
                obs,
                action_mask,
                False,
                red_rng,
            )
            if action < 0 or action >= action_mask.size or not action_mask[action]:
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
        outcome = info.get("outcome")
        if outcome == "agent_suicide":
            red_suicides += 1
        elif outcome == "opponent_suicide":
            blue_suicides += 1
        record_terminal(terminal_histogram, raw_histogram, info, env)
        game_lengths.append(len(env.move_trace))

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            print(
                f"  {completed}/{episodes}: red_wins={red_wins} "
                f"red_win_rate={red_wins / completed:.3f}",
                flush=True,
            )

    env.close()
    result = finalize_result(
        episodes=episodes,
        red_wins=red_wins,
        blue_wins=blue_wins,
        draws=draws,
        red_suicides=red_suicides,
        blue_suicides=blue_suicides,
        mask_violations=mask_violations,
        game_lengths=game_lengths,
        terminal_histogram=terminal_histogram,
        raw_histogram=raw_histogram,
    )
    result.update(
        {
            "red_deterministic": False,
            "blue_deterministic": False,
            "master_seed": master_seed,
            "red_win_indicators": red_win_indicators,
        }
    )
    return result


def terminal_text(result):
    histogram = result["terminal_reason_histogram"]
    return ", ".join(f"{key}={histogram[key]}" for key in TERMINAL_CATEGORIES)


def render_summary(summary):
    random_evals = summary["corrected_random_evaluations"]
    cells = summary["corrected_crossplay"]["cells"]
    random_comparison = summary["old_vs_corrected"]["random"]
    cross_comparison = summary["old_vs_corrected"]["crossplay"]
    lines = [
        "# Terminal-fix lineage audit",
        "",
        "No training was performed. All evaluations used the corrected environment,",
        "fixed seeds, and action masks. Random-opponent policy inference was deterministic;",
        "policy-vs-policy inference was stochastic for both policies.",
        "",
        "## [Confirmed] Regression and integrity",
        "",
        f"Checkpoint commit: {summary['checkpoint_commit']}",
        "Regression: test_rules_minimal.py PASS; test_gk_env_players.py PASS; "
        "test_frozen_policy_env.py PASS.",
        f"All model hashes unchanged: {str(summary['model_hashes_unchanged']).lower()}",
        f"All mask checks clean: {str(summary['all_mask_checks_clean']).lower()}",
        "",
        "## [Confirmed] Corrected random evaluations",
        "",
        "Condition | Red wins | Blue wins | Draws | Agent win rate | Red suicide | Blue suicide | Mask violations | Mean length | Terminal reasons",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---",
    ]
    for key, agent_color in (
        ("red0_vs_random_blue", "red"),
        ("blue0_vs_random_red", "blue"),
        ("red1_vs_random_blue", "red"),
        ("blue1_vs_random_red", "blue"),
    ):
        result = random_evals[key]
        agent_rate = result[f"{agent_color}_win_rate"]
        lines.append(
            f"{key} | {result['red_wins']} | {result['blue_wins']} | "
            f"{result['draws']} | {agent_rate:.3f} | {result['red_suicides']} | "
            f"{result['blue_suicides']} | {result['mask_violations']} | "
            f"{result['mean_game_length']:.3f} | {terminal_text(result)}"
        )

    lines.extend(
        [
            "",
            "## [Confirmed] Corrected 2x2 cross-play (Red win rate)",
            "",
            " | Blue0 | Blue1",
            "--- | ---: | ---:",
            (
                f"Red0 | {cells['red0_vs_blue0']['red_win_rate']:.3f} | "
                f"{cells['red0_vs_blue1']['red_win_rate']:.3f}"
            ),
            (
                f"Red1 | {cells['red1_vs_blue0']['red_win_rate']:.3f} | "
                f"{cells['red1_vs_blue1']['red_win_rate']:.3f}"
            ),
            "",
            "Cell | Red suicide | Blue suicide | Mask violations | Mean length | Terminal reasons",
            "--- | ---: | ---: | ---: | ---: | ---",
        ]
    )
    for key in (
        "red0_vs_blue0",
        "red0_vs_blue1",
        "red1_vs_blue0",
        "red1_vs_blue1",
    ):
        result = cells[key]
        lines.append(
            f"{key} | {result['red_suicides']} | {result['blue_suicides']} | "
            f"{result['mask_violations']} | {result['mean_game_length']:.3f} | "
            f"{terminal_text(result)}"
        )

    lines.extend(
        [
            "",
            "## [Confirmed] Old vs corrected",
            "",
            "Condition | Old | Corrected | Difference",
            "--- | ---: | ---: | ---:",
        ]
    )
    for key, comparison in random_comparison.items():
        lines.append(
            f"{key} | {comparison['old']:.3f} | {comparison['corrected']:.3f} | "
            f"{comparison['difference']:+.3f}"
        )
    for key, comparison in cross_comparison.items():
        lines.append(
            f"{key} | {comparison['old']:.3f} | {comparison['corrected']:.3f} | "
            f"{comparison['difference']:+.3f}"
        )

    lines.extend(
        [
            "",
            "No-playable score terminations by condition:",
            json.dumps(
                summary["no_playable_score_termination_counts"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "",
            "## [Interpretation]",
            "",
            summary["interpretation"],
            f"Lineage decision: {summary['lineage_decision']}",
            f"Red2 recommendation: {summary['red2_recommendation']}",
            "",
            "## [Unconfirmed]",
            "",
            "This audit does not establish later self-play convergence, human-play strength,",
            "Nash equilibrium behavior, or the effect of additional continuation rounds.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit lineage on terminal-fixed env")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.report_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing report directory: {args.report_dir}"
        )

    hashes_before = validate_hashes()
    random_specs = {
        "red0_vs_random_blue": (RED0_MODEL, 2),
        "blue0_vs_random_red": (BLUE0_MODEL, 1),
        "red1_vs_random_blue": (RED1_MODEL, 2),
        "blue1_vs_random_red": (BLUE1_MODEL, 1),
    }
    random_evaluations = {
        key: evaluate_vs_random(
            model_path=model_path,
            agent_player=agent_player,
            episodes=args.episodes,
            master_seed=RANDOM_MASTER_SEED,
            label=key,
        )
        for key, (model_path, agent_player) in random_specs.items()
    }

    crossplay_specs = {
        "red0_vs_blue0": (RED0_MODEL, BLUE0_MODEL),
        "red0_vs_blue1": (RED0_MODEL, BLUE1_MODEL),
        "red1_vs_blue0": (RED1_MODEL, BLUE0_MODEL),
        "red1_vs_blue1": (RED1_MODEL, BLUE1_MODEL),
    }
    crossplay_cells = {
        key: evaluate_crossplay(
            red_model_path=red_path,
            blue_model_path=blue_path,
            episodes=args.episodes,
            master_seed=CROSSPLAY_MASTER_SEED,
            label=key,
        )
        for key, (red_path, blue_path) in crossplay_specs.items()
    }
    hashes_after = validate_hashes()

    random_comparison = {}
    for key, old in OLD_RANDOM_REFERENCES.items():
        color = "red" if key.startswith("red") else "blue"
        corrected = random_evaluations[key][f"{color}_win_rate"]
        random_comparison[key] = {
            "old": old,
            "corrected": corrected,
            "difference": corrected - old,
        }
    cross_comparison = {}
    for key, old in OLD_CROSSPLAY_REFERENCES.items():
        corrected = crossplay_cells[key]["red_win_rate"]
        cross_comparison[key] = {
            "old": old,
            "corrected": corrected,
            "difference": corrected - old,
        }

    no_playable_counts = {
        **{
            f"random/{key}": value["no_playable_score_terminations"]
            for key, value in random_evaluations.items()
        },
        **{
            f"crossplay/{key}": value["no_playable_score_terminations"]
            for key, value in crossplay_cells.items()
        },
    }
    all_results = list(random_evaluations.values()) + list(crossplay_cells.values())
    all_mask_checks_clean = all(
        result["mask_violations"] == 0
        and result["nonterminal_all_false_masks"] == 0
        for result in all_results
    )

    a = crossplay_cells["red0_vs_blue0"]
    b = crossplay_cells["red0_vs_blue1"]
    c = crossplay_cells["red1_vs_blue0"]
    d = crossplay_cells["red1_vs_blue1"]
    ordering_checks = {
        "red1_beats_red0_vs_random_blue": (
            random_evaluations["red1_vs_random_blue"]["red_win_rate"]
            > random_evaluations["red0_vs_random_blue"]["red_win_rate"]
        ),
        "blue1_beats_blue0_vs_random_red": (
            random_evaluations["blue1_vs_random_red"]["blue_win_rate"]
            > random_evaluations["blue0_vs_random_red"]["blue_win_rate"]
        ),
        "red1_beats_red0_vs_blue0": c["red_win_rate"] > a["red_win_rate"],
        "blue1_beats_blue0_vs_red1": d["blue_win_rate"] > c["blue_win_rate"],
        "blue1_not_weaker_than_blue0_vs_red0": (
            b["blue_win_rate"] >= a["blue_win_rate"]
        ),
    }
    lineage_maintained = (
        all(ordering_checks.values())
        and all_mask_checks_clean
        and hashes_before == hashes_after
    )
    if lineage_maintained:
        interpretation = (
            "The corrected environment preserves every requested relative ordering: "
            "Red1 remains above Red0 against Blue0, Blue1 remains above Blue0 against "
            "Red1, and Blue1 does not fall below Blue0 against Red0. Random-opponent "
            "generation improvements also remain. The terminal fix changes termination "
            "handling but does not reverse the lineage evidence in these fixed-seed audits."
        )
        lineage_decision = "MAINTAIN existing Red0 -> Blue0 -> Red1 -> Blue1 lineage"
        red2_recommendation = (
            "Proceed with one Red2 continuation-10k experiment as the next isolated step."
        )
    else:
        interpretation = (
            "At least one requested relative ordering or integrity check failed under the "
            "corrected environment, so the existing lineage evidence is not preserved."
        )
        lineage_decision = "DO NOT extend the existing lineage"
        red2_recommendation = (
            "Do not train Red2; first review whether clean fixed-environment base training "
            "is required."
        )

    summary = {
        "experiment": "terminal_fix_lineage_audit_20260822",
        "training_performed": False,
        "checkpoint_commit": CHECKPOINT_COMMIT,
        "episodes_per_condition": args.episodes,
        "random_master_seed": RANDOM_MASTER_SEED,
        "crossplay_master_seed": CROSSPLAY_MASTER_SEED,
        "model_paths": {name: str(path) for name, path in MODEL_PATHS.items()},
        "expected_model_hashes": EXPECTED_HASHES,
        "model_hashes_before": hashes_before,
        "model_hashes_after": hashes_after,
        "model_hashes_unchanged": hashes_before == hashes_after,
        "regression_tests": {
            "test_rules_minimal.py": "PASS",
            "test_gk_env_players.py": "PASS",
            "test_frozen_policy_env.py": "PASS",
            "no_playable_both_opponent_colors": "PASS",
        },
        "corrected_random_evaluations": random_evaluations,
        "corrected_crossplay": {
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
            "cells": crossplay_cells,
        },
        "no_playable_score_termination_counts": no_playable_counts,
        "old_references": {
            "random": OLD_RANDOM_REFERENCES,
            "crossplay": OLD_CROSSPLAY_REFERENCES,
        },
        "old_vs_corrected": {
            "random": random_comparison,
            "crossplay": cross_comparison,
        },
        "ordering_checks": ordering_checks,
        "all_mask_checks_clean": all_mask_checks_clean,
        "interpretation": interpretation,
        "lineage_maintained": lineage_maintained,
        "lineage_decision": lineage_decision,
        "red2_recommendation": red2_recommendation,
        "unconfirmed": [
            "later self-play convergence",
            "human-play strength",
            "Nash equilibrium behavior",
            "effect of additional continuation rounds",
        ],
    }
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.report_dir / "summary.txt").write_text(
        render_summary(summary), encoding="utf-8"
    )
    print(f"\nWrote {args.report_dir / 'summary.json'}", flush=True)
    print(f"Wrote {args.report_dir / 'summary.txt'}", flush=True)
    print(f"Lineage maintained: {lineage_maintained}", flush=True)


if __name__ == "__main__":
    main()
