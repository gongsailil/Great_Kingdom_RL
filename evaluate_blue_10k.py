import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from gk_env import GreatKingdomEnv


AGENT_RANDOM = "random_blue"
AGENT_PPO = "blue_ppo_10k"
DEFAULT_REPORT_DIR = Path("reports/blue_10k_random_baseline_20260822")


def evaluate_condition(
    *,
    agent_kind,
    episodes,
    opponent_seed,
    agent_seed,
    model=None,
):
    if agent_kind == AGENT_PPO and model is None:
        raise ValueError("PPO evaluation requires a loaded model")

    env = GreatKingdomEnv(agent_player=1)
    wins = 0
    losses = 0
    draws = 0
    suicides = 0
    mask_violations = 0
    agent_move_counts = []
    first_moves = Counter()

    print(f"\n=== {agent_kind}_vs_random_red ===", flush=True)
    for episode in range(episodes):
        # The environment RNG is used only by Random Red. Random Blue gets a
        # separate per-episode RNG so agent/opponent sampling is not coupled.
        obs, _ = env.reset(seed=opponent_seed + episode)
        agent_rng = np.random.default_rng(agent_seed + episode)
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            mask = env.action_masks()
            if agent_kind == AGENT_RANDOM:
                selectable = np.flatnonzero(mask)
                if selectable.size == 0:
                    raise RuntimeError("Blue has no selectable action before termination")
                action = int(agent_rng.choice(selectable))
            else:
                action, _ = model.predict(
                    obs,
                    action_masks=mask,
                    deterministic=True,
                )
                action = int(np.asarray(action).item())

            if not bool(mask[action]):
                mask_violations += 1
            obs, _, terminated, truncated, info = env.step(action)

        winner = info.get("winner", env.logic.winner)
        if winner == env.agent_player:
            wins += 1
        elif winner == env.opponent_player:
            losses += 1
        else:
            draws += 1

        if info.get("outcome") == "agent_suicide":
            suicides += 1
        if env.first_agent_action is not None:
            first_moves[env.first_agent_action] += 1
        agent_move_counts.append(env.agent_moves)

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            print(
                f"  {completed}/{episodes}: wins={wins} "
                f"win_rate={wins / completed:.3f}",
                flush=True,
            )

    histogram = {
        f"{action % env.board_size},{action // env.board_size}": count
        for action, count in sorted(first_moves.items())
    }
    top_first_moves = [
        {
            "coordinate": [action % env.board_size, action // env.board_size],
            "action": action,
            "count": count,
            "frequency": count / episodes,
        }
        for action, count in first_moves.most_common(10)
    ]
    result = {
        "condition": f"{agent_kind}_vs_random_red",
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
        "agent_suicides": suicides,
        "agent_suicide_rate": suicides / episodes,
        "mask_violations": mask_violations,
        "mean_agent_moves": float(np.mean(agent_move_counts)),
        "first_move_coordinate_histogram": histogram,
        "top_first_moves": top_first_moves,
    }
    env.close()
    return result


def render_condition(result):
    lines = [
        f"episodes={result['episodes']}",
        (
            f"wins={result['wins']} losses={result['losses']} "
            f"draws={result['draws']} win_rate={result['win_rate']:.3f}"
        ),
        f"agent_suicide_rate={result['agent_suicide_rate']:.3f}",
        f"mask_violations={result['mask_violations']}",
        f"mean_agent_moves={result['mean_agent_moves']:.2f}",
        "top_first_moves:",
    ]
    for move in result["top_first_moves"]:
        x, y = move["coordinate"]
        lines.append(f"  ({x},{y}): {move['count']}/{result['episodes']}")
    return "\n".join(lines)


def render_summary(summary):
    baseline = summary["blue_random_baseline"]
    ppo = summary["blue_ppo_10k"]
    lines = [
        "# Blue 10k MaskablePPO vs Random Red",
        "",
        "This is a single-agent-vs-random prerequisite experiment, not self-play.",
        "Blue is the agent and moves first; Random Red retains komi=3.",
        "Observations are canonicalized to the agent perspective.",
        "",
        f"episodes per condition: {summary['configuration']['episodes_per_condition']}",
        f"opponent seed base: {summary['configuration']['opponent_seed_base']}",
        f"random Blue seed base: {summary['configuration']['agent_seed_base']}",
        f"model: {summary['configuration']['model_path']}",
        (
            "training timesteps: requested "
            f"{summary['configuration']['training_requested_timesteps']}, stored counter "
            f"{summary['configuration']['training_stored_num_timesteps']}"
        ),
        "PPO deterministic: true",
        "PPO action mask: enabled",
        "",
        "## Results",
        "",
        (
            "condition | episodes | wins | losses | draws | win_rate | "
            "suicide_rate | mask_violations | mean_agent_moves"
        ),
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
        (
            f"Random Blue vs Random Red | {baseline['episodes']} | {baseline['wins']} | "
            f"{baseline['losses']} | {baseline['draws']} | {baseline['win_rate']:.3f} | "
            f"{baseline['agent_suicide_rate']:.3f} | {baseline['mask_violations']} | "
            f"{baseline['mean_agent_moves']:.2f}"
        ),
        (
            f"Blue PPO 10k vs Random Red | {ppo['episodes']} | {ppo['wins']} | "
            f"{ppo['losses']} | {ppo['draws']} | {ppo['win_rate']:.3f} | "
            f"{ppo['agent_suicide_rate']:.3f} | {ppo['mask_violations']} | "
            f"{ppo['mean_agent_moves']:.2f}"
        ),
        "",
        f"win-rate delta (PPO - random baseline): {summary['win_rate_delta']:+.3f}",
        (
            "criterion (Blue PPO win rate > Random Blue baseline): "
            f"{'PASS' if summary['criterion_passed'] else 'FAIL'}"
        ),
        "",
        "## Random Blue first moves",
        "",
    ]
    for move in baseline["top_first_moves"]:
        x, y = move["coordinate"]
        lines.append(f"({x},{y}): {move['count']}/{baseline['episodes']}")

    lines.extend(["", "## Blue PPO first moves", ""])
    for move in ppo["top_first_moves"]:
        x, y = move["coordinate"]
        lines.append(f"({x},{y}): {move['count']}/{ppo['episodes']}")
    lines.append("")
    return "\n".join(lines)


def run_baseline(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    existing = [path for path in (json_path, text_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing reports: {existing}")

    baseline = evaluate_condition(
        agent_kind=AGENT_RANDOM,
        episodes=args.episodes,
        opponent_seed=args.opponent_seed,
        agent_seed=args.agent_seed,
    )
    partial_summary = {
        "status": "baseline_complete",
        "configuration": {
            "episodes_per_condition": args.episodes,
            "opponent_seed_base": args.opponent_seed,
            "agent_seed_base": args.agent_seed,
        },
        "blue_random_baseline": baseline,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(partial_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n" + render_condition(baseline), flush=True)
    print(f"Saved baseline checkpoint: {json_path}", flush=True)


def run_ppo(args):
    report_dir = Path(args.report_dir)
    json_path = report_dir / "summary.json"
    text_path = report_dir / "summary.txt"
    if not json_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {json_path}")
    if text_path.exists():
        raise FileExistsError(f"refusing to overwrite completed report: {text_path}")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "baseline_complete":
        raise ValueError("summary.json is not a baseline-only checkpoint")
    expected = summary["configuration"]
    if (
        expected["episodes_per_condition"] != args.episodes
        or expected["opponent_seed_base"] != args.opponent_seed
        or expected["agent_seed_base"] != args.agent_seed
    ):
        raise ValueError("PPO evaluation settings do not match the baseline")

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = MaskablePPO.load(model_path)
    ppo = evaluate_condition(
        agent_kind=AGENT_PPO,
        episodes=args.episodes,
        opponent_seed=args.opponent_seed,
        agent_seed=args.agent_seed,
        model=model,
    )

    baseline = summary["blue_random_baseline"]
    delta = ppo["win_rate"] - baseline["win_rate"]
    summary.update(
        {
            "status": "complete",
            "experiment": "Blue agent 10k MaskablePPO vs Random Red",
            "scope": "Single agent vs random; not self-play.",
            "blue_ppo_10k": ppo,
            "win_rate_delta": delta,
            "criterion": "Blue PPO win rate > Random Blue baseline win rate",
            "criterion_passed": delta > 0,
        }
    )
    summary["configuration"].update(
        {
            "agent_player": 1,
            "opponent_player": 2,
            "red_komi": 3.0,
            "model_path": str(model_path),
            "training_requested_timesteps": 10_000,
            "training_stored_num_timesteps": model.num_timesteps,
            "ppo_deterministic": True,
            "ppo_action_mask": True,
        }
    )
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(render_summary(summary), encoding="utf-8")
    print("\n" + render_condition(ppo), flush=True)
    print(f"win_rate_delta={delta:+.3f}", flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {text_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("baseline", "ppo"):
        subparser = subparsers.add_parser(phase)
        subparser.add_argument("--episodes", type=int, default=500)
        subparser.add_argument("--opponent-seed", type=int, default=30000)
        subparser.add_argument("--agent-seed", type=int, default=40000)
        subparser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    subparsers.choices["ppo"].add_argument(
        "--model-path",
        default="models/MaskablePPO_CNN/blue_masked_ppo_10000.zip",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.phase == "baseline":
        run_baseline(args)
    else:
        run_ppo(args)


if __name__ == "__main__":
    main()
