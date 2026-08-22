import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from gk_env import GreatKingdomEnv
RANDOM_OPENING = "random"
CORNER_OPENING = "corner_opening_random"
AGENT_RANDOM = "random"
AGENT_PPO = "ppo_10k"


def corner_distance(x: int, y: int, board_size: int) -> int:
    edge = board_size - 1
    return min(
        x + y,
        (edge - x) + y,
        x + (edge - y),
        (edge - x) + (edge - y),
    )


class OpeningBenchmarkEnv(GreatKingdomEnv):
    """GreatKingdomEnv with an optional first-Blue-move distribution shift.

    In corner_opening_random mode only Blue move #1 changes. Every later Blue
    move delegates to the original uniformly random opponent implementation.
    This is a synthetic distribution-shift benchmark, not a human opening
    model or an all-game corner heuristic.
    """

    def __init__(self, opening_policy: str):
        if opening_policy not in (RANDOM_OPENING, CORNER_OPENING):
            raise ValueError(f"unknown opening policy: {opening_policy}")
        self.opening_policy = opening_policy
        self._opponent_move_number = 0
        self.first_opponent_move = None
        super().__init__()

    def reset(self, seed=None, options=None):
        self._opponent_move_number = 0
        self.first_opponent_move = None
        return super().reset(seed=seed, options=options)

    def _opponent_move_random(self):
        is_opening = self._opponent_move_number == 0

        if is_opening and self.opening_policy == CORNER_OPENING:
            result = self._opponent_move_corner_biased()
        else:
            result = super()._opponent_move_random()

        if is_opening:
            blue_stones = [
                (x, y)
                for y in range(self.board_size)
                for x in range(self.board_size)
                if self.logic.board[y][x] == self.opponent_player
            ]
            if len(blue_stones) != 1:
                raise RuntimeError(
                    f"expected exactly one Blue opening stone, got {blue_stones}"
                )
            self.first_opponent_move = blue_stones[0]

        self._opponent_move_number += 1
        return result

    def _opponent_move_corner_biased(self):
        if self.logic.game_over:
            return self.logic.last_move_result

        # get_playable_spots excludes occupied and territory-forbidden moves;
        # selectable suicide moves remain candidates under the engine rules.
        playable = self.logic.get_playable_spots()
        if not playable:
            self.logic.check_game_end_simple()
            return self.logic.last_move_result

        weights = np.asarray(
            [
                1.0 / (1.0 + corner_distance(x, y, self.board_size))
                for x, y in playable
            ],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
        idx = int(self.np_random.choice(len(playable), p=probabilities))
        x, y = playable[idx]
        return self.logic.place_stone_detailed(x, y)


def select_agent_action(agent_kind, model, obs, mask, agent_rng):
    if agent_kind == AGENT_RANDOM:
        selectable = np.flatnonzero(mask)
        if selectable.size == 0:
            raise RuntimeError("Red has no selectable action before termination")
        return int(agent_rng.choice(selectable))

    action, _ = model.predict(
        obs,
        action_masks=mask,
        deterministic=True,
    )
    return int(np.asarray(action).item())


def summarize_openings(first_moves, board_size):
    histogram = Counter(first_moves)
    total = sum(histogram.values())
    mean_distance = (
        sum(corner_distance(x, y, board_size) * count for (x, y), count in histogram.items())
        / total
    )

    ordered_histogram = {
        f"{x},{y}": histogram[(x, y)]
        for x, y in sorted(histogram, key=lambda coordinate: (coordinate[1], coordinate[0]))
    }
    top_moves = [
        {
            "coordinate": [x, y],
            "count": count,
            "frequency": count / total,
            "d_corner": corner_distance(x, y, board_size),
        }
        for (x, y), count in histogram.most_common(10)
    ]
    return {
        "first_move_coordinate_histogram": ordered_histogram,
        "mean_d_corner": mean_distance,
        "top_first_moves": top_moves,
    }


def evaluate_condition(
    *,
    agent_kind,
    opening_policy,
    model,
    episodes,
    opponent_seed,
    agent_seed,
):
    env = OpeningBenchmarkEnv(opening_policy)
    wins = 0
    losses = 0
    draws = 0
    suicides = 0
    mask_violations = 0
    agent_move_counts = []
    first_opponent_moves = []

    label = f"{agent_kind}_red_vs_{opening_policy}_blue"
    print(f"\n=== {label} ===", flush=True)

    for episode in range(episodes):
        # Blue/opponent randomness and Red/random-agent randomness use separate
        # per-episode streams. Reusing the same seed sequences across conditions
        # makes each opening-policy pair directly reproducible.
        obs, _ = env.reset(seed=opponent_seed + episode)
        agent_rng = np.random.default_rng(agent_seed + episode)
        first_opponent_moves.append(env.first_opponent_move)

        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            mask = env.action_masks()
            action = select_agent_action(agent_kind, model, obs, mask, agent_rng)
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
        agent_move_counts.append(env.agent_moves)

        completed = episode + 1
        if completed % 100 == 0 or completed == episodes:
            print(
                f"  {completed}/{episodes}: wins={wins} "
                f"win_rate={wins / completed:.3f}",
                flush=True,
            )

    result = {
        "condition": label,
        "red_agent": agent_kind,
        "blue_opponent": opening_policy,
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
        "agent_suicides": suicides,
        "agent_suicide_rate": suicides / episodes,
        "mask_violations": mask_violations,
        "mean_agent_moves": float(np.mean(agent_move_counts)),
    }
    result.update(summarize_openings(first_opponent_moves, env.board_size))
    env.close()
    return result


def render_summary(summary):
    lines = [
        "# 10k PPO corner-opening distribution-shift benchmark",
        "",
        (
            "Blue move #1 only uses the corner-biased prior "
            "weight=1/(1+d_corner); later Blue moves are uniformly random."
        ),
        "This is a synthetic benchmark, not a human opening model.",
        "",
        f"episodes per condition: {summary['configuration']['episodes_per_condition']}",
        f"opponent seed base: {summary['configuration']['opponent_seed_base']}",
        f"random Red seed base: {summary['configuration']['agent_seed_base']}",
        f"model: {summary['configuration']['model_path']}",
        "PPO deterministic: true",
        "PPO action mask: enabled",
        "",
        "## 2x2 results",
        "",
        (
            "condition | episodes | wins | losses | draws | win_rate | "
            "agent_suicide_rate | mask_violations | mean_agent_moves"
        ),
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]

    for result in summary["conditions"]:
        lines.append(
            f"{result['condition']} | {result['episodes']} | {result['wins']} | "
            f"{result['losses']} | {result['draws']} | {result['win_rate']:.3f} | "
            f"{result['agent_suicide_rate']:.3f} | {result['mask_violations']} | "
            f"{result['mean_agent_moves']:.2f}"
        )

    lines.extend(["", "## Blue first-move distributions", ""])
    for opening_policy, distribution in summary["opening_distributions"].items():
        lines.append(f"{opening_policy}: mean_d_corner={distribution['mean_d_corner']:.3f}")
        for move in distribution["top_first_moves"]:
            x, y = move["coordinate"]
            lines.append(
                f"  ({x},{y}): {move['count']} "
                f"({move['frequency']:.3f}), d_corner={move['d_corner']}"
            )
        lines.append("")

    comparison = summary["comparisons"]
    lines.extend(
        [
            "## Comparison",
            "",
            (
                "PPO advantage over random Red vs random Blue: "
                f"{comparison['ppo_advantage_random_opening']:+.3f}"
            ),
            (
                "PPO advantage over random Red vs corner-opening Blue: "
                f"{comparison['ppo_advantage_corner_opening']:+.3f}"
            ),
            (
                "PPO win-rate change (corner-opening minus random Blue): "
                f"{comparison['ppo_corner_minus_random_opening']:+.3f}"
            ),
            (
                "Criterion (PPO beats random Red under corner opening): "
                f"{'PASS' if comparison['criterion_passed'] else 'FAIL'}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="models/MaskablePPO_CNN/masked_ppo_10000.zip",
    )
    parser.add_argument("--episodes", type=int, default=500)
    # Keep the opponent seed aligned with evaluate_masked.py so the existing
    # 10k-vs-random result remains directly comparable. Random Red uses a
    # separate stream to avoid coupling agent and opponent randomness.
    parser.add_argument("--opponent-seed", type=int, default=30000)
    parser.add_argument("--agent-seed", type=int, default=40000)
    parser.add_argument(
        "--output-dir",
        default="reports/generalization_10k_corner_opening_20260822",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output_dir = Path(args.output_dir)
    json_path = output_dir / "summary.json"
    text_path = output_dir / "summary.txt"
    existing = [path for path in (json_path, text_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing reports: {existing}")

    print(f"Loading model without training: {model_path}", flush=True)
    model = MaskablePPO.load(model_path)

    condition_specs = [
        (AGENT_RANDOM, RANDOM_OPENING),
        (AGENT_PPO, RANDOM_OPENING),
        (AGENT_RANDOM, CORNER_OPENING),
        (AGENT_PPO, CORNER_OPENING),
    ]
    conditions = [
        evaluate_condition(
            agent_kind=agent_kind,
            opening_policy=opening_policy,
            model=model,
            episodes=args.episodes,
            opponent_seed=args.opponent_seed,
            agent_seed=args.agent_seed,
        )
        for agent_kind, opening_policy in condition_specs
    ]

    by_key = {
        (result["red_agent"], result["blue_opponent"]): result
        for result in conditions
    }
    random_distribution = {
        key: by_key[(AGENT_RANDOM, RANDOM_OPENING)][key]
        for key in (
            "first_move_coordinate_histogram",
            "mean_d_corner",
            "top_first_moves",
        )
    }
    corner_distribution = {
        key: by_key[(AGENT_RANDOM, CORNER_OPENING)][key]
        for key in (
            "first_move_coordinate_histogram",
            "mean_d_corner",
            "top_first_moves",
        )
    }

    # With identical per-episode opponent seeds, the opening histogram must not
    # depend on whether Red is random or PPO.
    for opening_policy in (RANDOM_OPENING, CORNER_OPENING):
        random_red_histogram = by_key[(AGENT_RANDOM, opening_policy)][
            "first_move_coordinate_histogram"
        ]
        ppo_red_histogram = by_key[(AGENT_PPO, opening_policy)][
            "first_move_coordinate_histogram"
        ]
        if random_red_histogram != ppo_red_histogram:
            raise AssertionError(
                f"Blue opening histogram changed with Red agent: {opening_policy}"
            )

    random_random = by_key[(AGENT_RANDOM, RANDOM_OPENING)]["win_rate"]
    ppo_random = by_key[(AGENT_PPO, RANDOM_OPENING)]["win_rate"]
    random_corner = by_key[(AGENT_RANDOM, CORNER_OPENING)]["win_rate"]
    ppo_corner = by_key[(AGENT_PPO, CORNER_OPENING)]["win_rate"]
    summary = {
        "benchmark": "corner-biased Blue opening distribution shift",
        "scope": (
            "Only Blue move #1 changes; this is not a human opening model "
            "and not an all-game corner heuristic."
        ),
        "configuration": {
            "episodes_per_condition": args.episodes,
            "model_path": str(model_path),
            "ppo_deterministic": True,
            "ppo_action_mask": True,
            "opponent_seed_base": args.opponent_seed,
            "agent_seed_base": args.agent_seed,
            "corner_weight": "1 / (1 + d_corner)",
        },
        "conditions": conditions,
        "opening_distributions": {
            RANDOM_OPENING: random_distribution,
            CORNER_OPENING: corner_distribution,
        },
        "opening_histograms_match_across_red_agents": True,
        "comparisons": {
            "ppo_advantage_random_opening": ppo_random - random_random,
            "ppo_advantage_corner_opening": ppo_corner - random_corner,
            "ppo_corner_minus_random_opening": ppo_corner - ppo_random,
            "random_red_corner_minus_random_opening": random_corner - random_random,
            "criterion": "PPO win rate vs corner-opening Blue > random Red win rate",
            "criterion_passed": ppo_corner > random_corner,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(render_summary(summary), encoding="utf-8")
    print(f"\nSaved {json_path}", flush=True)
    print(f"Saved {text_path}", flush=True)
    print("\n" + render_summary(summary), flush=True)


if __name__ == "__main__":
    main()
