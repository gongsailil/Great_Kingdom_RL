import argparse
from collections import Counter

import numpy as np
from sb3_contrib import MaskablePPO

from gk_env import GreatKingdomEnv


def evaluate(model_path: str, episodes: int, seed: int):
    model = MaskablePPO.load(model_path)
    env = GreatKingdomEnv()

    wins = 0
    losses = 0
    draws = 0
    suicides = 0
    mask_violations = 0
    first_moves = Counter()
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            mask = env.action_masks()
            action, _ = model.predict(
                obs,
                action_masks=mask,
                deterministic=True,
            )
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)

        winner = info.get("winner")
        if winner == env.agent_player:
            wins += 1
        elif winner == env.opponent_player:
            losses += 1
        else:
            draws += 1

        if info.get("outcome") == "agent_suicide":
            suicides += 1
        if info.get("outcome") == "mask_violation":
            mask_violations += 1

        first = info.get("first_agent_action")
        if first is not None:
            first_moves[first] += 1
        lengths.append(info.get("agent_moves", 0))

    print(f"episodes={episodes}")
    print(f"wins={wins} losses={losses} draws={draws} win_rate={wins / episodes:.3f}")
    print(f"agent_suicide_rate={suicides / episodes:.3f}")
    print(f"mask_violations={mask_violations}")
    print(f"mean_agent_moves={np.mean(lengths):.2f}")
    print("top_first_moves:")
    for action, count in first_moves.most_common(10):
        x = action % env.board_size
        y = action // env.board_size
        print(f"  ({x},{y}) action={action}: {count}/{episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=30000)
    args = parser.parse_args()
    evaluate(args.model_path, args.episodes, args.seed)
