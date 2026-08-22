from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_cnn import SmallCNN
from gk_env import GreatKingdomEnv


def make_env(rank: int, base_seed: int = 20260822):
    def _init():
        env = GreatKingdomEnv()
        env.reset(seed=base_seed + rank)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    model = MaskablePPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="auto",
        learning_rate=3e-4,
        batch_size=512,
        n_steps=128,
        policy_kwargs=policy_kwargs,
        seed=20260822,
    )

    models_dir = Path("models/MaskablePPO_CNN")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Minimal E2E experiment first. Do not start an infinite training loop.
    total_timesteps = 200_000
    print("=== Minimal experiment: MaskablePPO vs random opponent ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(models_dir / f"masked_ppo_{total_timesteps}")
    env.close()
