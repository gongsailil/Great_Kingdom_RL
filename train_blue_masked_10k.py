from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_cnn import SmallCNN
from gk_env import GreatKingdomEnv


TOTAL_TIMESTEPS = 10_000
MODEL_PATH = Path("models/MaskablePPO_CNN/blue_masked_ppo_10000.zip")


def make_env(rank: int, base_seed: int = 20260822):
    def _init():
        env = GreatKingdomEnv(agent_player=1)
        env.reset(seed=base_seed + rank)
        return Monitor(env)

    return _init


if __name__ == "__main__":
    if MODEL_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing model: {MODEL_PATH}")

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

    try:
        print("=== Blue MaskablePPO 10k vs random Red ===", flush=True)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(MODEL_PATH)
        print(f"saved_model={MODEL_PATH}", flush=True)
        print(f"requested_timesteps={TOTAL_TIMESTEPS}", flush=True)
        print(f"stored_num_timesteps={model.num_timesteps}", flush=True)
    finally:
        env.close()
