from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_cnn import SmallCNN
from frozen_policy_env import FrozenPolicyOpponentEnv


TOTAL_TIMESTEPS = 10_000
BLUE_MODEL_PATH = Path("models/MaskablePPO_CNN/blue_masked_ppo_10000.zip")
OUTPUT_MODEL_PATH = Path(
    "models/MaskablePPO_CNN/red_br_vs_blue10k_10000.zip"
)


def make_env(rank: int, base_seed: int = 20260822):
    def _init():
        env = FrozenPolicyOpponentEnv(
            agent_player=2,
            opponent_model_path=BLUE_MODEL_PATH,
            opponent_deterministic=False,
            opponent_seed=base_seed + 10_000 + rank,
        )
        env.reset(seed=base_seed + rank)
        return Monitor(env)

    return _init


if __name__ == "__main__":
    if not BLUE_MODEL_PATH.is_file():
        raise FileNotFoundError(BLUE_MODEL_PATH)
    if OUTPUT_MODEL_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite existing model: {OUTPUT_MODEL_PATH}"
        )

    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )
    # A new policy is trained from scratch. The old Red model is an evaluation
    # baseline only and is never loaded for continuation training.
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
        print(
            "=== Red best-response 10k vs stochastic frozen Blue 10k ===",
            flush=True,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(OUTPUT_MODEL_PATH)
        print(f"saved_model={OUTPUT_MODEL_PATH}", flush=True)
        print(f"requested_timesteps={TOTAL_TIMESTEPS}", flush=True)
        print(f"stored_num_timesteps={model.num_timesteps}", flush=True)
    finally:
        env.close()
