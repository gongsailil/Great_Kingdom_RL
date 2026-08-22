from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from frozen_policy_env import FrozenPolicyOpponentEnv


ADDITIONAL_TIMESTEPS = 10_000
BLUE1_MODEL_PATH = Path(
    "models/MaskablePPO_CNN/blue10k_ft_vs_red1_plus10k.zip"
)
RED2_MODEL_PATH = Path(
    "models/MaskablePPO_CNN/red2_ft_vs_blue1_plus10k.zip"
)
OUTPUT_MODEL_PATH = Path(
    "models/MaskablePPO_CNN/blue2_ft_vs_red2_plus10k.zip"
)


def make_env(rank: int, base_seed: int = 20260822):
    def _init():
        env = FrozenPolicyOpponentEnv(
            agent_player=1,
            opponent_model_path=RED2_MODEL_PATH,
            opponent_deterministic=False,
            opponent_seed=base_seed + 10_000 + rank,
        )
        env.reset(seed=base_seed + rank)
        return Monitor(env)

    return _init


if __name__ == "__main__":
    for model_path in (BLUE1_MODEL_PATH, RED2_MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    if OUTPUT_MODEL_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite existing model: {OUTPUT_MODEL_PATH}"
        )

    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    try:
        # Loading Blue1 restores its policy, optimizer, schedules, and counters.
        # Only Blue is updated; Red2 remains an environment-owned frozen model.
        model = MaskablePPO.load(
            BLUE1_MODEL_PATH,
            env=env,
            device="auto",
        )
        optimizer_state = model.policy.optimizer.state_dict()["state"]
        if not optimizer_state:
            raise RuntimeError("Blue1 optimizer state was not restored")

        print("=== Blue2 continuation +10k vs stochastic frozen Red2 ===", flush=True)
        print(f"start_num_timesteps={model.num_timesteps}", flush=True)
        print(f"start_n_updates={model._n_updates}", flush=True)
        print(
            f"restored_optimizer_state_entries={len(optimizer_state)}",
            flush=True,
        )

        model.learn(
            total_timesteps=ADDITIONAL_TIMESTEPS,
            reset_num_timesteps=False,
        )
        model.save(OUTPUT_MODEL_PATH)
        print(f"saved_model={OUTPUT_MODEL_PATH}", flush=True)
        print(f"requested_additional_timesteps={ADDITIONAL_TIMESTEPS}", flush=True)
        print(f"end_num_timesteps={model.num_timesteps}", flush=True)
        print(f"end_n_updates={model._n_updates}", flush=True)
    finally:
        env.close()
