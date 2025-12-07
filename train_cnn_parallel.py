from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gk_env import GreatKingdomEnv
from custom_cnn import SmallCNN
import os

def make_env():
    def _init():
        env = GreatKingdomEnv()
        return Monitor(env)
    return _init


if __name__ == "__main__":
    NUM_ENVS = 8
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=0.0003,
        batch_size=512,
        n_steps=128,
        policy_kwargs=policy_kwargs,
    )

    models_dir = "models/PPO_CNN"
    os.makedirs(models_dir, exist_ok=True)

    TIMESTEPS = 200_000
    iters = 0

    print("=== PPO + CustomCNN + Start parallel environment learning ===")

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        save_path = f"{models_dir}/{TIMESTEPS * iters}"
        model.save(save_path)
        print(f">> Model saved completed: {save_path}")
