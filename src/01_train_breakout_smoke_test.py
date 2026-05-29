import os

import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

gym.register_envs(ale_py)


OUTPUT_DIR = r"D:/YS/assim3/01"


def make_breakout_env():
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = make_breakout_env()

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
    )

    model.learn(total_timesteps=10_000)

    model.save(os.path.join(OUTPUT_DIR, "ppo_breakout_smoke_test"))

    env.close()

    print("Smoke test finished successfully.")


if __name__ == "__main__":
    main()