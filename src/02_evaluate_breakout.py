import os

import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper


gym.register_envs(ale_py)


OUTPUT_DIR = r"D:/YS/assim3/02"
MODEL_PATH = r"D:/YS/assim3/01/ppo_breakout_smoke_test.zip"


def make_env():

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode=None,
        frameskip=1,
        repeat_action_probability=0.0,
    )

    env = AtariWrapper(env)

    return env


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = make_env()

    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()

    total_reward = 0

    max_steps = 3000

    terminated = False
    truncated = False

    for step in range(max_steps):

        action, _states = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()

    result_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_result.txt"
    )

    with open(result_path, "w") as f:
        f.write(f"Total Reward = {total_reward}\n")

    print(f"Total Reward = {total_reward}")

    env.close()

    print("Evaluation finished.")


if __name__ == "__main__":
    main()