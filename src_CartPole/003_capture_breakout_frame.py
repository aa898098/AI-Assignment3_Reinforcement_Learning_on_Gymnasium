import os

import cv2
import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper


gym.register_envs(ale_py)


OUTPUT_DIR = r"D:/YS/assim3/Breakout_Figures"
MODEL_PATH = r"D:/YS/assim3/06/models/ppo_breakout_lr_1e-4_further_final"


def make_env():

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="rgb_array",
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

    terminated = False
    truncated = False

    best_frame = None

    max_steps = 3000

    for step in range(max_steps):

        action, _states = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()

        if frame is not None:

            # Prefer frames where game objects exist
            if reward > 0 or step > 500:
                best_frame = frame.copy()

        if terminated or truncated:
            obs, info = env.reset()

    if best_frame is not None:

        output_path = os.path.join(
            OUTPUT_DIR,
            "breakout_example_frame.png"
        )

        # RGB -> BGR
        best_frame = cv2.cvtColor(
            best_frame,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(output_path, best_frame)

        print(f"Saved screenshot to: {output_path}")

    else:
        print("No valid frame captured.")

    env.close()


if __name__ == "__main__":
    main()