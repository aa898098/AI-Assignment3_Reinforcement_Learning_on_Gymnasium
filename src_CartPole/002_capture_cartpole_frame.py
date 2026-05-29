import os

import cv2
import gymnasium as gym
from stable_baselines3 import PPO


OUTPUT_DIR = r"D:/YS/assim3/CartPole_Figures"
MODEL_PATH = r"D:/YS/assim3/CartPole/001/models/ppo_cartpole_baseline_final"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = gym.make(
        "CartPole-v1",
        render_mode="rgb_array",
    )

    model = PPO.load(MODEL_PATH)

    obs, info = env.reset(seed=414551032)

    selected_frame = None

    for step in range(200):
        action, _states = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()

        if step == 80:
            selected_frame = frame.copy()
            break

        if terminated or truncated:
            obs, info = env.reset()

    if selected_frame is not None:
        output_path = os.path.join(
            OUTPUT_DIR,
            "cartpole_example_frame.png"
        )

        selected_frame = cv2.cvtColor(
            selected_frame,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(output_path, selected_frame)

        print(f"Saved screenshot to: {output_path}")
    else:
        print("No valid frame captured.")

    env.close()


if __name__ == "__main__":
    main()