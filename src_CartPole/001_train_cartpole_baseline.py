import os
import json
import time
import platform
from datetime import datetime

import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


OUTPUT_DIR = r"D:/YS/assim3/CartPole/001"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

CONFIG = {
    "script": "001_train_cartpole_baseline.py",
    "environment": "CartPole-v1",
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "total_timesteps": 100_000,
    "checkpoint_freq": 25_000,
    "seed": 414551032,
    "device": "cpu",
}


def make_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_system_info():
    info = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_requested": CONFIG["device"],
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)

    return info


def make_env(seed):
    env = gym.make(CONFIG["environment"])

    monitor_path = os.path.join(LOG_DIR, "monitor.csv")
    env = Monitor(env, filename=monitor_path)

    env.reset(seed=seed)

    return env


def main():
    make_output_dirs()

    save_json(os.path.join(OUTPUT_DIR, "config.json"), CONFIG)
    save_json(os.path.join(OUTPUT_DIR, "system_info.json"), get_system_info())

    env = make_env(CONFIG["seed"])

    checkpoint_callback = CheckpointCallback(
        save_freq=CONFIG["checkpoint_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_cartpole_baseline",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        CONFIG["policy"],
        env,
        verbose=1,
        device=CONFIG["device"],
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
        ent_coef=CONFIG["ent_coef"],
        vf_coef=CONFIG["vf_coef"],
        seed=CONFIG["seed"],
        tensorboard_log=LOG_DIR,
    )

    start_time = time.time()

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=checkpoint_callback,
        tb_log_name="ppo_cartpole_baseline",
    )

    training_seconds = time.time() - start_time

    final_model_path = os.path.join(
        MODEL_DIR,
        "ppo_cartpole_baseline_final"
    )

    model.save(final_model_path)

    summary = {
        "status": "finished",
        "total_timesteps": CONFIG["total_timesteps"],
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60,
        "final_model_path": final_model_path + ".zip",
        "monitor_csv": os.path.join(LOG_DIR, "monitor.csv"),
        "checkpoint_dir": CHECKPOINT_DIR,
    }

    save_json(os.path.join(OUTPUT_DIR, "training_summary.json"), summary)

    env.close()

    print("CartPole baseline training finished successfully.")
    print(f"Training minutes: {training_seconds / 60:.2f}")
    print(f"Final model saved to: {final_model_path}.zip")


if __name__ == "__main__":
    main()