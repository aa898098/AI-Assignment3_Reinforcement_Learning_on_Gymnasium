import os
import json
import time
import platform
from datetime import datetime

import ale_py
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


gym.register_envs(ale_py)


OUTPUT_DIR = r"D:/YS/assim3/06"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

PRETRAINED_MODEL_PATH = r"D:/YS/assim3/05/models/ppo_breakout_lr_1e-4_extended_final.zip"

CONFIG = {
    "script": "06_extend_breakout_further.py",
    "environment": "ALE/Breakout-v5",
    "algorithm": "PPO",
    "policy": "CnnPolicy",
    "source_model": PRETRAINED_MODEL_PATH,
    "source_training_timesteps": 2_000_000,
    "additional_timesteps": 4_000_000,
    "expected_total_timesteps": 6_000_000,
    "learning_rate": 1e-4,
    "checkpoint_freq": 500_000,
    "seed": 414551032,
    "device": "cuda",
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
        info["torch_arch_list"] = torch.cuda.get_arch_list()

    return info


def make_env(seed):
    env = gym.make(
        CONFIG["environment"],
        render_mode=None,
        frameskip=1,
        repeat_action_probability=0.0,
    )

    env = AtariWrapper(env)

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
        name_prefix="ppo_breakout_lr_1e-4_further",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO.load(
        PRETRAINED_MODEL_PATH,
        env=env,
        device=CONFIG["device"],
    )

    start_time = time.time()

    model.learn(
        total_timesteps=CONFIG["additional_timesteps"],
        callback=checkpoint_callback,
        tb_log_name="ppo_breakout_lr_1e-4_further",
        reset_num_timesteps=False,
    )

    training_seconds = time.time() - start_time

    final_model_path = os.path.join(
        MODEL_DIR,
        "ppo_breakout_lr_1e-4_further_final"
    )

    model.save(final_model_path)

    summary = {
        "status": "finished",
        "source_model": PRETRAINED_MODEL_PATH,
        "source_training_timesteps": CONFIG["source_training_timesteps"],
        "additional_timesteps": CONFIG["additional_timesteps"],
        "expected_total_timesteps": CONFIG["expected_total_timesteps"],
        "learning_rate": CONFIG["learning_rate"],
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60,
        "final_model_path": final_model_path + ".zip",
        "monitor_csv": os.path.join(LOG_DIR, "monitor.csv"),
        "checkpoint_dir": CHECKPOINT_DIR,
    }

    save_json(os.path.join(OUTPUT_DIR, "training_summary.json"), summary)

    env.close()

    print("Further extended training finished successfully.")
    print(f"Training minutes: {training_seconds / 60:.2f}")
    print(f"Final model saved to: {final_model_path}.zip")


if __name__ == "__main__":
    main()