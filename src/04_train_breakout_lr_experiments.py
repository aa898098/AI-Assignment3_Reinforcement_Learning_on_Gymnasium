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


BASE_OUTPUT_DIR = r"D:/YS/assim3/04"

BASE_CONFIG = {
    "script": "04_train_breakout_lr_experiments.py",
    "environment": "ALE/Breakout-v5",
    "algorithm": "PPO",
    "policy": "CnnPolicy",
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "total_timesteps": 500_000,
    "checkpoint_freq": 100_000,
    "seed": 414551032,
    "device": "cuda",
}

EXPERIMENTS = [
    {
        "name": "lr_1e-4",
        "learning_rate": 1e-4,
    },
    {
        "name": "lr_2_5e-4",
        "learning_rate": 2.5e-4,
    },
    {
        "name": "lr_1e-3",
        "learning_rate": 1e-3,
    },
]


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
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
        info["torch_arch_list"] = torch.cuda.get_arch_list()

    return info


def make_dirs(exp_dir):
    dirs = {
        "models": os.path.join(exp_dir, "models"),
        "logs": os.path.join(exp_dir, "logs"),
        "checkpoints": os.path.join(exp_dir, "checkpoints"),
    }

    os.makedirs(exp_dir, exist_ok=True)

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs


def make_env(config, log_dir):
    env = gym.make(
        config["environment"],
        render_mode=None,
        frameskip=1,
        repeat_action_probability=0.0,
    )

    env = AtariWrapper(env)

    monitor_path = os.path.join(log_dir, "monitor.csv")
    env = Monitor(env, filename=monitor_path)

    env.reset(seed=config["seed"])

    return env


def run_experiment(exp):
    config = BASE_CONFIG.copy()
    config.update(exp)

    exp_dir = os.path.join(BASE_OUTPUT_DIR, config["name"])
    dirs = make_dirs(exp_dir)

    save_json(os.path.join(exp_dir, "config.json"), config)
    save_json(os.path.join(exp_dir, "system_info.json"), get_system_info())

    env = make_env(config, dirs["logs"])

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=dirs["checkpoints"],
        name_prefix=f"ppo_breakout_{config['name']}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        config["policy"],
        env,
        verbose=1,
        device=config["device"],
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        seed=config["seed"],
        tensorboard_log=dirs["logs"],
    )

    start_time = time.time()

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=checkpoint_callback,
        tb_log_name=f"ppo_breakout_{config['name']}",
    )

    training_seconds = time.time() - start_time

    final_model_path = os.path.join(
        dirs["models"],
        f"ppo_breakout_{config['name']}_final"
    )

    model.save(final_model_path)

    summary = {
        "status": "finished",
        "experiment_name": config["name"],
        "learning_rate": config["learning_rate"],
        "total_timesteps": config["total_timesteps"],
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60,
        "final_model_path": final_model_path + ".zip",
        "monitor_csv": os.path.join(dirs["logs"], "monitor.csv"),
        "checkpoint_dir": dirs["checkpoints"],
    }

    save_json(os.path.join(exp_dir, "training_summary.json"), summary)

    env.close()

    print("=" * 80)
    print(f"Finished experiment: {config['name']}")
    print(f"Training minutes: {training_seconds / 60:.2f}")
    print(f"Final model saved to: {final_model_path}.zip")
    print("=" * 80)


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    all_start_time = time.time()

    for exp in EXPERIMENTS:
        run_experiment(exp)

    total_seconds = time.time() - all_start_time

    overall_summary = {
        "status": "finished",
        "num_experiments": len(EXPERIMENTS),
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "experiments": [exp["name"] for exp in EXPERIMENTS],
    }

    save_json(
        os.path.join(BASE_OUTPUT_DIR, "overall_summary.json"),
        overall_summary
    )

    print("All experiments finished.")
    print(f"Total minutes: {total_seconds / 60:.2f}")


if __name__ == "__main__":
    main()