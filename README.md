Project 3: Reinforcement Learning on Gymnasium
Student ID: 414551032

This project investigates reinforcement learning using Proximal Policy Optimization (PPO) on two different environments: Breakout from the Atari suite and CartPole from the classic control suite.

The goal is to understand how task characteristics influence reinforcement learning performance, convergence speed, and training stability. To achieve this, the project conducts learning rate comparison experiments and extended training studies on Breakout, and compares the results with CartPole as a dense-reward control task.

In addition, the project analyzes the effects of reward structure, observation complexity, training duration, and hyperparameter selection on PPO learning behavior across different reinforcement learning environments.

```text
AIHW3/
├── src/
│   ├── 01_train_breakout_smoke_test.py
│   ├── 02_evaluate_breakout.py
│   ├── 03_train_breakout_baseline.py
│   ├── 04_train_breakout_lr_experiments.py
│   ├── 05_extend_breakout_best_lr.py
│   ├── 06_extend_breakout_further.py
│   └── 07_plot_breakout_results.py
│
├── src_CartPole/
│   ├── 001_train_cartpole_baseline.py
│   ├── 002_capture_cartpole_frame.py
│   ├── 003_capture_breakout_frame.py
│   └── 004_plot_cartpole_results.py
│
└── README.md
```


Notes:
1. Learning rate comparison on Atari Breakout
2. Long-term PPO training on Breakout (500k → 2M → 6M timesteps)
3. Comparison between Atari Breakout and CartPole-v1
4. Analysis of reward structure, observation complexity, and convergence behavior
