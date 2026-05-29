import os

import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = r"D:/YS/assim3/CartPole/001/logs/monitor.csv"
OUTPUT_DIR = r"D:/YS/assim3/CartPole/004"

ROLLING_WINDOW = 20

BLUE_LINE = "#1f77b4"
BLUE_BAR = "#4C9FD8"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, comment="#")
    df["episode"] = range(1, len(df) + 1)
    df["rolling_reward"] = df["r"].rolling(
        window=ROLLING_WINDOW,
        min_periods=1
    ).mean()

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    # Reward curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(
        df["episode"],
        df["rolling_reward"],
        color=BLUE_LINE,
        linewidth=2,
        label=f"Rolling Mean Reward (window={ROLLING_WINDOW})"
    )

    plt.axhline(
        y=500,
        color=BLUE_BAR,
        linestyle="--",
        linewidth=1.5,
        label="Maximum Reward"
    )

    plt.title("CartPole PPO Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()

    curve_path = os.path.join(
        OUTPUT_DIR,
        "cartpole_rolling_reward_curve.png"
    )
    plt.savefig(curve_path, dpi=300)
    plt.close()

    # Summary bar chart
    summary = {
        "Mean Reward": df["r"].mean(),
        "Max Reward": df["r"].max(),
        "Final Rolling Reward": df["rolling_reward"].iloc[-1],
    }

    plt.figure(figsize=(7, 4.5))
    plt.bar(
        list(summary.keys()),
        list(summary.values()),
        color=BLUE_BAR
    )

    plt.title("CartPole Reward Summary")
    plt.ylabel("Reward")
    plt.tight_layout()

    bar_path = os.path.join(
        OUTPUT_DIR,
        "cartpole_reward_summary_bar.png"
    )
    plt.savefig(bar_path, dpi=300)
    plt.close()

    summary_path = os.path.join(
        OUTPUT_DIR,
        "cartpole_summary_table.csv"
    )

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"Saved: {curve_path}")
    print(f"Saved: {bar_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()