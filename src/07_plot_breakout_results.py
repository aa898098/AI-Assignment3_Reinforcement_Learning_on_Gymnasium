import os
import json

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = r"D:/YS/assim3/07"
os.makedirs(OUTPUT_DIR, exist_ok=True)


EXPERIMENTS = [
    {
        "name": "LR 1e-4 (500k)",
        "monitor_csv": r"D:/YS/assim3/04/lr_1e-4/logs/monitor.csv",
    },
    {
        "name": "LR 2.5e-4 (500k)",
        "monitor_csv": r"D:/YS/assim3/04/lr_2_5e-4/logs/monitor.csv",
    },
    {
        "name": "LR 1e-3 (500k)",
        "monitor_csv": r"D:/YS/assim3/04/lr_1e-3/logs/monitor.csv",
    },
    {
        "name": "LR 1e-4 Extended (2M)",
        "monitor_csv": r"D:/YS/assim3/05/logs/monitor.csv",
    },
    {
        "name": "LR 1e-4 Further Extended (6M)",
        "monitor_csv": r"D:/YS/assim3/06/logs/monitor.csv",
    },
]


ROLLING_WINDOW = 100
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})


def load_monitor_csv(path):
    df = pd.read_csv(path, comment="#")
    df["episode"] = range(1, len(df) + 1)
    df["rolling_reward"] = df["r"].rolling(
        window=ROLLING_WINDOW,
        min_periods=1,
    ).mean()
    return df


def summarize_experiment(name, df):
    return {
        "name": name,
        "episodes": int(len(df)),
        "mean_reward": float(df["r"].mean()),
        "std_reward": float(df["r"].std()),
        "min_reward": float(df["r"].min()),
        "max_reward": float(df["r"].max()),
        "median_reward": float(df["r"].median()),
        "q75_reward": float(df["r"].quantile(0.75)),
        "final_rolling_reward": float(df["rolling_reward"].iloc[-1]),
        "mean_episode_length": float(df["l"].mean()),
        "max_episode_length": int(df["l"].max()),
    }


def plot_raw_rewards(all_data):
    plt.figure(figsize=(12, 6))

    for item in all_data:
        df = item["df"]
        plt.plot(
            df["episode"],
            df["r"],
            alpha=0.35,
            label=item["name"],
        )

    plt.title("Breakout Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "breakout_raw_rewards.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rolling_rewards(all_data):
    plt.figure(figsize=(12, 6))

    for item in all_data:
        df = item["df"]
        plt.plot(
            df["episode"],
            df["rolling_reward"],
            label=item["name"],
        )

    plt.title(f"Breakout Rolling Mean Reward (Window = {ROLLING_WINDOW})")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Mean Reward")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "breakout_rolling_rewards.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_reward_boxplot(all_data):
    rewards = [item["df"]["r"] for item in all_data]
    labels = [item["name"] for item in all_data]

    plt.figure(figsize=(12, 6))
    plt.boxplot(rewards, labels=labels, showfliers=False)

    plt.title("Breakout Reward Distribution")
    plt.xlabel("Experiment")
    plt.ylabel("Episode Reward")
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "breakout_reward_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_summary_bar_chart(summaries):
    names = [item["name"] for item in summaries]
    values = [item["final_rolling_reward"] for item in summaries]

    plt.figure(figsize=(12, 6))
    plt.bar(names, values)

    plt.title("Breakout Final Rolling Mean Reward")
    plt.xlabel("Experiment")
    plt.ylabel("Final Rolling Mean Reward")
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "breakout_final_rolling_reward_bar.png"
    )
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    all_data = []
    summaries = []

    for exp in EXPERIMENTS:
        df = load_monitor_csv(exp["monitor_csv"])

        all_data.append(
            {
                "name": exp["name"],
                "df": df,
            }
        )

        summaries.append(
            summarize_experiment(exp["name"], df)
        )

    plot_raw_rewards(all_data)
    plot_rolling_rewards(all_data)
    plot_reward_boxplot(all_data)
    plot_summary_bar_chart(summaries)

    summary_df = pd.DataFrame(summaries)

    summary_csv_path = os.path.join(
        OUTPUT_DIR,
        "breakout_summary_table.csv"
    )
    summary_json_path = os.path.join(
        OUTPUT_DIR,
        "breakout_summary_table.json"
    )

    summary_df.to_csv(summary_csv_path, index=False)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4)

    print(summary_df)

    print("Plots saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()