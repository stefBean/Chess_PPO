from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("tensorboard_scalars_export_full.csv")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

# Keep top-level runs only, not TensorBoard nested subruns
df = df[~df["run"].str.contains("/", regex=False)].copy()

thesis_runs = [
    "DopamineSelf_2026-05-14_09-32-26",
    "DopamineSelf_2026-05-14_14-06-23",
    "DopamineVsPPO_2026-05-14_09-32-42",
    "DopamineVsPPO_2026-05-14_14-05-26",
    "DopamineVsPPO_2026-05-15_15-16-15",
]

df = df[df["run"].isin(thesis_runs)].copy()

def condition_from_run(run: str) -> str:
    if run.startswith("DopamineSelf"):
        return "Dopamine self-play"
    if run.startswith("DopamineVsPPO"):
        return "Dopamine vs. PPO"
    return "Other"

df["condition"] = df["run"].apply(condition_from_run)

def plot_tag(tag: str, ylabel: str, filename: str, rolling: int = 5):
    sub = df[df["tag"] == tag].copy()

    if sub.empty:
        print(f"Missing tag: {tag}")
        return

    sub = sub.sort_values(["condition", "run", "step"])
    sub["smooth"] = (
        sub.groupby("run")["value"]
        .transform(lambda s: s.rolling(rolling, min_periods=1).mean())
    )

    grouped = (
        sub.groupby(["condition", "step"])["smooth"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(6.8, 3.8))

    for condition, g in grouped.groupby("condition"):
        plt.plot(g["step"], g["mean"], label=condition)

        if g["std"].notna().any():
            plt.fill_between(
                g["step"],
                g["mean"] - g["std"],
                g["mean"] + g["std"],
                alpha=0.18,
            )

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()

plot_tag(
    "Dopamine/td_error_abs_mean",
    "Mean absolute TD error",
    "fig_td_error_abs_mean.pdf",
)

plot_tag(
    "Dopamine/td_error_std",
    "TD error standard deviation",
    "fig_td_error_std.pdf",
)

plot_tag(
    "Dopamine/td_error_mean",
    "Mean TD error",
    "fig_td_error_mean.pdf",
)

plot_tag(
    "Dopamine/td_error_positive_rate",
    "Positive TD-error rate",
    "fig_td_error_positive_rate.pdf",
)

plot_tag(
    "Dopamine/mood_scale_mean",
    "Mean mood scale",
    "fig_mood_scale_mean.pdf",
)

plot_tag(
    "Dopamine/mood_optimistic_rate",
    "Optimistic mood rate",
    "fig_mood_optimistic_rate.pdf",
)

plot_tag(
    "Dopamine/mood_pessimistic_rate",
    "Pessimistic mood rate",
    "fig_mood_pessimistic_rate.pdf",
)

plot_tag(
    "Dopamine/actor_signal_mean",
    "Mean actor signal",
    "fig_actor_signal_mean.pdf",
)

plot_tag(
    "Dopamine/actor_signal_std",
    "Actor signal standard deviation",
    "fig_actor_signal_std.pdf",
)

print("Figures saved to:", OUT_DIR.resolve())