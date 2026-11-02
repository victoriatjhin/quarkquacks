"""Plot the scaling study saved by scripts/scaling.py.

    python scripts/plot_scaling.py   ->  figures/scaling.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID = "#1F2A37", "#5B6775", "#E5E9EE"
RANDOM, WARM = "#2F6FD6", "#D9731A"          # validated pair (colorblind-safe)
SIM, CLASSICAL = "#7C3AED", "#0E9F6E"        # validated pair for the timing panel


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main(path="data/scaling.csv", out="figures/scaling.png"):
    df = pd.read_csv(path)
    s = df.groupby(["geese", "qubits", "start"]).mean(numeric_only=True).reset_index()
    sizes = sorted(s["geese"].unique())
    labels = [f"{g} geese\n({g * g} qubits)" for g in sizes]
    x = range(len(sizes))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # 1. optimizer steps
    ax = axes[0]
    style(ax, "QAOA optimizer steps (lower is better)")
    for i, (name, color) in enumerate([("random", RANDOM), ("warm-start", WARM)]):
        vals = s[s["start"] == name]["steps"].to_numpy()
        pos = [p + (i - 0.5) * (w + 0.02) for p in x]
        bars = ax.bar(pos, vals, width=w, color=color, label="warm start" if name == "warm-start" else "random start")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center",
                    fontsize=9, color=INK)
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    # 2. chance of best plan vs random guessing
    ax = axes[1]
    style(ax, "Chance of the best plan, × random guessing")
    for i, (name, color) in enumerate([("random", RANDOM), ("warm-start", WARM)]):
        sub = s[s["start"] == name]
        vals = (sub["p_optimal"] / sub["random_guess"]).to_numpy()
        pos = [p + (i - 0.5) * (w + 0.02) for p in x]
        bars = ax.bar(pos, vals, width=w, color=color, label="warm start" if name == "warm-start" else "random start")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.0f}×", ha="center",
                    fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1, (s["p_optimal"] / s["random_guess"]).max() * 4)
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    # 3. time
    ax = axes[2]
    style(ax, "Time per solve (seconds, log scale)")
    sim = s.groupby("geese")["qaoa_sim_s"].mean().to_numpy()
    classical = s.groupby("geese")["classical_s"].mean().to_numpy()
    ax.plot(list(x), sim, "-o", color=SIM, lw=2, ms=8, label="QAOA, simulated on a laptop")
    ax.plot(list(x), classical, "-o", color=CLASSICAL, lw=2, ms=8, label="Exact classical solver")
    ax.set_yscale("log")
    ax.set_xticks(list(x), labels)
    ax.set_xlim(-0.3, len(sizes) - 0.7)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
