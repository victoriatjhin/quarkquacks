"""Headline results chart: QAOA optimizer steps, random start vs. ML warm start.

Reads data/warmstart_results.csv (written by `python -m ml.warmstart`, 90
held-out test scenarios) and saves figures/warmstart.png.

    python scripts/plot_warmstart.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID = "#1F2A37", "#5B6775", "#E5E9EE"
RANDOM, WARM = "#2F6FD6", "#D9731A"  # validated colorblind-safe pair, same as the scaling chart


def main(path="data/warmstart_results.csv", out="figures/warmstart.png"):
    df = pd.read_csv(path)
    s = df.groupby(["n", "start"]).agg(steps=("steps", "mean"), p=("p_optimal", "mean"),
                                       count=("id", "count")).reset_index()
    sizes = sorted(s["n"].unique())
    rnd = s[s["start"] == "random"].set_index("n")
    warm = s[s["start"] == "warm-start"].set_index("n")
    n_test = int(s.groupby("n")["count"].first().sum())

    fig, ax = plt.subplots(figsize=(8, 4.6))
    w, x = 0.36, range(len(sizes))
    for i, (sub, color, label) in enumerate([(rnd, RANDOM, "random start"), (warm, WARM, "ML warm start")]):
        vals = [sub.loc[n, "steps"] for n in sizes]
        pos = [p + (i - 0.5) * (w + 0.02) for p in x]
        bars = ax.bar(pos, vals, width=w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=10, color=INK)
    for p, n in zip(x, sizes):
        cut = 1 - warm.loc[n, "steps"] / rnd.loc[n, "steps"]
        ax.text(p + (w + 0.02) / 2, warm.loc[n, "steps"] + 22, f"−{cut:.0%}", ha="center",
                fontsize=12, weight="bold", color=WARM)

    cuts = [1 - warm.loc[n, "steps"] / rnd.loc[n, "steps"] for n in sizes]
    ax.set_title(f"ML warm start cuts QAOA optimizer steps by {min(cuts):.0%}–{max(cuts):.0%}",
                 loc="left", fontsize=13, color=INK, pad=24)
    ax.text(0, 1.025, f"Average steps on {n_test} test scenarios the model never saw",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.set_xticks(list(x), [f"{n} geese\n({n * n} qubits)" for n in sizes])
    ax.set_ylabel("optimizer steps (lower is better)", color=MUTED, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(rnd["steps"]) * 1.15)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
