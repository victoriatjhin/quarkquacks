"""Evaluate carpool mode: QAOA sampling vs. random guessing, same number of samples.

For each sample budget, on 30 scenarios (2 geese, 4 orders, 12 qubits):
- how often the best plan appears directly in the samples
- how often repair or the classical fallback was needed
- share of samples that were valid plans

    python scripts/carpool_eval.py     ->  data/carpool_eval.csv, figures/carpool.png
"""

import argparse
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quarkquacks import Campus  # noqa: E402
from quarkquacks.carpool import make_carpool_scenario, solve_carpool  # noqa: E402


def plot(df, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1F2A37", "#5B6775", "#E5E9EE"
    colors = {"qaoa": "#D9731A", "random": "#2F6FD6"}  # validated colorblind-safe pair
    names = {"qaoa": "QAOA samples", "random": "random guessing"}
    s = df.groupby(["shots", "sampler"]).agg(direct=("direct_best", "mean"),
                                             valid=("valid_shot_rate", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title, fmt in [
            (axes[0], "direct", "Best plan found directly in the samples", "{:.0%}"),
            (axes[1], "valid", "Share of samples that are valid plans", "{:.1%}")]:
        for sampler in ("qaoa", "random"):
            sub = s[s["sampler"] == sampler]
            ax.plot(sub["shots"], sub[col], "-o", lw=2, ms=7, color=colors[sampler], label=names[sampler])
            last = sub.iloc[-1]
            ax.annotate(fmt.format(last[col]), (last["shots"], last[col]), textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=9, color=ink)
        ax.set_xscale("log")
        ax.set_xticks(sorted(s["shots"].unique()), [str(v) for v in sorted(s["shots"].unique())])
        ax.set_xlabel("samples per solve (log scale)", color=muted, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0 if col == "direct" else 1))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.legend(frameon=False, fontsize=9, labelcolor=ink, loc="lower right")
    axes[0].set_ylim(0, 1.08)
    axes[1].set_ylim(0, None)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--shots", type=int, nargs="+", default=[20, 50, 200, 1000])
    ap.add_argument("--geese", type=int, default=2)
    ap.add_argument("--orders", type=int, default=4)
    ap.add_argument("--out", default="data/carpool_eval.csv")
    args = ap.parse_args()

    campus, rows = Campus(), []
    for shots in args.shots:
        for sampler in ("qaoa", "random"):
            for seed in range(args.scenarios):
                scenario = make_carpool_scenario(campus, args.geese, args.orders, random.Random(seed))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = solve_carpool(campus, scenario, shots=shots, seed=seed, sampler=sampler)
                rows.append({"shots": shots, "sampler": sampler, "seed": seed, "source": r["source"],
                             "direct_best": r["source"] == "sampled" and r["approx_ratio"] > 1 - 1e-9,
                             "approx_ratio": r["approx_ratio"], "valid_shot_rate": r["valid_shot_rate"],
                             "n_qubits": r["n_qubits"]})
            sub = pd.DataFrame([x for x in rows if x["shots"] == shots and x["sampler"] == sampler])
            print(f"{shots:>5} samples, {sampler:6s}: best found directly {sub.direct_best.mean():.0%}, "
                  f"repaired {(sub.source == 'repaired').mean():.0%}, fallback {(sub.source == 'fallback').mean():.0%}, "
                  f"valid samples {sub.valid_shot_rate.mean():.1%}, quality {sub.approx_ratio.mean():.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    plot(df, "figures/carpool.png")


if __name__ == "__main__":
    main()
