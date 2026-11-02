"""Animated demo: geese deliver orders on collision-free routes around bystanders.

    python demo/animate.py --geese 3 --seed 7      ->  figures/demo.gif

With no --seed, it picks a scenario where at least one goose waits for another
goose and for a person, so the collision avoidance is visible.
"""

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quarkquacks import Campus, solve_classical  # noqa: E402
from quarkquacks.routes import simulate  # noqa: E402

GOOSE_COLORS = ["#2F6FD6", "#D9731A", "#0E9F6E", "#7C3AED"]
PERSON = "#5B6775"


def run(n, seed):
    campus = Campus()
    scn = campus.make_scenario(n, random.Random(seed))
    perm, _ = solve_classical(scn["cost"])
    assignment = dict(enumerate(perm))
    frames, stats = simulate(campus, scn, assignment, n_people=3, seed=seed)
    return campus, scn, assignment, frames, stats


def pick_seed(n):
    """First scenario where a goose waits for a goose and for a person."""
    for seed in range(500):
        *_, stats = run(n, seed)
        if stats["all_delivered"] and stats["waits_for_goose"] and stats["waits_for_person"]:
            return seed
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geese", type=int, default=3, choices=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="figures/demo.gif")
    args = ap.parse_args()

    seed = pick_seed(args.geese) if args.seed is None else args.seed
    campus, scn, assignment, frames, stats = run(args.geese, seed)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    walls = [[1 if ch == "#" else 0 for ch in row] for row in campus.grid]

    def draw(i):
        ax.clear()
        frame = frames[i]
        ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.6)
        for sym, (r, c) in campus.named.items():
            ax.text(c, r, sym, ha="center", va="center", fontsize=9, weight="bold",
                    color="#1D4ED8" if sym.isdigit() else "#B91C1C", alpha=0.7)
        for (r, c) in frame["people"]:
            ax.scatter(c, r, s=90, marker="s", color=PERSON, zorder=3)
        for g, cell in frame["geese"].items():
            if cell is None:
                continue
            r, c = cell
            color = GOOSE_COLORS[g % len(GOOSE_COLORS)]
            waiting = g in frame["waiting"]
            ax.scatter(c, r, s=260, color="white", edgecolor=color, lw=3, zorder=4)
            ax.text(c, r, f"G{g}", ha="center", va="center", fontsize=7, zorder=5)
            if waiting:
                ax.text(c + 0.45, r - 0.6, "wait", fontsize=7, color=color, weight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8))
        delivered = sum(cell is None for cell in frame["geese"].values())
        ax.set_title(f"t = {frame['t']}   ·   delivered {delivered}/{len(assignment)}   ·   "
                     "■ = person", fontsize=10, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=180)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=6))
    plt.close(fig)
    print(f"Saved {args.out} (seed {seed}, {len(frames)} frames, "
          f"{stats['waits_for_goose']} waits for geese, {stats['waits_for_person']} for people)")


if __name__ == "__main__":
    main()
