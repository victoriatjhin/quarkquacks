"""Animated demo: geese deliver orders on collision-free routes around bystanders.

    python demo/animate.py                           ->  figures/demo.gif (4 geese)
    python demo/animate.py --bundle --out figures/demo_bundle.gif

With no --seed, it picks a scenario where at least one goose waits for another
goose and for a person (and, with --bundle, one goose carries two orders), so
the collision avoidance and bundling are visible.
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
sys.path.insert(0, os.path.dirname(__file__))
from draw import GOOSE_COLORS, INK, MUTED, draw_campus, draw_goose, draw_person  # noqa: E402

from quarkquacks import Campus, solve_classical  # noqa: E402
from quarkquacks.routes import assignment_drops, simulate  # noqa: E402


def run(n, seed, bundle=False):
    campus = Campus()
    scn = campus.make_scenario(n, random.Random(seed), bundle=bundle, priority=True)
    perm, _ = solve_classical(scn["cost"])
    assignment = dict(enumerate(perm))
    frames, stats = simulate(campus, scn, assignment, n_people=3, seed=seed)
    return campus, scn, assignment, frames, stats


def pick_seed(n, bundle=False):
    """First scenario that shows waiting for a goose and a person (and a bundle)."""
    for seed in range(500):
        _, scn, _, frames, stats = run(n, seed, bundle)
        has_bundle = any(len(set(o.get("rooms", []))) == 2 for o in scn["orders"])  # two different rooms
        if (stats["all_delivered"] and stats["waits_for_goose"] and stats["waits_for_person"]
                and len(frames) <= 70  # short enough to watch
                and (has_bundle or not bundle)):
            return seed
    return 0


def describe(order):
    rooms = list(dict.fromkeys(order.get("rooms", [order["room"]])))  # same room twice -> once
    what = " + ".join(order.get("items", [order["item"]]))
    return f"{what} → room{'s' if len(rooms) > 1 else ''} {', '.join(rooms)}"


def wait_line(frame, drops, delivered_at):
    """'Wait per order:  #0 18 ✓   #1 23 …' for one frame (live counter)."""
    parts = []
    for d in drops:
        key = (d["goose"], d["stop"])
        if key in delivered_at and delivered_at[key] <= frame["t"]:
            parts.append(f"#{d['order']} {delivered_at[key]} ✓")
        else:
            parts.append(f"#{d['order']} {frame['t']} …")
    return "Wait per order (steps):   " + "    ".join(parts)


def render_gif(campus, frames, legend, out, n_trips, drops=None):
    """Save an animation of simulated frames (from quarkquacks.routes.simulate).

    drops (from quarkquacks.routes.assignment_drops) adds a live wait counter per order.
    """
    fig, ax = plt.subplots(figsize=(8, 5.8 if drops else 5.6))
    delivered_at = {}
    for f in frames:  # first frame at which each drop-off stop was reached
        for g, reached in f.get("stops", {}).items():
            for stop in range(reached):
                delivered_at.setdefault((g, stop), f["t"])

    def draw(i):
        ax.clear()
        frame = frames[i]
        draw_campus(ax, campus)
        for (r, c) in frame["people"]:
            draw_person(ax, c, r)
        for g, cell in frame["geese"].items():
            if cell is not None:
                r, c = cell
                draw_goose(ax, c, r, GOOSE_COLORS[g % len(GOOSE_COLORS)], label=f"G{g}",
                           waiting=g in frame["waiting"])
        delivered = sum(cell is None for cell in frame["geese"].values())
        ax.set_title(f"t = {frame['t']}  ·  trips delivered {delivered}/{n_trips}",
                     fontsize=10, loc="left", color=INK)
        ax.text(0, -0.04, legend, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
        if drops:
            ax.text(0, -0.10, wait_line(frame, drops, delivered_at), transform=ax.transAxes, fontsize=8,
                    color=INK, va="top", weight="bold")
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=180)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=6))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geese", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bundle", action="store_true", help="pre-bundle nearby orders into one trip")
    ap.add_argument("--out", default="figures/demo.gif")
    args = ap.parse_args()

    seed = pick_seed(args.geese, args.bundle) if args.seed is None else args.seed
    campus, scn, assignment, frames, stats = run(args.geese, seed, args.bundle)
    legend = "   ".join(f"G{g}: {describe(scn['orders'][o])}" for g, o in sorted(assignment.items()))
    render_gif(campus, frames, legend, args.out, len(assignment), assignment_drops(scn, assignment))
    print(f"Saved {args.out} (seed {seed}, {len(frames)} frames, "
          f"{stats['waits_for_goose']} waits for geese, {stats['waits_for_person']} for people)")


if __name__ == "__main__":
    main()
