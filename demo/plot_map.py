"""Static campus map for the README and slides.

    python demo/plot_map.py      ->  figures/campus_map.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from draw import GOOSE_COLORS, draw_campus, draw_goose, draw_person  # noqa: E402

from quarkquacks import Campus  # noqa: E402


def main(out="figures/campus_map.png"):
    campus = Campus()
    fig, ax = plt.subplots(figsize=(8, 4.9))
    draw_campus(ax, campus)
    for g, (r, c) in enumerate([(4, 3), (4, 14), (8, 9)]):
        draw_goose(ax, c, r, GOOSE_COLORS[g], label=f"G{g}")
    for r, c in [(2, 11), (7, 4)]:
        draw_person(ax, c, r)
    ax.set_title("Campus map: food counters (orange), study rooms 1–6 (blue), geese and people",
                 loc="left", fontsize=10, color="#1F2A37")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
