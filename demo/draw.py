"""Shared drawing helpers: labelled campus map, Canada-goose icons, people."""

from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle

WALL, FLOOR, INK, MUTED = "#4A5868", "#FFFFFF", "#1F2A37", "#5B6775"
COUNTER_FILL, ROOM_FILL = "#FDE7D2", "#E3EEFB"
COUNTER_INK, ROOM_INK = "#B4530B", "#1D4ED8"
GOOSE_COLORS = ["#2F6FD6", "#D9731A", "#0E9F6E", "#7C3AED", "#C0392B"]
PERSON = "#6B7280"
COUNTER_NAMES = {"A": "Coffee", "B": "Pizza", "C": "Bubble\ntea"}


def draw_campus(ax, campus, labels=True):
    """Walls, hallways, food counters and study rooms."""
    grid = campus.grid
    ax.set_facecolor(WALL)
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch != "#":
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=FLOOR, edgecolor="none", zorder=0))
    for sym, (r, c) in campus.named.items():
        is_room = sym.isdigit()
        ax.add_patch(FancyBboxPatch((c - 0.42, r - 0.42), 0.84, 0.84, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=ROOM_FILL if is_room else COUNTER_FILL,
                                    edgecolor=ROOM_INK if is_room else COUNTER_INK, lw=1, zorder=1))
        if labels:
            text = sym if is_room else COUNTER_NAMES.get(sym, sym)
            ax.text(c, r, text, ha="center", va="center", zorder=2, weight="bold",
                    fontsize=8 if is_room else 5.5, color=ROOM_INK if is_room else COUNTER_INK, linespacing=0.9)
    ax.set_xlim(-0.5, len(grid[0]) - 0.5)
    ax.set_ylim(len(grid) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_goose(ax, x, y, color, label=None, waiting=False, zorder=5):
    """A small Canada goose: coloured body (identity), black neck and head, white chinstrap."""
    ax.add_patch(Ellipse((x + 0.06, y + 0.14), 0.8, 0.44, facecolor=color, edgecolor=INK, lw=0.8, zorder=zorder))
    ax.add_patch(Ellipse((x + 0.2, y + 0.22), 0.38, 0.15, facecolor="white", edgecolor="none", alpha=0.35,
                         zorder=zorder + 0.1))                                         # wing highlight
    ax.plot([x - 0.26, x - 0.3, x - 0.27], [y + 0.04, y - 0.12, y - 0.26], color=INK, lw=3.6,
            solid_capstyle="round", zorder=zorder + 0.2)                               # neck
    ax.add_patch(Ellipse((x - 0.33, y - 0.31), 0.24, 0.17, facecolor=INK, edgecolor="none", zorder=zorder + 0.3))
    ax.add_patch(Ellipse((x - 0.3, y - 0.27), 0.08, 0.05, facecolor="white", edgecolor="none",
                         zorder=zorder + 0.4))                                         # white chinstrap
    ax.add_patch(Ellipse((x - 0.48, y - 0.3), 0.14, 0.06, facecolor=INK, edgecolor="none",
                         zorder=zorder + 0.3))                                         # beak
    if label:
        ax.text(x + 0.48, y + 0.44, label, fontsize=6.5, weight="bold", color=color, zorder=zorder + 1,
                ha="left", va="center")
    if waiting:
        ax.text(x + 0.05, y - 0.66, "wait", fontsize=6.5, color=color, weight="bold", ha="center",
                zorder=zorder + 2, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8))


def draw_person(ax, x, y, zorder=4):
    """A simple pedestrian: head and body."""
    ax.add_patch(Circle((x, y - 0.22), 0.11, facecolor=PERSON, edgecolor="none", zorder=zorder))
    ax.add_patch(FancyBboxPatch((x - 0.13, y - 0.07), 0.26, 0.34, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=PERSON, edgecolor="none", zorder=zorder))
