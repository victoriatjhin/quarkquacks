"""Run the whole QuarkQuacks pipeline on one scenario.

campus scenario -> ML-predicted angles -> Qiskit QAOA -> goose assignments
-> check against the classical solver -> plot

    python run_pipeline.py --geese 3 --seed 0
"""

import argparse
import os
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.model import predict_angles  # noqa: E402
from quantum.solver import solve_qaoa  # noqa: E402
from quarkquacks import Campus, solve_classical  # noqa: E402

GOOSE_COLORS = ["#2563eb", "#e8590c", "#16a34a", "#a855f7"]


def plot_plan(campus, scenario, assignment, path):
    """Campus map with each goose's delivery drawn as goose -> counter -> room."""
    grid = campus.grid
    fig, ax = plt.subplots(figsize=(9, 5.5))
    walls = [[1 if ch == "#" else 0 for ch in row] for row in grid]
    ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.6)

    for symbol, (r, c) in campus.named.items():
        label = symbol if symbol.isdigit() else f"{symbol}"
        ax.text(c, r, label, ha="center", va="center", fontsize=9, weight="bold",
                color="#1d4ed8" if symbol.isdigit() else "#b91c1c")

    handles = []
    for goose, order_idx in assignment.items():
        order = scenario["orders"][order_idx]
        gr, gc = scenario["bots"][goose]
        cr, cc = campus.named[order["counter"]]
        rr, rc = campus.named[order["room"]]
        color = GOOSE_COLORS[goose % len(GOOSE_COLORS)]
        ax.plot([gc, cc, rc], [gr, cr, rr], "--", color=color, lw=2, alpha=0.85)
        ax.scatter([gc], [gr], s=180, color="white", edgecolor=color, lw=2.5, zorder=3)
        ax.text(gc, gr, f"G{goose}", ha="center", va="center", fontsize=7, zorder=4)
        handles.append(plt.Line2D([], [], color=color, lw=2, ls="--",
                                  label=f"G{goose}: {order['item']} to room {order['room']}"))

    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=len(handles), fontsize=8, frameon=False)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("Delivery plan: goose → counter → room (straight-line sketch)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geese", type=int, default=3, choices=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/pipeline_example.png")
    args = ap.parse_args()

    campus = Campus()
    scenario = campus.make_scenario(args.geese, random.Random(args.seed))
    cost = scenario["cost"]

    gamma, beta = predict_angles(scenario)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = solve_qaoa(cost, warm_start=(gamma, beta), seed=args.seed)
        cold = solve_qaoa(cost, seed=args.seed)
    perm, best_cost = solve_classical(cost)

    def total(assignment):
        return None if assignment is None else sum(cost[g][o] for g, o in assignment.items())

    print(f"Scenario: {args.geese} geese, seed {args.seed}")
    for g, order in enumerate(scenario["orders"]):
        print(f"  order {g}: {order['item']} to room {order['room']}")
    print(f"\nPredicted angles: gamma={gamma:.3f}, beta={beta:.3f}")
    print(f"QAOA warm start: cost {total(warm['assignment'])}, {warm['evaluations']} optimizer steps")
    print(f"QAOA random:     cost {total(cold['assignment'])}, {cold['evaluations']} optimizer steps")
    print(f"Classical best:  cost {best_cost}")

    if warm["assignment"] is not None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plot_plan(campus, scenario, warm["assignment"], args.out)
        print(f"\nPlan saved to {args.out}")
    else:
        print("\nQAOA returned an invalid assignment; no plot saved.")


if __name__ == "__main__":
    main()
