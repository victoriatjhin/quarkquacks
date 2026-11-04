"""Run the whole QuarkQuacks pipeline on one scenario.

campus scenario -> ML-predicted angles -> Qiskit QAOA -> goose assignments
-> check against the classical solver -> plot

    python run_pipeline.py --geese 3 --seed 0
"""

import argparse
import os
import random
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.model import predict_angles  # noqa: E402
from quantum.solver import solve_qaoa  # noqa: E402
from quarkquacks import Campus, solve_classical  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo"))
from draw import GOOSE_COLORS, draw_campus, draw_goose  # noqa: E402


def describe(order):
    rooms = list(dict.fromkeys(order.get("rooms", [order["room"]])))  # same room twice -> once
    what = " + ".join(order.get("items", [order["item"]]))
    return f"{what} to room{'s' if len(rooms) > 1 else ''} {', '.join(rooms)}"


def plot_plan(campus, scenario, assignment, path):
    """Campus map with each goose's trip: goose -> counter -> room(s), drawn as straight segments."""
    fig, ax = plt.subplots(figsize=(9, 5.8))
    draw_campus(ax, campus)
    handles = []
    for goose, order_idx in sorted(assignment.items()):
        order = scenario["orders"][order_idx]
        gr, gc = scenario["bots"][goose]
        stops = [campus.named[order["counter"]]] + [campus.named[r] for r in order.get("rooms", [order["room"]])]
        color = GOOSE_COLORS[goose % len(GOOSE_COLORS)]
        ax.plot([gc] + [c for _, c in stops], [gr] + [r for r, _ in stops], "--", color=color, lw=2,
                alpha=0.9, zorder=1.5)  # under the counter and room labels
        draw_goose(ax, gc, gr, color, label=f"G{goose}")
        handles.append(plt.Line2D([], [], color=color, lw=2, ls="--", label=f"G{goose}: {describe(order)}"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
              fontsize=8, frameon=False)
    ax.set_title("Delivery plan: goose → counter → room (straight-line sketch)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geese", type=int, default=3, choices=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bundle", action="store_true", help="pre-bundle nearby orders into one trip")
    ap.add_argument("--out", default="figures/pipeline_example.png")
    args = ap.parse_args()

    campus = Campus()
    scenario = campus.make_scenario(args.geese, random.Random(args.seed), bundle=args.bundle)
    cost = scenario["cost"]

    gamma, beta = predict_angles(scenario)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = solve_qaoa(cost, warm_start=(gamma, beta), seed=args.seed)
        cold = solve_qaoa(cost, seed=args.seed)
    perm, best_cost = solve_classical(cost)

    def total(assignment):
        return None if assignment is None else sum(cost[g][o] for g, o in assignment.items())

    print(f"Scenario: {args.geese} geese, seed {args.seed}{', bundled' if args.bundle else ''}")
    for g, order in enumerate(scenario["orders"]):
        print(f"  trip {g}: {describe(order)}")
    if scenario.get("waiting"):
        print(f"  waiting for next round: orders {scenario['waiting']}")
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
