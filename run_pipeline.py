"""Run the whole QuarkQuacks pipeline on one scenario, end to end.

Assignment mode (default, one trip per goose):
    campus scenario -> ML-predicted angles -> Qiskit QAOA -> classical check
    -> collision-free delivery simulation -> plan plot (+ optional animation)

Carpool mode (--carpool, a goose may carry two orders):
    campus scenario -> carpool QUBO -> Qiskit QAOA samples -> check / repair /
    fallback -> collision-free delivery simulation -> plan plot (+ animation)

    python run_pipeline.py                         # 4 geese
    python run_pipeline.py --bundle                # 4 geese, nearby orders pre-bundled
    python run_pipeline.py --carpool               # 2 geese share 4 orders
    python run_pipeline.py --gif figures/run.gif   # also save the delivery animation
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
from quarkquacks.carpool import make_carpool_scenario, solve_carpool  # noqa: E402
from quarkquacks.routes import check_frames, simulate  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo"))
from animate import render_gif  # noqa: E402
from draw import GOOSE_COLORS, draw_campus, draw_goose  # noqa: E402


def describe(order):
    rooms = list(dict.fromkeys(order.get("rooms", [order["room"]])))  # same room twice -> once
    what = " + ".join(order.get("items", [order["item"]]))
    return f"{what} to room{'s' if len(rooms) > 1 else ''} {', '.join(rooms)}"


def describe_orders(orders):
    """Text for several separate orders carried by one goose (carpool)."""
    rooms = list(dict.fromkeys(o["room"] for o in orders))
    return f"{' + '.join(o['item'] for o in orders)} to room{'s' if len(rooms) > 1 else ''} {', '.join(rooms)}"


def plot_plan(campus, scenario, trips, path, title):
    """trips: {goose: (stop cells, label)}. Straight-line sketch of each goose's trip."""
    fig, ax = plt.subplots(figsize=(9, 5.8))
    draw_campus(ax, campus)
    handles = []
    for goose, (stops, label) in sorted(trips.items()):
        gr, gc = scenario["bots"][goose]
        color = GOOSE_COLORS[goose % len(GOOSE_COLORS)]
        ax.plot([gc] + [c for _, c in stops], [gr] + [r for r, _ in stops], "--", color=color, lw=2,
                alpha=0.9, zorder=1.5)  # under the counter and room labels
        draw_goose(ax, gc, gr, color, label=f"G{goose}")
        handles.append(plt.Line2D([], [], color=color, lw=2, ls="--", label=f"G{goose}: {label}"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=min(len(handles), 4),
              fontsize=8, frameon=False)
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def deliver(campus, scenario, assignment, args, waypoints=None):
    """Collision-free delivery of the chosen plan; the safety check raises if anything collides."""
    frames, stats = simulate(campus, scenario, assignment, n_people=args.people, seed=args.seed,
                             waypoints=waypoints)
    check_frames(frames)
    print(f"Delivery: {'all trips delivered' if stats['all_delivered'] else 'NOT all delivered'} "
          f"in {stats['ticks']} steps, 0 collisions; waited {stats['waits_for_goose']}x for geese, "
          f"{stats['waits_for_person']}x for people")
    return frames, stats


def run_assignment(args, campus):
    scenario = campus.make_scenario(args.geese, random.Random(args.seed), bundle=args.bundle)
    cost = scenario["cost"]

    def total(assignment):
        return None if assignment is None else sum(cost[g][o] for g, o in assignment.items())

    print(f"Scenario: {args.geese} geese ({args.geese ** 2} qubits), seed {args.seed}"
          f"{', bundled' if args.bundle else ''}")
    for g, order in enumerate(scenario["orders"]):
        print(f"  trip {g}: {describe(order)}")
    if scenario.get("waiting"):
        print(f"  waiting for next round: orders {scenario['waiting']}")

    gamma, beta = predict_angles(scenario)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = solve_qaoa(cost, warm_start=(gamma, beta), seed=args.seed)
        cold = None if args.no_compare else solve_qaoa(cost, seed=args.seed)
    perm, best_cost = solve_classical(cost)

    print(f"\nML predicted angles: gamma={gamma:.3f}, beta={beta:.3f}")
    print(f"QAOA warm start: cost {total(warm['assignment'])}, {warm['evaluations']} optimizer steps")
    if cold:
        print(f"QAOA random:     cost {total(cold['assignment'])}, {cold['evaluations']} optimizer steps")
    print(f"Classical best:  cost {best_cost}")

    plan = warm["assignment"]
    if plan is None:  # safety gate: never send geese out with an invalid plan
        plan = dict(enumerate(perm))
        print("QAOA plan was invalid: using the classical plan instead")
    elif total(plan) > best_cost + 1e-9:
        print(f"QAOA plan is valid but {total(plan) - best_cost:.0f} steps longer than the best")

    frames, _ = deliver(campus, scenario, plan, args)
    trips = {g: ([campus.named[scenario["orders"][o]["counter"]]] +
                 [campus.named[r] for r in scenario["orders"][o].get("rooms", [scenario["orders"][o]["room"]])],
                 describe(scenario["orders"][o])) for g, o in plan.items()}
    return scenario, plan, trips, frames


def run_carpool(args, campus):
    geese = args.geese if args.geese_set else 2
    scenario = make_carpool_scenario(campus, geese, args.orders, random.Random(args.seed))
    print(f"Carpool scenario: {geese} geese, {args.orders} orders, seed {args.seed}")
    for i, o in enumerate(scenario["orders"]):
        print(f"  order {i}: {o['item']} to room {o['room']}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_carpool(campus, scenario, shots=args.shots, seed=args.seed)
    problem, plan = r["problem"], r["plan"]
    print(f"\nCarpool QUBO: {r['n_qubits']} qubits; QAOA {args.shots} samples, "
          f"{r['valid_shot_rate']:.1%} valid")
    print(f"Plan from: {r['source']} (sampled = straight from QAOA, repaired = near-miss fixed, "
          f"fallback = classical)")
    print(f"Total customer wait: {r['cost']:.0f} steps; best possible {r['optimal_cost']:.0f} "
          f"(quality {r['approx_ratio']:.0%})")
    for g, orders in sorted(plan.items()):
        print(f"  G{g}: orders {orders}" if orders else f"  G{g}: idle")

    busy = {g: min(orders) for g, orders in plan.items() if orders}  # first come, first served
    waypoints = problem.waypoints(plan)
    frames, _ = deliver(campus, scenario, busy, args, waypoints)
    trips = {g: (waypoints[g], describe_orders([scenario["orders"][o] for o in sorted(orders)]))
             for g, orders in plan.items() if orders}
    return scenario, busy, trips, frames


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geese", type=int, default=None, help="default 4 (2 with --carpool)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bundle", action="store_true", help="pre-bundle nearby orders into one trip")
    ap.add_argument("--carpool", action="store_true", help="QAOA decides which orders share a goose")
    ap.add_argument("--orders", type=int, default=4, help="orders in carpool mode")
    ap.add_argument("--shots", type=int, default=1000, help="QAOA samples in carpool mode")
    ap.add_argument("--people", type=int, default=3, help="pedestrians in the delivery simulation")
    ap.add_argument("--no-compare", action="store_true", help="skip the random-start QAOA comparison")
    ap.add_argument("--out", default="figures/pipeline_example.png")
    ap.add_argument("--gif", default=None, help="also save the delivery animation here")
    args = ap.parse_args(argv)
    args.geese_set = args.geese is not None
    if args.geese is None:
        args.geese = 4
    if not args.carpool and not 2 <= args.geese <= 4:
        ap.error("--geese must be 2-4 (4 geese = 16 qubits)")

    campus = Campus()
    if args.carpool:
        scenario, assignment, trips, frames = run_carpool(args, campus)
        title = "Carpool plan: goose → counters → rooms (straight-line sketch)"
    else:
        scenario, assignment, trips, frames = run_assignment(args, campus)
        title = "Delivery plan: goose → counter → room (straight-line sketch)"

    plot_plan(campus, scenario, trips, args.out, title)
    print(f"\nPlan saved to {args.out}")
    if args.gif:
        legend = "   ".join(f"G{g}: {label}" for g, (_, label) in sorted(trips.items()))
        render_gif(campus, frames, legend, args.gif, len(trips))
        print(f"Animation saved to {args.gif}")


if __name__ == "__main__":
    main()
