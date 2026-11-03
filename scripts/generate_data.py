"""Generate training data for the QAOA warm-start model.

For each random scenario: build the QUBO, find the true best assignment, and
grid-search the best p=1 QAOA angles (gamma, beta) as labels.

    python scripts/generate_data.py --n 300 --seed 42
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quarkquacks import Campus, normalized_cost_vector, solve_classical  # noqa: E402
from quarkquacks.qaoa_sim import grid_search, qaoa_p1_probs, refine  # noqa: E402
from quarkquacks.qubo import assignment_to_index  # noqa: E402
from quarkquacks.features import scenario_features as features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    ap.add_argument("--append", action="store_true",
                    help="add to the existing dataset instead of replacing it")
    ap.add_argument("--refine-existing", action="store_true",
                    help="re-label the existing dataset with refined angles, no new scenarios")
    args = ap.parse_args()
    if args.refine_existing:
        return refine_existing(args.out)

    rng = random.Random(args.seed)
    campus = Campus()
    os.makedirs(args.out, exist_ok=True)
    rows, scenarios, t0 = [], [], time.time()
    csv_path = os.path.join(args.out, "dataset.csv")
    json_path = os.path.join(args.out, "scenarios.json")
    if args.append and os.path.exists(json_path):
        with open(json_path) as f:
            scenarios = json.load(f)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    first_id = len(scenarios)

    for i in range(first_id, first_id + args.n):
        scn = campus.make_scenario(rng.choice(args.sizes), rng)
        C = normalized_cost_vector(scn["cost"])
        perm, opt_cost = solve_classical(scn["cost"])
        e, gamma, beta = refine(C, *grid_search(C, args.grid)[1:])
        p_opt = float(qaoa_p1_probs(C, gamma, beta)[assignment_to_index(perm)])

        scn.update({"id": i, "optimal_assignment": perm, "optimal_cost": opt_cost,
                    "best_gamma": gamma, "best_beta": beta})
        scenarios.append(scn)
        rows.append({"id": i, **features(scn), "best_gamma": round(gamma, 4),
                     "best_beta": round(beta, 4), "best_energy": round(e, 5),
                     "p_optimal": round(p_opt, 5), "optimal_cost": opt_cost})
        done = i + 1 - first_id
        if done % 25 == 0 or done == args.n:
            print(f"{done}/{args.n} scenarios ({time.time() - t0:.0f}s)")

    with open(json_path, "w") as f:
        json.dump(scenarios, f, indent=1)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out}/")


def refine_existing(out):
    """Polish the angle labels of an existing dataset in place."""
    csv_path, json_path = os.path.join(out, "dataset.csv"), os.path.join(out, "scenarios.json")
    with open(json_path) as f:
        scenarios = json.load(f)
    rows, t0 = [], time.time()
    for k, scn in enumerate(scenarios):
        C = normalized_cost_vector(scn["cost"])
        e, gamma, beta = refine(C, scn["best_gamma"], scn["best_beta"])
        p_opt = float(qaoa_p1_probs(C, gamma, beta)[assignment_to_index(scn["optimal_assignment"])])
        scn.update({"best_gamma": gamma, "best_beta": beta})
        rows.append({"id": scn["id"], **features(scn), "best_gamma": round(gamma, 4),
                     "best_beta": round(beta, 4), "best_energy": round(e, 5),
                     "p_optimal": round(p_opt, 5), "optimal_cost": scn["optimal_cost"]})
        if (k + 1) % 50 == 0 or k + 1 == len(scenarios):
            print(f"refined {k + 1}/{len(scenarios)} ({time.time() - t0:.0f}s)", flush=True)
    with open(json_path, "w") as f:
        json.dump(scenarios, f, indent=1)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
