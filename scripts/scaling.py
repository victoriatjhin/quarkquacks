"""Scaling study: how QAOA effort grows from 2 to 4 geese (4, 9, 16 qubits).

For each size, on fresh test scenarios (not used in training):
- QAOA optimizer steps from a random start vs. the ML warm start
- chance of measuring the best plan at the final angles
- wall-clock time: QAOA simulation vs. the exact classical solver

The warm-start model was trained on 2-3 geese, so 4 geese tests how well it
extrapolates to a size it never saw.

    python scripts/scaling.py --per-size 20
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.model import predict_angles  # noqa: E402
from quarkquacks import Campus, normalized_cost_vector, solve_classical  # noqa: E402
from quarkquacks.qaoa_sim import energy, qaoa_p1_probs  # noqa: E402
from quarkquacks.qubo import assignment_to_index  # noqa: E402


def optimise(C, start, tol=1e-2):
    res = minimize(lambda x: energy(C, x[0], x[1]), start, method="COBYLA",
                   options={"maxiter": 300, "tol": tol})
    return res.nfev, res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--per-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10_000)  # far from training seeds
    ap.add_argument("--out", default="data/scaling.csv")
    args = ap.parse_args()

    campus = Campus()
    rng = random.Random(args.seed)
    angle_rng = np.random.default_rng(args.seed)
    rows = []
    for n in args.sizes:
        t_size = time.time()
        for _ in range(args.per_size):
            scn = campus.make_scenario(n, rng)

            t = time.perf_counter()
            perm, _ = solve_classical(scn["cost"])
            classical_s = time.perf_counter() - t

            C = normalized_cost_vector(scn["cost"])
            opt_idx = assignment_to_index(perm)
            starts = {"random": angle_rng.uniform([0, 0], [2 * np.pi, np.pi]),
                      "warm-start": np.array(predict_angles(scn))}
            for name, start in starts.items():
                t = time.perf_counter()
                steps, (g, b) = optimise(C, start)
                qaoa_s = time.perf_counter() - t
                rows.append({"geese": n, "qubits": n * n, "start": name, "steps": steps,
                             "p_optimal": float(qaoa_p1_probs(C, g, b)[opt_idx]),
                             "random_guess": 1 / 2 ** (n * n),
                             "qaoa_sim_s": qaoa_s, "classical_s": classical_s})
        print(f"{n} geese done ({time.time() - t_size:.0f}s)")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["geese", "qubits", "start"]).agg(
        steps=("steps", "mean"), p_optimal=("p_optimal", "mean"),
        random_guess=("random_guess", "first"), qaoa_sim_s=("qaoa_sim_s", "mean"),
        classical_s=("classical_s", "mean"))
    with pd.option_context("display.float_format", "{:.5g}".format):
        print(summary.to_string())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
