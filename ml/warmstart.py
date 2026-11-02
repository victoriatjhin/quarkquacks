"""Measure how much the warm-start model helps QAOA.

On the held-out test scenarios, optimise p=1 QAOA starting from random angles
vs. the model's predicted angles, and compare optimizer steps and quality.

    python -m ml.model        # train first
    python -m ml.warmstart
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ml.model import DATA_DIR, load_split, predict_angles
from quarkquacks.qaoa_sim import energy, qaoa_p1_probs
from quarkquacks.qubo import assignment_to_index, normalized_cost_vector


def optimise(C, start, tol=1e-2):
    """Run COBYLA on the QAOA energy; return (final energy, steps, gamma, beta)."""
    res = minimize(lambda x: energy(C, x[0], x[1]), start, method="COBYLA",
                   options={"maxiter": 200, "tol": tol})
    return res.fun, res.nfev, res.x[0], res.x[1]


def main(seed=0):
    _, test = load_split(seed)
    with open(os.path.join(DATA_DIR, "scenarios.json")) as f:
        scenarios = {s["id"]: s for s in json.load(f)}

    rng = np.random.default_rng(seed)
    rows = []
    for _, row in test.iterrows():
        scn = scenarios[row["id"]]
        C = normalized_cost_vector(scn["cost"])
        opt_idx = assignment_to_index(scn["optimal_assignment"])
        starts = {"random": rng.uniform([0, 0], [2 * np.pi, np.pi]),
                  "warm-start": np.array(predict_angles(scn))}
        for name, start in starts.items():
            e, steps, g, b = optimise(C, start)
            rows.append({"id": int(row["id"]), "n": int(row["n"]), "start": name, "steps": steps,
                         "start_energy": energy(C, *start), "final_energy": e,
                         "p_optimal": qaoa_p1_probs(C, g, b)[opt_idx]})

    res = pd.DataFrame(rows)
    summary = res.groupby(["n", "start"])[["start_energy", "steps", "final_energy", "p_optimal"]].mean()
    print(f"Test scenarios: {len(test)}\n")
    print(summary.round(4).to_string())
    res.to_csv(os.path.join(DATA_DIR, "warmstart_results.csv"), index=False)
    print(f"\nPer-scenario results saved to {DATA_DIR}/warmstart_results.csv")


if __name__ == "__main__":
    main()
