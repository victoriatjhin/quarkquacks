"""Real Qiskit QAOA: ML warm start vs. random start on fresh scenarios.

The 71-91% fewer-steps result comes from the fast NumPy simulation used for
training. This script checks what happens in the actual Qiskit solver:
how often each start returns the best plan, an invalid plan, and how many
optimizer steps it took.

    python scripts/qiskit_eval.py --geese 4 --scenarios 20   ->  data/qiskit_eval.csv
"""

import argparse
import os
import random
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.model import predict_angles  # noqa: E402
from quantum.solver import solve_qaoa  # noqa: E402
from quarkquacks import Campus, solve_classical  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geese", type=int, nargs="+", default=[4])
    ap.add_argument("--scenarios", type=int, default=20)
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--out", default="data/qiskit_eval.csv")
    args = ap.parse_args()

    campus, rows = Campus(), []
    for n in args.geese:
        for k in range(args.scenarios):
            seed = 20_000 + k  # not used in training
            scn = campus.make_scenario(n, random.Random(seed), bundle=args.bundle)
            _, best = solve_classical(scn["cost"])
            for start in ("warm", "random"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = solve_qaoa(scn["cost"], warm_start=predict_angles(scn) if start == "warm" else None, seed=seed)
                plan = r["assignment"]
                total = None if plan is None else sum(scn["cost"][g][o] for g, o in plan.items())
                rows.append({"geese": n, "seed": seed, "start": start, "steps": r["evaluations"],
                             "valid": plan is not None,
                             "optimal": total is not None and abs(total - best) < 1e-9})
            print(f"{n} geese, scenario {k + 1}/{args.scenarios}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["geese", "start"]).agg(best_plan=("optimal", "mean"), valid=("valid", "mean"),
                                                 steps=("steps", "mean"))
    print(summary.round(3).to_string())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
