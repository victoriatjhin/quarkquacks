"""Random vs warm-start QAOA on the test scenarios.
Run from the repo root:   python -m quantum.benchmark
"""
import csv, json, os, time
import numpy as np
from scipy.optimize import linear_sum_assignment

from ml.model import DATA_DIR, load_split, predict_angles
from quantum.solver import solve_qaoa


def is_degenerate(cost):
    c = np.asarray(cost, dtype=float)
    return bool(np.allclose(c - c[:, [0]] - c[[0], :] + c[0, 0], 0))


def load_test_scenarios(seed=0):
    _, test = load_split(seed)                      # same fixed split C uses
    with open(os.path.join(DATA_DIR, "scenarios.json")) as f:
        by_id = {s["id"]: s for s in json.load(f)}
    return [by_id[i] for i in test["id"]]


def compare(scenarios, out_csv="benchmark_results.csv", max_n=3, limit=None, seed=0):
    train, _ = load_split(seed)
    cols = ["best_gamma", "best_beta"]
    lo, hi = train[cols].min().to_numpy(), train[cols].max().to_numpy()

    scenarios = [s for s in scenarios if len(s["cost"]) <= max_n][:limit]
    n_deg = sum(is_degenerate(s["cost"]) for s in scenarios)
    print(f"{len(scenarios)} scenarios, {n_deg} degenerate (all plans tie)")

    rows = []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "n", "degenerate", "mode",
                                          "found_optimal", "p_optimal", "chance",
                                          "evaluations", "seconds"])
        w.writeheader()
        for i, scn in enumerate(scenarios):
            cost = np.asarray(scn["cost"], dtype=float)
            n = len(cost)
            r, c = linear_sum_assignment(cost)
            opt_cost = float(cost[r, c].sum())
            starts = {"random": tuple(np.random.default_rng(i).uniform(lo, hi)),
                      "warm": predict_angles(scn)}
            for mode, start in starts.items():
                t0 = time.time()
                out = solve_qaoa(cost, warm_start=start, seed=i)
                row = {"scenario": scn["id"], "n": n, "degenerate": is_degenerate(cost),
                       "mode": mode,
                       "found_optimal": out["cost"] is not None
                                        and abs(out["cost"] - opt_cost) < 1e-9,
                       "p_optimal": out["p_optimal"], "chance": out.get("chance"),
                       "evaluations": out["evaluations"],
                       "seconds": round(time.time() - t0, 2)}
                w.writerow(row); f.flush(); rows.append(row)
            print(f"scenario {scn['id']} done (n={n})")

    for label, keep in (("all", lambda r: True), ("non-degenerate", lambda r: not r["degenerate"])):
        print(f"--- {label} ---")
        for n in sorted({r["n"] for r in rows}):
            for mode in ("random", "warm"):
                sel = [r for r in rows if r["n"] == n and r["mode"] == mode and keep(r)]
                if sel:
                    print(f"n={n} {mode:6s} runs={len(sel)} "
                          f"success={np.mean([r['found_optimal'] for r in sel]):.2f} "
                          f"mean_p_optimal={np.mean([r['p_optimal'] for r in sel]):.4f} "
                          f"mean_evals={np.mean([r['evaluations'] for r in sel]):.1f}")
    return rows


if __name__ == "__main__":
    compare(load_test_scenarios(), limit=3)     # first run: 3 scenarios, to time it