import json, os
from ml.model import DATA_DIR, load_split
from quantum.solver import solve_qaoa

_, test = load_split(0)
with open(os.path.join(DATA_DIR, "scenarios.json")) as f:
    by_id = {s["id"]: s for s in json.load(f)}

row = test[test["n"] == 2].iloc[1]                 # a 2-bot test scenario
scn = by_id[row["id"]]
out = solve_qaoa(scn["cost"], warm_start=(row["best_gamma"], row["best_beta"]), maxiter=20)
print("our p_optimal at the dataset's best angles:", out["p_optimal"])
print("dataset p_optimal:", row["p_optimal"], "| chance:", out.get("chance"))