import json, os
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector
from scipy.optimize import linear_sum_assignment
from ml.model import DATA_DIR, load_split
from quarkquacks.qubo import build_qubo, bits_to_assignment
from quantum.solver import to_qiskit_angles

def p_optimal_at(cost, gamma, beta, b_scale=1.0, g_scale=1.0):
    cost = np.asarray(cost, dtype=float); n = len(cost)
    qubo = build_qubo(cost)
    op, _ = qubo.to_ising()
    circ = QAOAAnsatz(cost_operator=op, reps=1)
    b, g = to_qiskit_angles(qubo, gamma, beta)            # [beta, gamma/scale]
    probs = Statevector(circ.assign_parameters([b * b_scale, g * g_scale])).probabilities()
    r, c = linear_sum_assignment(cost); opt = cost[r, c].sum()
    total = 0.0
    for idx, p in enumerate(probs):
        a = bits_to_assignment([(idx >> k) & 1 for k in range(n * n)], n)
        if a is not None and abs(sum(cost[bb, o] for bb, o in a.items()) - opt) < 1e-9:
            total += p
    return total

_, test = load_split(0)
by_id = {s["id"]: s for s in json.load(open(os.path.join(DATA_DIR, "scenarios.json")))}
row = test[test["n"] == 2].iloc[1]
cost = by_id[row["id"]]["cost"]
print("dataset p_optimal:", row["p_optimal"])
for bs in (1.0, 0.5, 2.0):
    for gs in (1.0, 0.5, 2.0):
        print(f"beta x{bs}, gamma x{gs}: {p_optimal_at(cost, row['best_gamma'], row['best_beta'], bs, gs):.4f}")