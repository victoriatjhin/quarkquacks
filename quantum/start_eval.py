"""Quality of STARTING angles: random vs ML-predicted, no optimizer.
Run from the repo root:   python -m quantum.start_eval
"""
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector
from scipy.optimize import linear_sum_assignment

from ml.model import load_split, predict_angles
from quantum.quantum_benchmark import load_test_scenarios, is_degenerate
from quantum.solver import to_qiskit_angles
from quarkquacks.qubo import build_qubo, bits_to_assignment


def make_evaluator(cost):
    cost = np.asarray(cost, dtype=float); n = len(cost)
    qubo = build_qubo(cost)
    op, _ = qubo.to_ising()
    circ = QAOAAnsatz(cost_operator=op, reps=1)
    r, c = linear_sum_assignment(cost); opt = cost[r, c].sum()
    good = []
    for idx in range(2 ** (n * n)):
        a = bits_to_assignment([(idx >> k) & 1 for k in range(n * n)], n)
        if a is not None and abs(sum(cost[b, o] for b, o in a.items()) - opt) < 1e-9:
            good.append(idx)

    def p_at(gamma, beta):
        angles = to_qiskit_angles(qubo, gamma, beta)
        probs = Statevector(circ.assign_parameters(angles)).probabilities()
        return float(probs[good].sum())
    return p_at, len(good) / 2 ** (n * n)


def main(max_n=3, seed=0, n_random=5):
    train, _ = load_split(seed)
    cols = ["best_gamma", "best_beta"]
    lo, hi = train[cols].min().to_numpy(), train[cols].max().to_numpy()
    rows = []
    for i, scn in enumerate(load_test_scenarios(seed)):
        n = len(scn["cost"])
        if n > max_n:
            continue
        p_at, chance = make_evaluator(scn["cost"])
        rng = np.random.default_rng(i)
        p_rand = float(np.mean([p_at(*rng.uniform(lo, hi)) for _ in range(n_random)]))
        p_warm = p_at(*predict_angles(scn))
        rows.append((n, is_degenerate(scn["cost"]), chance, p_rand, p_warm))
    for n in sorted({r[0] for r in rows}):
        sel = [r for r in rows if r[0] == n and not r[1]]
        if sel:
            print(f"n={n} non-degenerate scenarios={len(sel)} "
                  f"chance={np.mean([r[2] for r in sel]):.4f} "
                  f"random start={np.mean([r[3] for r in sel]):.4f} "
                  f"predicted start={np.mean([r[4] for r in sel]):.4f} "
                  f"warm beats random in {np.mean([r[4] > r[3] for r in sel]):.0%}")


if __name__ == "__main__":
    main()