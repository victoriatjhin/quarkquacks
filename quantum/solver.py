"""Qiskit QAOA solver for the goose-to-order assignment.

Starter version for Person B to extend.

solve_qaoa(cost, warm_start=None) runs p=1 QAOA and returns the decoded
assignment. Pass warm_start=(gamma, beta) from ml.model.predict_angles to
warm-start, or leave it None for a random start.

Angle convention (verified to match quarkquacks/qaoa_sim.py exactly):
- Qiskit orders QAOA parameters as [beta, gamma].
- The ML model's gamma is for costs scaled to max |value| = 1, while Qiskit uses
  the unscaled Ising operator, so Qiskit's gamma = model gamma / scale.
to_qiskit_angles() does both conversions.
"""

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from quarkquacks.qubo import bits_to_assignment, build_qubo, qubo_cost_vector


def to_qiskit_angles(qubo, gamma, beta):
    """Convert model angles (gamma, beta) to Qiskit's QAOA parameter list [beta, gamma / scale]."""
    scale = float(np.abs(qubo_cost_vector(qubo)).max())
    return [beta, gamma / scale]


def solve_qaoa(cost, warm_start=None, maxiter=100, seed=0):
    """Return {"assignment": {goose: order} or None, "objective", "evaluations"}."""
    cost = np.asarray(cost, dtype=float)
    n = len(cost)
    qubo = build_qubo(cost)

    if warm_start is None:
        initial_point = np.random.default_rng(seed).uniform(0, np.pi, 2)
    else:
        initial_point = to_qiskit_angles(qubo, *warm_start)

    evals = []
    # Compile the circuit to basic gates first: same results, about 50x faster to simulate.
    compiler = generate_preset_pass_manager(optimization_level=1, basis_gates=["h", "rx", "rz", "rzz", "cx"])
    qaoa = QAOA(
        sampler=StatevectorSampler(seed=seed),
        optimizer=COBYLA(maxiter=maxiter),
        reps=1,
        initial_point=initial_point,
        callback=lambda count, params, value, meta: evals.append(value),
        transpiler=compiler,
    )
    result = MinimumEigenOptimizer(qaoa).solve(qubo)
    assignment = bits_to_assignment(result.x[: n * n], n)
    real_cost = None if assignment is None else float(
        sum(cost[b, o] for b, o in assignment.items()))

    # the exact best plan, from the classical solver
    rows, cols = linear_sum_assignment(cost)
    optimal = {int(r): int(c) for r, c in zip(rows, cols)}

    # share of all samples that landed on exactly that plan
    p_optimal = sum(s.probability for s in result.samples
                    if bits_to_assignment(s.x[: n * n], n) == optimal)
    return {"assignment": assignment,
            "cost": real_cost,
            "p_optimal": p_optimal,
            "evaluations": len(evals)}


if __name__ == "__main__":
    from scipy.optimize import linear_sum_assignment

    cost = np.array([[2, 9, 7],
                     [8, 1, 6],
                     [5, 9, 3]], dtype=float)

    r, c = linear_sum_assignment(cost)
    print("Exact:", dict(zip(r.tolist(), c.tolist())), "cost", cost[r, c].sum())
    print("QAOA, random start:", solve_qaoa(cost))
    print("QAOA, start (0.5, 0.5):", solve_qaoa(cost, warm_start=(0.5, 0.5)))
