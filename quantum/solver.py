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
    n = len(cost)
    qubo = build_qubo(cost)

    if warm_start is None:
        initial_point = np.random.default_rng(seed).uniform(0, np.pi, 2)
    else:
        initial_point = to_qiskit_angles(qubo, *warm_start)

    evals = []
    qaoa = QAOA(
        sampler=StatevectorSampler(seed=seed),
        optimizer=COBYLA(maxiter=maxiter),
        reps=1,
        initial_point=initial_point,
        callback=lambda count, params, value, meta: evals.append(value),
    )
    result = MinimumEigenOptimizer(qaoa).solve(qubo)
    return {"assignment": bits_to_assignment(result.x[: n * n], n),
            "objective": float(result.fval),
            "evaluations": len(evals)}


if __name__ == "__main__":
    import random

    from ml.model import predict_angles
    from quarkquacks import Campus, solve_classical

    scn = Campus().make_scenario(3, random.Random(0))
    print("QAOA, random start:", solve_qaoa(scn["cost"]))
    print("QAOA, warm start:  ", solve_qaoa(scn["cost"], warm_start=predict_angles(scn)))
    print("Classical:         ", solve_classical(scn["cost"]))
