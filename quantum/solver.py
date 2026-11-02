"""Qiskit QAOA solver for the goose-to-order assignment.

Starter version for Person B to extend.


solve_qaoa(cost, initial_point=None) runs QAOA on the Aer simulator and returns
the decoded assignment. Pass initial_point=[gamma, beta] from the ML model to
warm-start, or leave it None for a random start.

Note on angle conventions: Qiskit's QAOA orders parameters as [beta, gamma] and
may scale the cost differently from quarkquacks/qaoa_sim.py. Check that the
predicted angles help before relying on them (see TODO below).
"""

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from quarkquacks.qubo import bits_to_assignment, build_quadratic_program, PENALTY
from qiskit_optimization.converters import QuadraticProgramToQubo


def solve_qaoa(cost, initial_point=None, reps=1, maxiter=100, seed=0):
    """Return dict with assignment {bot: order}, scaled objective and optimizer evaluations."""
    n = len(cost)
    qubo = QuadraticProgramToQubo(penalty=PENALTY).convert(build_quadratic_program(cost))

    evals = []
    rng = np.random.default_rng(seed)
    if initial_point is None:
        initial_point = rng.uniform(0, np.pi, 2 * reps)
    else:
        gamma, beta = initial_point
        initial_point = [beta, gamma]          # Qiskit order: [beta, gamma]

    qaoa = QAOA(
        sampler=StatevectorSampler(seed=seed),
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        initial_point=initial_point,
        callback=lambda count, params, value, meta: evals.append(value),
    )
    result = MinimumEigenOptimizer(qaoa).solve(qubo)
    assignment = bits_to_assignment(result.x[: n * n], n)
    # TODO (B): compare evaluations with random vs. ML-predicted starting angles
    # on the test scenarios, and confirm the angle convention matches qaoa_sim.py.
    return {"assignment": assignment, "objective": float(result.fval), "evaluations": len(evals)}


if __name__ == "__main__":
    import random
    from quarkquacks import Campus, solve_classical

    scn = Campus().make_scenario(2, random.Random(0))
    print("QAOA:     ", solve_qaoa(scn["cost"]))
    print("Classical:", solve_classical(scn["cost"]))
