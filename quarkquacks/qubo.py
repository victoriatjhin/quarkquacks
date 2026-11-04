"""Turn the bot-to-order assignment into a QUBO: the bridge to quantum.

Everyone (especially the quantum solver and the ML model) must use build_qubo
from here, so costs are scaled the same way and predicted angles transfer.
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Penalty for breaking a rule (an order delivered twice, a bot left idle, ...).
# Must be larger than any single scaled cost (max 1.0) but not so large that it
# drowns out the delivery costs. The library default is huge and makes QAOA
# behave almost like random guessing.
PENALTY = 1.5


def build_quadratic_program(cost):
    """x_b_o = 1 if bot b delivers order o. Costs scaled to [0, 1]."""
    cost = np.asarray(cost, dtype=float)
    cost = cost / cost.max()
    n = len(cost)
    qp = QuadraticProgram("campus_delivery")
    for b in range(n):
        for o in range(n):
            qp.binary_var(f"x_{b}_{o}")
    qp.minimize(linear={f"x_{b}_{o}": float(cost[b, o]) for b in range(n) for o in range(n)})
    for o in range(n):
        qp.linear_constraint({f"x_{b}_{o}": 1 for b in range(n)}, "==", 1, f"order_{o}")
    for b in range(n):
        qp.linear_constraint({f"x_{b}_{o}": 1 for o in range(n)}, "==", 1, f"bot_{b}")
    return qp


def build_qubo(cost, penalty=PENALTY):
    return QuadraticProgramToQubo(penalty=penalty).convert(build_quadratic_program(cost))


def qubo_cost_vector(qubo):
    """QUBO value of every bitstring. Bit k of the index = variable k = qubit k."""
    nq = qubo.get_num_vars()
    obj = qubo.objective
    linear = obj.linear.to_array()
    quad = obj.quadratic.to_array()
    out = np.empty(2 ** nq)
    chunk = 1 << 16  # keeps memory small at 16+ qubits
    for start in range(0, 2 ** nq, chunk):
        idx = np.arange(start, min(start + chunk, 2 ** nq))
        x = ((idx[:, None] >> np.arange(nq)) & 1).astype(float)
        out[idx] = obj.constant + x @ linear + np.einsum("bi,ij,bj->b", x, quad, x)
    return out


def normalized_cost_vector(cost, penalty=PENALTY):
    """Cost vector scaled to max |value| = 1, the form the QAOA angles are tuned on."""
    c = qubo_cost_vector(build_qubo(cost, penalty))
    return c / np.abs(c).max()


def bits_to_assignment(bits, n):
    """Decode x_b_o bits (bit index b*n + o) into {bot: order}. Returns None if invalid."""
    x = np.asarray(bits).reshape(n, n)
    if not (x.sum(axis=0) == 1).all() or not (x.sum(axis=1) == 1).all():
        return None
    return {int(b): int(np.argmax(x[b])) for b in range(n)}


def assignment_to_index(perm):
    """Bitstring index for an assignment given as a list perm[bot] = order."""
    n = len(perm)
    return sum(1 << (b * n + perm[b]) for b in range(n))

print("Running qubo.py was successful")