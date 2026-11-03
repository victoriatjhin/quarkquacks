"""Fast exact simulation of p=1 QAOA with numpy.

Used for generating training data and quick experiments. The real Qiskit QAOA
lives in quantum/solver.py; both use the same cost normalisation.
"""

import numpy as np
from scipy.optimize import minimize


def qaoa_p1_probs(C, gamma, beta):
    """Probability of each bitstring after one QAOA layer on cost vector C."""
    nq = int(np.log2(len(C)))
    psi = np.full(len(C), 1 / np.sqrt(len(C)), dtype=complex)
    psi *= np.exp(-1j * gamma * C)                      # cost layer
    cb, sb = np.cos(beta), -1j * np.sin(beta)
    for k in range(nq):                                 # mixer: RX(2*beta) on each qubit
        psi = psi.reshape(-1, 2, 2 ** k)
        a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
        psi[:, 0, :], psi[:, 1, :] = cb * a + sb * b, sb * a + cb * b
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2


def energy(C, gamma, beta):
    return float(qaoa_p1_probs(C, gamma, beta) @ C)


def grid_search(C, steps=24):
    """Best (energy, gamma, beta) on a steps x steps grid."""
    best = (np.inf, 0.0, 0.0)
    for g in np.linspace(0, 2 * np.pi, steps, endpoint=False):
        for b in np.linspace(0, np.pi, steps, endpoint=False):
            e = energy(C, g, b)
            if e < best[0]:
                best = (e, g, b)
    return best


def refine(C, gamma, beta, tol=1e-4):
    """Polish grid-search angles with a short local search.

    The grid only covers gamma in [0, 2*pi); on larger problems the best gamma
    can lie outside it or between grid points, so the grid answer alone makes a
    poor training label. Returns (energy, gamma, beta).
    """
    res = minimize(lambda x: energy(C, x[0], x[1]), [gamma, beta], method="COBYLA",
                   options={"maxiter": 400, "tol": tol, "rhobeg": 0.3})
    return float(res.fun), float(res.x[0]), float(res.x[1])
