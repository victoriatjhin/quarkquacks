"""Exact classical solver: the ground truth QAOA is compared against (Person C)."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_classical(cost):
    """Return (perm, total_cost) where perm[bot] = order. Exact, runs instantly."""
    cost = np.asarray(cost, dtype=float)
    rows, cols = linear_sum_assignment(cost)
    perm = [int(c) for _, c in sorted(zip(rows, cols))]
    return perm, float(cost[rows, cols].sum())
