import random

import numpy as np

from quarkquacks import Campus, normalized_cost_vector, solve_classical
from quarkquacks.qubo import assignment_to_index, bits_to_assignment


def test_scenario_shape():
    scn = Campus().make_scenario(3, random.Random(1))
    assert len(scn["bots"]) == 3 and len(scn["orders"]) == 3
    assert np.array(scn["cost"]).shape == (3, 3)


def test_qubo_minimum_is_true_optimum():
    """The best bitstring of the QUBO must be the best real assignment."""
    rng, campus = random.Random(7), Campus()
    for _ in range(20):
        scn = campus.make_scenario(rng.choice([2, 3]), rng)
        C = normalized_cost_vector(scn["cost"])
        perm, _ = solve_classical(scn["cost"])
        assert np.isclose(C.min(), C[assignment_to_index(perm)])


def test_decode_rejects_invalid():
    assert bits_to_assignment([1, 0, 0, 1], 2) == {0: 0, 1: 1}
    assert bits_to_assignment([1, 1, 0, 0], 2) is None
