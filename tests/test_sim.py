"""Campus simulator contract: what the QUBO, ML model and routes rely on."""

import random

import numpy as np
import pytest

from quarkquacks import Campus
from quarkquacks.sim import bfs_distances


@pytest.fixture(scope="module")
def campus():
    return Campus()


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_scenario_shape_and_costs(campus, n):
    scn = campus.make_scenario(n, random.Random(n))
    cost = np.array(scn["cost"])
    assert cost.shape == (n, n)                     # geese x orders, square
    assert (cost > 0).all()                         # QUBO scaling divides by the max
    assert len(set(map(tuple, scn["bots"]))) == n   # no two geese start on one cell
    assert all(campus.grid[r][c] != "#" for r, c in scn["bots"])


def test_same_seed_same_scenario(campus):
    """Training data and benchmarks must be repeatable."""
    a = campus.make_scenario(3, random.Random(42))
    b = campus.make_scenario(3, random.Random(42))
    assert a == b


def test_cost_is_walking_distance(campus):
    scn = campus.make_scenario(3, random.Random(7))
    for g, bot in enumerate(scn["bots"]):
        for o, order in enumerate(scn["orders"]):
            counter, room = campus.named[order["counter"]], campus.named[order["room"]]
            expected = bfs_distances(tuple(bot))[counter] + bfs_distances(counter)[room]
            assert scn["cost"][g][o] == expected


def test_every_cell_reachable(campus):
    """No goose can be stuck: every hallway cell connects to every counter."""
    for counter in campus.dist_from.values():
        assert set(counter) == set(campus.floor)
