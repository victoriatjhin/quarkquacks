import random
import warnings

import numpy as np
import pytest

from quarkquacks import Campus
from quarkquacks.carpool import CarpoolProblem, make_carpool_scenario, solve_carpool
from quarkquacks.qubo import qubo_cost_vector
from quarkquacks.routes import check_frames, simulate


@pytest.fixture(scope="module")
def campus():
    return Campus()


def test_qubo_minimum_is_best_valid_plan(campus):
    """The lowest QUBO score must be the true best plan, so QAOA aims at the right answer."""
    for seed in range(8):
        problem = CarpoolProblem(campus, make_carpool_scenario(campus, 2, 4, random.Random(seed)))
        qubo = problem.build_qubo()
        z = int(np.argmin(qubo_cost_vector(qubo)))
        bits = [(z >> k) & 1 for k in range(qubo.get_num_vars())]
        assert problem.plan_cost(problem.decode(bits)) == pytest.approx(problem.brute_force()[0])


@pytest.mark.parametrize("sampler,shots", [("qaoa", 1000), ("qaoa", 5), ("random", 5)])
def test_always_returns_a_valid_plan(campus, sampler, shots):
    """Even with very few samples, check -> repair -> fallback always gives a valid plan."""
    for seed in range(5):
        scenario = make_carpool_scenario(campus, 2, 4, random.Random(seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = solve_carpool(campus, scenario, shots=shots, seed=seed, sampler=sampler)
        assert r["problem"].plan_cost(r["plan"]) == pytest.approx(r["cost"])
        assert 0 < r["approx_ratio"] <= 1 + 1e-9


def test_impossible_request_is_rejected(campus):
    """5 orders cannot fit on 2 geese carrying 2 each: refuse up front instead of guessing."""
    with pytest.raises(ValueError):
        CarpoolProblem(campus, make_carpool_scenario(campus, 2, 5, random.Random(0)))


def test_carpool_plan_runs_collision_free(campus):
    """The chosen plan executes on the collision-free routes and every order is delivered."""
    for seed in range(4):
        scenario = make_carpool_scenario(campus, 2, 4, random.Random(seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = solve_carpool(campus, scenario, seed=seed)
        busy = {g: min(orders) for g, orders in r["plan"].items() if orders}  # first-come priority
        frames, stats = simulate(campus, scenario, busy, n_people=3, seed=seed,
                                 waypoints=r["problem"].waypoints(r["plan"]))
        check_frames(frames)
        assert stats["all_delivered"]
