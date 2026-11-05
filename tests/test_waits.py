"""Per-order wait-time counter: every order gets a delivery step that matches the simulation."""

import random

import pytest

from quarkquacks import Campus, solve_classical
from quarkquacks.carpool import CarpoolProblem, make_carpool_scenario
from quarkquacks.routes import assignment_drops, order_wait_times, simulate


@pytest.fixture(scope="module")
def campus():
    return Campus()


def _optimal(scenario):
    perm, _ = solve_classical(scenario["cost"])
    return dict(enumerate(perm))


@pytest.mark.parametrize("n,bundle", [(2, False), (4, False), (4, True)])
def test_every_order_gets_a_wait_time(campus, n, bundle):
    for seed in range(5):
        scn = campus.make_scenario(n, random.Random(seed), bundle=bundle)
        plan = _optimal(scn)
        frames, stats = simulate(campus, scn, plan, seed=seed)
        waits = order_wait_times(stats, assignment_drops(scn, plan))
        expected = sum(len(t.get("rooms", [t["room"]])) for t in scn["orders"])
        assert len(waits) == expected, "one wait entry per delivered order"
        assert all(w["wait"] is not None for w in waits), "every order should be delivered"
        assert max(w["wait"] for w in waits) == stats["ticks"], "last drop-off ends the run"
        for w in waits:  # the goose is gone from the map from its last drop-off on
            assert 0 < w["wait"] <= stats["ticks"]


def test_wait_matches_frames(campus):
    """The recorded drop-off step is the first frame where that stop counts as reached."""
    scn = campus.make_scenario(3, random.Random(11))
    plan = _optimal(scn)
    frames, stats = simulate(campus, scn, plan, seed=11)
    for w in order_wait_times(stats, assignment_drops(scn, plan)):
        first = next(f["t"] for f in frames if f["stops"][w["goose"]] > w["stop"])
        assert first == w["wait"]


def test_bundled_orders_keep_their_arrival_ids(campus):
    scn = campus.make_scenario(4, random.Random(3), bundle=True)
    drops = assignment_drops(scn, _optimal(scn))
    served = sorted(i for t in scn["orders"] for i in t["orders"])
    assert [d["order"] for d in drops] == served


def test_carpool_drop_stops_match_waypoints(campus):
    for seed in range(4):
        scn = make_carpool_scenario(campus, 2, 4, random.Random(seed))
        problem = CarpoolProblem(campus, scn)
        _, plan = problem.brute_force()
        wps = problem.waypoints(plan)
        for g, stops in problem.drop_stops(plan).items():
            assert sorted(o for o, _ in stops) == sorted(plan[g])
            for o, stop in stops:
                assert wps[g][stop] == campus.named[scn["orders"][o]["room"]]
