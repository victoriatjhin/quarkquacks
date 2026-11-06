"""Multi-order bundling: up to 3 orders per trip, and bundles across nearby counters."""

import random

import numpy as np
import pytest

from quarkquacks import Campus, solve_classical
from quarkquacks.routes import assignment_drops, check_frames, goose_waypoints, order_wait_times, simulate
from quarkquacks.sim import MAX_COUNTER_GAP, MAX_ROOM_GAP

MODES = [(2, False), (2, True), (3, False), (3, True)]


@pytest.fixture(scope="module")
def campus():
    return Campus()


def _plan(scn):
    perm, _ = solve_classical(scn["cost"])
    return dict(enumerate(perm))


@pytest.mark.parametrize("size,cross", MODES)
def test_bundles_follow_the_rules(campus, size, cross):
    for seed in range(60):
        scn = campus.make_scenario(4, random.Random(seed), bundle=True, bundle_size=size, cross_counter=cross)
        arrivals = scn["arrivals"]
        assert len(arrivals) == size * 4, "k orders per goose arrive when bundles hold up to k"
        for trip in scn["orders"]:
            members = [arrivals[i] for i in trip["orders"]]
            assert 1 <= len(members) <= size
            assert trip["rooms"] == [m["room"] for m in members], "rooms and order ids must stay aligned"
            assert trip["items"] == [m["item"] for m in members], "items and order ids must stay aligned"
            assert set(trip["counters"]) == {m["counter"] for m in members}
            if cross:
                for a in trip["counters"]:
                    for b in trip["counters"]:
                        assert campus._counter_gap(a, b) <= MAX_COUNTER_GAP
            else:
                assert len(trip["counters"]) == 1, "without cross_counter a trip uses one counter"
            for a in trip["rooms"]:
                for b in trip["rooms"]:
                    assert campus.room_gap(a, b) <= MAX_ROOM_GAP
        served = [i for t in scn["orders"] for i in t["orders"]]
        assert sorted(served + scn["waiting"]) == list(range(len(arrivals))), "every order exactly once"
        firsts = [min(t["orders"]) for t in scn["orders"]]
        assert firsts == sorted(firsts), "trips served first come, first served"


def test_new_modes_actually_happen(campus):
    """Across seeds, some trips really carry 3 orders, and some really mix counters."""
    three = mixed = 0
    for seed in range(60):
        big = campus.make_scenario(4, random.Random(seed), bundle=True, bundle_size=3)
        cross = campus.make_scenario(4, random.Random(seed), bundle=True, cross_counter=True)
        three += any(len(t["orders"]) == 3 for t in big["orders"])
        mixed += any(len(t["counters"]) == 2 for t in cross["orders"])
    assert three > 10 and mixed > 10


@pytest.mark.parametrize("size,cross", MODES)
def test_cost_matches_the_walk(campus, size, cross):
    """Cost = walking steps along goose -> every counter -> every room, in the planned order."""
    for seed in range(20):
        scn = campus.make_scenario(3, random.Random(seed), bundle=True, bundle_size=size, cross_counter=cross)
        for g, bot in enumerate(scn["bots"]):
            for o, trip in enumerate(scn["orders"]):
                stops = goose_waypoints(campus, scn, {g: o})[g]
                cells = [tuple(bot)] + stops
                walk = sum(campus._dist(a, b) for a, b in zip(cells, cells[1:]))
                assert scn["cost"][g][o] == walk


def test_trip_order_is_shortest(campus):
    """No other visiting order of a trip's counters and rooms walks less."""
    import itertools
    for seed in range(30):
        scn = campus.make_scenario(3, random.Random(seed), bundle=True, bundle_size=3, cross_counter=True)
        for trip in scn["orders"]:
            def walk(cs, rooms):
                cells = [campus.named[c] for c in cs] + [campus.named[r] for r in rooms]
                return sum(campus._dist(a, b) for a, b in zip(cells, cells[1:]))
            best = min(walk(cs, rs) for cs in itertools.permutations(trip["counters"])
                       for rs in itertools.permutations(trip["rooms"]))
            assert walk(trip["counters"], trip["rooms"]) == best


@pytest.mark.parametrize("size,cross", MODES)
def test_delivers_collision_free_with_waits(campus, size, cross):
    for seed in range(8):
        scn = campus.make_scenario(3, random.Random(seed), bundle=True, priority=True, bundle_size=size,
                                   cross_counter=cross)
        plan = _plan(scn)
        frames, stats = simulate(campus, scn, plan, seed=seed)
        check_frames(frames)
        assert stats["all_delivered"]
        waits = order_wait_times(stats, assignment_drops(scn, plan))
        assert len(waits) == sum(len(t["orders"]) for t in scn["orders"])
        assert all(w["wait"] is not None for w in waits)
        # each wait is the step the goose reached that order's room
        for w in waits:
            room = scn["arrivals"][w["order"]]["room"]
            reached = next(f["t"] for f in frames if f["stops"][w["goose"]] > w["stop"])
            assert reached == w["wait"]
            assert goose_waypoints(campus, scn, plan)[w["goose"]][w["stop"]] == campus.named[room]


def test_default_bundling_unchanged(campus):
    """bundle_size 2, one counter: same trips as the original rule (only the id order is fixed)."""
    for seed in range(40):
        scn = campus.make_scenario(3, random.Random(seed), bundle=True)
        assert len(scn["arrivals"]) == 6
        for t in scn["orders"]:
            assert len(t["counters"]) == 1 and len(t["orders"]) <= 2


def test_bundle_size_is_checked(campus):
    with pytest.raises(ValueError):
        campus.make_scenario(3, random.Random(0), bundle=True, bundle_size=4)


def test_priority_still_breaks_ties(campus):
    from quantum.quantum_benchmark import is_degenerate
    tied = sum(is_degenerate(campus.make_scenario(4, random.Random(s), bundle=True, priority=True,
                                                  bundle_size=3, cross_counter=True)["cost"])
               for s in range(40))
    assert tied <= 2
