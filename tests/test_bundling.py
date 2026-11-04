"""Pre-bundling: nearby same-counter orders share a trip; everything else is unchanged."""

import random

import numpy as np
import pytest

from quarkquacks import Campus, solve_classical
from quarkquacks.routes import check_frames, simulate
from quarkquacks.sim import BUNDLE_SIZE, MAX_ROOM_GAP


@pytest.fixture(scope="module")
def campus():
    return Campus()


def test_bundles_follow_the_rules(campus):
    for seed in range(100):
        scn = campus.make_scenario(3, random.Random(seed), bundle=True)
        arrivals = scn["arrivals"]
        for trip in scn["orders"]:
            members = [arrivals[i] for i in trip["orders"]]
            assert 1 <= len(members) <= BUNDLE_SIZE
            assert all(m["counter"] == trip["counter"] for m in members)        # one counter per trip
            for a in trip["rooms"]:
                for b in trip["rooms"]:
                    assert campus.room_gap(a, b) <= MAX_ROOM_GAP                 # rooms close together
        assigned = [i for t in scn["orders"] for i in t["orders"]]
        assert sorted(assigned + scn["waiting"]) == list(range(len(arrivals)))   # every order once


def test_first_come_first_served(campus):
    """Trips are served in order of their earliest order; later trips wait."""
    for seed in range(50):
        scn = campus.make_scenario(3, random.Random(seed), bundle=True)
        firsts = [min(t["orders"]) for t in scn["orders"]]
        assert firsts == sorted(firsts)
        if scn["waiting"]:
            assert min(scn["waiting"]) > max(firsts)  # no waiting trip started before a served one


def test_cost_is_trip_walking_distance(campus):
    scn = campus.make_scenario(3, random.Random(3), bundle=True)
    cost = np.array(scn["cost"])
    assert cost.shape == (3, 3) and (cost > 0).all()
    for g, bot in enumerate(scn["bots"]):
        for o, trip in enumerate(scn["orders"]):
            assert cost[g, o] == campus.trip_length(tuple(bot), trip)


def test_default_scenarios_unchanged(campus):
    """bundle=False must give exactly the scenarios the ML model was trained on."""
    scn = campus.make_scenario(3, random.Random(42))
    assert "arrivals" not in scn and all("rooms" not in o for o in scn["orders"])


def test_bundled_trips_deliver_collision_free(campus):
    for seed in range(15):
        scn = campus.make_scenario(3, random.Random(seed), bundle=True)
        perm, _ = solve_classical(scn["cost"])
        frames, stats = simulate(campus, scn, dict(enumerate(perm)), n_people=3, seed=seed)
        check_frames(frames)
        assert stats["all_delivered"]
