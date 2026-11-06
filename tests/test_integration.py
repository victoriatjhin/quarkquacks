"""Integration test: every part of QuarkQuacks working together.

Runs the whole chain on fresh scenarios, the way the demo does:

    simulator (A) -> ML angles (C) -> Qiskit QAOA (B) -> classical check (C)
    -> collision-free routes (A)

Each assertion message names which hand-off broke, so a failure points at the
person whose part to look at. Run with: pytest tests/test_integration.py -v
"""

import random
import warnings

import numpy as np
import pytest

from ml.model import predict_angles
from quantum.solver import solve_qaoa
from quarkquacks import Campus, solve_classical
from quarkquacks.routes import check_frames, simulate

pytestmark = pytest.mark.integration


def _total(cost, assignment):
    return sum(cost[g][o] for g, o in assignment.items())


@pytest.fixture(scope="module")
def campus():
    return Campus()


@pytest.mark.parametrize("n_geese,seed,bundle", [(2, 101, False), (3, 202, False), (4, 303, False),
                                                 (4, 404, True)])
def test_full_pipeline(campus, n_geese, seed, bundle):
    # 1. Simulator -> scenario contract
    scenario = campus.make_scenario(n_geese, random.Random(seed), bundle=bundle, priority=True)
    cost = np.asarray(scenario["cost"], dtype=float)
    assert cost.shape == (n_geese, n_geese), "simulator: cost matrix must be n x n (geese x orders)"
    assert (cost > 0).all(), "simulator: all costs must be positive (QUBO scaling divides by max)"
    assert len(scenario["bots"]) == n_geese and len(scenario["orders"]) == n_geese, \
        "simulator: need one goose position and one order per goose"

    # 2. ML warm start
    gamma, beta = predict_angles(scenario)
    assert np.isfinite([gamma, beta]).all(), "ML: predicted angles must be real numbers"

    # 3. Qiskit QAOA, warm-started
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = solve_qaoa(scenario["cost"], warm_start=(gamma, beta), seed=seed)
    assert warm["assignment"] is not None, "solver: QAOA returned an invalid plan (check build_qubo / penalty)"
    assert sorted(warm["assignment"].values()) == list(range(n_geese)), \
        "solver: every order must go to exactly one goose"

    # 4. Classical answer key. Up to 9 qubits the solver's samples cover the best plan;
    #    at 16 qubits QAOA (p=1) finds it most of the time, so allow a near-best plan.
    _, best = solve_classical(scenario["cost"])
    got = _total(scenario["cost"], warm["assignment"])
    if n_geese <= 3:
        assert got == pytest.approx(best), "solver vs classical: warm-started QAOA did not find the optimal plan"
    else:
        assert best <= got <= 1.10 * best, "solver vs classical: QAOA plan is more than 10% worse than the best"

    # 5. Collision-free routes for the chosen plan
    frames, stats = simulate(campus, scenario, warm["assignment"], n_people=3, seed=seed)
    check_frames(frames)  # raises if geese collide, swap or step on a person
    assert stats["all_delivered"], "routes: not every order was delivered in time"


@pytest.mark.parametrize("mode", [[], ["--bundle"], ["--bundle-size", "3", "--cross-counter"], ["--carpool"]])
def test_run_pipeline_script(tmp_path, mode):
    """The one-command demo runs end to end in every mode and saves its plot and animation."""
    import run_pipeline

    out, gif = tmp_path / "plan.png", tmp_path / "run.gif"
    args = ["--geese", "2", "--seed", "1", "--no-compare", "--out", str(out), "--gif", str(gif)] + mode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline.main(args)
    assert out.exists(), "run_pipeline.py: plan plot was not saved"
    assert gif.exists(), "run_pipeline.py: animation was not saved"


def test_carpool_end_to_end(campus):
    """Carpool: QAOA plan is valid, matches the best possible here, and delivers collision-free."""
    from quarkquacks.carpool import make_carpool_scenario, solve_carpool

    scenario = make_carpool_scenario(campus, 2, 4, random.Random(7))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_carpool(campus, scenario, shots=1000, seed=7)
    assert r["problem"].plan_cost(r["plan"]) is not None, "carpool: returned plan breaks a rule"
    assert r["approx_ratio"] == pytest.approx(1.0), "carpool: 1000 samples should reach the best plan"
    busy = {g: min(o) for g, o in r["plan"].items() if o}
    frames, stats = simulate(campus, scenario, busy, n_people=3, seed=7, waypoints=r["problem"].waypoints(r["plan"]))
    check_frames(frames)
    assert stats["all_delivered"], "carpool routes: not every order was delivered"
