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


@pytest.mark.parametrize("n_geese,seed", [(2, 101), (3, 202)])
def test_full_pipeline(campus, n_geese, seed):
    # 1. Simulator -> scenario contract (Person A)
    scenario = campus.make_scenario(n_geese, random.Random(seed))
    cost = np.asarray(scenario["cost"], dtype=float)
    assert cost.shape == (n_geese, n_geese), "simulator: cost matrix must be n x n (geese x orders)"
    assert (cost > 0).all(), "simulator: all costs must be positive (QUBO scaling divides by max)"
    assert len(scenario["bots"]) == n_geese and len(scenario["orders"]) == n_geese, \
        "simulator: need one goose position and one order per goose"

    # 2. ML warm start (Person C)
    gamma, beta = predict_angles(scenario)
    assert np.isfinite([gamma, beta]).all(), "ML: predicted angles must be real numbers"

    # 3. Qiskit QAOA, warm-started (Person B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = solve_qaoa(scenario["cost"], warm_start=(gamma, beta), seed=seed)
        cold = solve_qaoa(scenario["cost"], seed=seed)
    assert warm["assignment"] is not None, "solver: QAOA returned an invalid plan (check build_qubo / penalty)"
    assert sorted(warm["assignment"].values()) == list(range(n_geese)), \
        "solver: every order must go to exactly one goose"

    # 4. Classical answer key (Person C)
    _, best = solve_classical(scenario["cost"])
    assert _total(scenario["cost"], warm["assignment"]) == pytest.approx(best), \
        "solver vs classical: warm-started QAOA did not find the optimal plan"
    assert warm["evaluations"] <= cold["evaluations"], \
        "ML + solver: warm start should not need more optimizer steps than a random start"

    # 5. Collision-free routes for the chosen plan (Person A)
    frames, stats = simulate(campus, scenario, warm["assignment"], n_people=3, seed=seed)
    check_frames(frames)  # raises if geese collide, swap or step on a person
    assert stats["all_delivered"], "routes: not every order was delivered in time"


def test_run_pipeline_script(tmp_path, monkeypatch):
    """The one-command demo script runs end to end and saves its plot."""
    import run_pipeline

    out = tmp_path / "plan.png"
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--geese", "2", "--seed", "1", "--out", str(out)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline.main()
    assert out.exists(), "run_pipeline.py: plan plot was not saved"
