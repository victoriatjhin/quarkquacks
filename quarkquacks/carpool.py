"""Carpool mode: QAOA decides which orders share a goose to minimise customer wait.

Separate add-on: it reads the campus map and the route planner but changes
neither, so the main pipeline and the ML warm start are unaffected.

How a plan is chosen (quantum proposes, classical checks):
  1. Pre-check: real trip costs for every goose and every 1- or 2-order trip
     (walking distances by BFS, best stop order); pairs that make a customer
     wait much longer than going alone are marked unrealistic.
  2. Carpool QUBO: x_g_o = 1 if goose g carries order o. Rules: every order
     exactly once, at most CAPACITY orders per goose, unrealistic pairs banned.
     Pair costs are the quadratic terms.
  3. QAOA (p=1) in Qiskit, sampled many times.
  4. Every sample is decoded and checked; invalid near-misses are repaired;
     if nothing valid appears, an exact classical search is the fallback.
  5. A final check runs on whatever plan is returned, so it is always valid.

    from quarkquacks.carpool import make_carpool_scenario, solve_carpool
"""

import itertools
import random

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from .qaoa_sim import grid_search, refine
from .qubo import qubo_cost_vector
from .sim import ITEMS, bfs_distances

CAPACITY = 2
MAX_PAIR_RATIO = 1.6  # a pair is realistic if its total wait <= 1.6 x the two solo waits


class Distances:
    """Shortest walking distance between any two cells, cached by source."""

    def __init__(self, grid):
        self.grid, self._cache = grid, {}

    def __call__(self, a, b):
        if a not in self._cache:
            self._cache[a] = bfs_distances(a, self.grid)
        return self._cache[a][b]


def make_carpool_scenario(campus, n_geese, n_orders, rng=None):
    """More orders than geese. Order index = arrival order (first come, first served)."""
    rng = rng or random.Random()
    named = set(campus.named.values())
    free = [c for c in campus.floor if c not in named]
    orders = []
    for _ in range(n_orders):
        item = rng.choice(list(ITEMS))
        orders.append({"item": item, "counter": ITEMS[item]["counter"], "room": rng.choice(campus.rooms)})
    return {"bots": rng.sample(free, n_geese), "orders": orders}


def best_trip(campus, dist, start, orders):
    """Fastest stop order for one goose carrying `orders` (pickups before drop-offs).

    Returns (total customer wait, stops as cells, wait per order,
    stops as (kind, order position, cell) tuples).
    """
    stops = []
    for k, o in enumerate(orders):
        stops += [("pick", k, campus.named[o["counter"]]), ("drop", k, campus.named[o["room"]])]
    best = None
    for seq in itertools.permutations(stops):
        picked, ok = set(), True
        for kind, k, _ in seq:
            if kind == "pick":
                picked.add(k)
            elif k not in picked:
                ok = False
                break
        if not ok:
            continue
        t, here, waits = 0, start, {}
        for kind, k, cell in seq:
            t += dist(here, cell)
            here = cell
            if kind == "drop":
                waits[k] = t
        total = sum(waits.values())
        if best is None or total < best[0]:
            best = (total, [cell for _, _, cell in seq], waits, list(seq))
    return best


class CarpoolProblem:
    """Trip costs and validity rules for one scenario."""

    def __init__(self, campus, scenario, capacity=CAPACITY, max_pair_ratio=MAX_PAIR_RATIO):
        self.campus, self.scenario, self.capacity = campus, scenario, capacity
        self.G, self.M = len(scenario["bots"]), len(scenario["orders"])
        dist = Distances(campus.grid)
        orders, bots = scenario["orders"], [tuple(b) for b in scenario["bots"]]
        self.solo = np.array([[best_trip(campus, dist, bots[g], [orders[o]])[0]
                               for o in range(self.M)] for g in range(self.G)], dtype=float)
        self.pair, self.allowed = {}, {}
        for g in range(self.G):
            for i, j in itertools.combinations(range(self.M), 2):
                total = best_trip(campus, dist, bots[g], [orders[i], orders[j]])[0]
                self.pair[g, i, j] = total
                self.allowed[g, i, j] = total <= max_pair_ratio * (self.solo[g, i] + self.solo[g, j])
        self._dist, self._bots = dist, bots
        if self.brute_force() is None:  # pre-check: never ask for an impossible plan
            raise ValueError("no valid carpool plan: too many orders for the geese, or pair limit too strict")

    # ----- checking plans -----
    def plan_cost(self, plan):
        """Total customer wait for {goose: [orders]}, or None if the plan breaks a rule."""
        seen = sorted(o for orders in plan.values() for o in orders)
        if seen != list(range(self.M)):
            return None  # an order missing or delivered twice
        total = 0.0
        for g, orders in plan.items():
            if len(orders) > self.capacity:
                return None
            if len(orders) == 1:
                total += self.solo[g, orders[0]]
            elif len(orders) == 2:
                i, j = sorted(orders)
                if not self.allowed[g, i, j]:
                    return None
                total += self.pair[g, i, j]
            elif len(orders) > 2:
                return None
        return total

    def brute_force(self):
        """Exact best plan by checking every option (tiny at hackathon sizes)."""
        best = None
        for choice in itertools.product(range(self.G), repeat=self.M):
            plan = {g: [o for o in range(self.M) if choice[o] == g] for g in range(self.G)}
            cost = self.plan_cost(plan)
            if cost is not None and (best is None or cost < best[0]):
                best = (cost, plan)
        return best

    def repair(self, plan):
        """Fix a near-miss: drop duplicates, then give missing orders to the cheapest goose with room."""
        plan = {g: list(dict.fromkeys(orders)) for g, orders in plan.items()}
        owners = {}
        for g, orders in plan.items():
            for o in orders:
                owners.setdefault(o, []).append(g)
        for o, gs in owners.items():  # delivered twice: keep the cheaper goose
            keep = min(gs, key=lambda g: self.solo[g, o])
            for g in gs:
                if g != keep:
                    plan[g].remove(o)
        for g in plan:  # over capacity or unrealistic pair: drop the latest-arriving orders
            plan[g].sort()
            while len(plan[g]) > self.capacity:
                plan[g].pop()
            if len(plan[g]) == 2 and not self.allowed[(g, *plan[g])]:
                plan[g].pop()
        for o in range(self.M):  # missing orders
            if not any(o in orders for orders in plan.values()):
                options = []
                for g in plan:
                    trial = {h: list(v) for h, v in plan.items()}
                    trial[g] = sorted(trial[g] + [o])
                    if len(trial[g]) <= self.capacity and (
                            len(trial[g]) < 2 or self.allowed[(g, *trial[g])]):
                        options.append((self.solo[g, o], g))
                if not options:
                    return None
                plan[min(options)[1]].append(o)
        return plan if self.plan_cost(plan) is not None else None

    # ----- the quantum problem -----
    def build_qubo(self, penalty=None):
        """Carpool QUBO; costs scaled so the largest solo trip is 1."""
        scale = self.solo.max()
        deltas = {k: (self.pair[k] - self.solo[k[0], k[1]] - self.solo[k[0], k[2]]) / scale
                  for k in self.pair}
        max_delta = max(deltas.values())
        if penalty is None:  # bigger than anything a broken rule could save
            penalty = 1.0 + (self.capacity + 1) * max(max_delta, 0.1)
        qp = QuadraticProgram("carpool")
        for g in range(self.G):
            for o in range(self.M):
                qp.binary_var(f"x_{g}_{o}")
        linear = {f"x_{g}_{o}": self.solo[g, o] / scale for g in range(self.G) for o in range(self.M)}
        quadratic = {}
        for (g, i, j), d in deltas.items():
            # realistic pair: its extra wait; unrealistic pair: banned with the penalty
            quadratic[f"x_{g}_{i}", f"x_{g}_{j}"] = d if self.allowed[g, i, j] else penalty
        qp.minimize(linear=linear, quadratic=quadratic)
        for o in range(self.M):
            qp.linear_constraint({f"x_{g}_{o}": 1 for g in range(self.G)}, "==", 1, f"order_{o}")
        for g in range(self.G):
            qp.linear_constraint({f"x_{g}_{o}": 1 for o in range(self.M)}, "<=", self.capacity, f"cap_{g}")
        return QuadraticProgramToQubo(penalty=penalty).convert(qp)

    def decode(self, bits):
        """First G*M bits -> {goose: [orders]} (may be invalid)."""
        return {g: [o for o in range(self.M) if bits[g * self.M + o]] for g in range(self.G)}

    def waypoints(self, plan):
        """Stops for each busy goose, in the fastest order, for the route planner."""
        out = {}
        for g, orders in plan.items():
            if orders:
                ords = [self.scenario["orders"][o] for o in sorted(orders)]
                out[g] = best_trip(self.campus, self._dist, self._bots[g], ords)[1]
        return out

    def drop_stops(self, plan):
        """{goose: [(order index, waypoint index of its drop-off)]}, matching waypoints()."""
        out = {}
        for g, orders in plan.items():
            if orders:
                ids = sorted(orders)
                seq = best_trip(self.campus, self._dist, self._bots[g],
                                [self.scenario["orders"][o] for o in ids])[3]
                out[g] = [(ids[k], i) for i, (kind, k, _) in enumerate(seq) if kind == "drop"]
        return out


def choose_plan(problem, counts):
    """Pick the best valid plan from sampled bitstrings: check, repair, then fall back.

    counts: {bitstring: times seen}, Qiskit style (rightmost character = qubit 0).
    Returns (cost, plan, source, valid_shot_rate).
    """
    shots = sum(counts.values())
    best, valid_shots, repaired_best = None, 0, None
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        bits = [int(b) for b in reversed(key)]
        plan = problem.decode(bits)
        cost = problem.plan_cost(plan)
        if cost is not None:
            valid_shots += count
            if best is None or cost < best[0]:
                best = (cost, plan)
        else:
            fixed = problem.repair(plan)
            if fixed is not None:
                c = problem.plan_cost(fixed)
                if repaired_best is None or c < repaired_best[0]:
                    repaired_best = (c, fixed)

    if best is not None and (repaired_best is None or best[0] <= repaired_best[0]):
        cost, plan, source = best[0], best[1], "sampled"
    elif repaired_best is not None:
        cost, plan, source = repaired_best[0], repaired_best[1], "repaired"
    else:
        cost, plan = problem.brute_force()
        source = "fallback"
    assert problem.plan_cost(plan) is not None, "final safety check failed"  # never ship a broken plan
    return cost, plan, source, valid_shots / shots


def qaoa_counts(qubo, shots, seed):
    """p=1 QAOA in Qiskit with angles tuned on the exact cost table; returns sample counts."""
    C = qubo_cost_vector(qubo)
    scale = float(np.abs(C).max())
    _, gamma, beta = refine(C / scale, *grid_search(C / scale, 24)[1:])
    operator, _ = qubo.to_ising()
    circuit = QAOAAnsatz(operator, reps=1).assign_parameters([beta, gamma / scale])  # Qiskit: [beta, gamma]
    circuit.measure_all()
    circuit = transpile(circuit, basis_gates=["h", "rx", "rz", "rzz", "cx"])  # much faster to sample
    return StatevectorSampler(seed=seed).run([circuit], shots=shots).result()[0].data.meas.get_counts()


def random_counts(n_qubits, shots, seed):
    """Baseline: uniformly random bitstrings, i.e. guessing without QAOA."""
    rng = np.random.default_rng(seed)
    counts = {}
    for z in rng.integers(0, 2 ** n_qubits, size=shots):
        key = format(int(z), f"0{n_qubits}b")
        counts[key] = counts.get(key, 0) + 1
    return counts


def solve_carpool(campus, scenario, shots=1000, seed=0, penalty=None, sampler="qaoa"):
    """Return the chosen plan plus how it was found and how good it is.

    sampler="random" swaps QAOA for random guessing, as a fair baseline.
    """
    problem = CarpoolProblem(campus, scenario)
    qubo = problem.build_qubo(penalty)
    n_qubits = qubo.get_num_vars()
    counts = (qaoa_counts(qubo, shots, seed) if sampler == "qaoa"
              else random_counts(n_qubits, shots, seed))
    cost, plan, source, valid_rate = choose_plan(problem, counts)
    optimum, _ = problem.brute_force()
    return {"plan": plan, "cost": cost, "source": source, "optimal_cost": optimum,
            "approx_ratio": optimum / cost, "valid_shot_rate": valid_rate,
            "n_qubits": n_qubits, "problem": problem}
