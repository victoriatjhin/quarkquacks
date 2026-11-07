"""Campus map, shortest paths and random goose delivery scenarios.

Optional pre-bundling (make_scenario(..., bundle=True)): before assignment,
orders from the same counter whose rooms are close together are grouped into
one trip, so one goose can carry two orders. Off by default, so the training
data and results are unchanged.
"""

import itertools
import random
from collections import deque

import numpy as np

# '#' wall   '.' hallway   'A'/'B'/'C' food court counters   '1'-'6' study rooms
CAMPUS = [
    "###################",
    "#A..B..C#....1....#",
    "#.......#.........#",
    "#.......####.###..#",
    "#.................#",
    "####.####...#######",
    "#2.....#.....#...3#",
    "#......#.....#....#",
    "#...........#.....#",
    "#4.....#.....5...6#",
    "###################",
]

COUNTERS = {"A": "coffee", "B": "pizza", "C": "bubble tea"}
ITEMS = {item: {"counter": counter} for counter, item in COUNTERS.items()}
MAX_ROOM_GAP = 6       # bundle two orders only if their rooms are <= 6 steps apart
BUNDLE_SIZE = 2        # a goose carries at most two bundled orders (default)
MAX_BUNDLE_SIZE = 3    # largest bundle allowed (--bundle-size 3)
MAX_COUNTER_GAP = 6    # cross-counter bundles: counters at most 6 steps apart


def find_cells(grid=CAMPUS):
    """Return {symbol: (row, col)} for counters/rooms, plus all walkable cells."""
    named, floor = {}, []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch != "#":
                floor.append((r, c))
            if ch not in "#.":
                named[ch] = (r, c)
    return named, floor


def bfs_distances(start, grid=CAMPUS):
    """Shortest walking distance from start to every reachable cell."""
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if grid[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


class Campus:
    """Precomputes distances from every counter so scenarios are fast to build."""

    def __init__(self, grid=CAMPUS):
        self.grid = grid
        self.named, self.floor = find_cells(grid)
        self.rooms = sorted(k for k in self.named if k.isdigit())
        self.dist_from = {self.named[k]: bfs_distances(self.named[k], grid) for k in COUNTERS}
        self._room_dist = {}

    def room_gap(self, room_a, room_b):
        """Walking distance between two study rooms."""
        a = self.named[room_a]
        if a not in self._room_dist:
            self._room_dist[a] = bfs_distances(a, self.grid)
        return self._room_dist[a][self.named[room_b]]

    def bundle_orders(self, orders, max_room_gap=MAX_ROOM_GAP, bundle_size=BUNDLE_SIZE,
                      cross_counter=False, max_counter_gap=MAX_COUNTER_GAP):
        """Group orders first-come-first-served into trips of at most bundle_size.

        A new order joins the earliest open trip whose rooms are all within
        max_room_gap steps of its room, and whose counter is the same (default)
        or, with cross_counter=True, within max_counter_gap steps of every counter
        already in the trip. Otherwise it starts a new trip. Returns trips in order
        of their earliest order; each trip visits its counters, then its rooms, in
        the order that walks least ("counters", "rooms", "orders" stay aligned).
        """
        if not 1 <= bundle_size <= MAX_BUNDLE_SIZE:
            raise ValueError(f"bundle_size must be 1-{MAX_BUNDLE_SIZE}")
        bundles = []
        for idx, order in enumerate(orders):
            for b in bundles:
                counters = {orders[i]["counter"] for i in b["orders"]}
                if cross_counter:
                    counter_ok = all(self._counter_gap(c, order["counter"]) <= max_counter_gap for c in counters)
                else:
                    counter_ok = counters == {order["counter"]}
                if (len(b["orders"]) < bundle_size and counter_ok
                        and all(self.room_gap(orders[i]["room"], order["room"]) <= max_room_gap
                                for i in b["orders"])):
                    b["orders"].append(idx)
                    break
            else:
                bundles.append({"orders": [idx]})
        return [self._plan_trip(orders, b["orders"]) for b in bundles]

    def _counter_gap(self, a, b):
        return self.dist_from[self.named[a]][self.named[b]]

    def _plan_trip(self, orders, ids):
        """Trip for orders `ids`: visit counters, then rooms, in the order that walks least.

        The goose's own walk to the first counter is added per goose in trip_length.
        """
        counters = list(dict.fromkeys(orders[i]["counter"] for i in ids))
        best = None
        for cs in itertools.permutations(counters):
            for drop in itertools.permutations(ids):
                stops = [self.named[c] for c in cs] + [self.named[orders[i]["room"]] for i in drop]
                walk = sum(self._dist(a, b) for a, b in zip(stops, stops[1:]))
                if best is None or walk < best[0]:
                    best = (walk, list(cs), list(drop))
        _, cs, drop = best
        return {"item": orders[drop[0]]["item"], "counter": cs[0], "counters": cs,
                "orders": drop, "rooms": [orders[i]["room"] for i in drop],
                "items": [orders[i]["item"] for i in drop], "room": orders[drop[-1]]["room"]}

    def trip_length(self, start, order):
        """Steps for one goose: start -> every counter -> every room of the order or bundle."""
        counters = order.get("counters", [order["counter"]])
        first = self.named[counters[0]]
        steps = self.dist_from[first][start]
        here = first
        for cell in [self.named[c] for c in counters[1:]] + [self.named[r] for r in order.get("rooms", [order["room"]])]:
            steps += self.dist_from[here][cell] if here in self.dist_from else self._dist(here, cell)
            here = cell
        return steps

    def _dist(self, a, b):
        if a not in self._room_dist:
            self._room_dist[a] = bfs_distances(a, self.grid)
        return self._room_dist[a][b]

    def make_scenario(self, n, rng=None, bundle=False, max_room_gap=MAX_ROOM_GAP, priority=False,
                      bundle_size=BUNDLE_SIZE, cross_counter=False):
        """n geese and n orders. Cost = walking distance (goose -> counter -> room).

        bundle=True: 2n orders arrive, nearby same-counter orders are bundled, and
        the first n trips (first come, first served) are assigned to the n geese;
        later orders wait for the next round ("waiting" in the result).
        bundle_size (up to 3) and cross_counter (orders from nearby counters may
        share a trip) widen the bundling rule; with bundle_size k, k*n orders arrive.

        priority=True: cost = priority x walking distance, where each order (or
        trip) gets its own priority by arrival: the first gets n, the last gets 1.
        Plain distance makes most scenarios degenerate (many assignments tie),
        because the counter -> room leg is the same for every goose, so swapping
        the geese on two same-counter orders never changes the total. A distinct
        priority per order breaks those ties. The plain distances are kept in
        "distance".
        """
        rng = rng or random.Random()
        scn = (self._make_bundled(n, rng, max_room_gap, bundle_size, cross_counter) if bundle
               else self._make_plain(n, rng))
        return add_priority(scn) if priority else scn

    def _make_plain(self, n, rng):
        bots = rng.sample(self.floor, n)
        orders = []
        for _ in range(n):
            item = rng.choice(list(ITEMS))
            orders.append({"item": item, "counter": ITEMS[item]["counter"],
                           "room": rng.choice(self.rooms)})
        cost = np.zeros((n, n))
        for b, bot in enumerate(bots):
            for o, order in enumerate(orders):
                pickup = self.named[order["counter"]]
                d = self.dist_from[pickup]
                cost[b, o] = d[bot] + d[self.named[order["room"]]]
        return {"bots": bots, "orders": orders, "cost": cost.tolist()}

    def _make_bundled(self, n, rng, max_room_gap, bundle_size=BUNDLE_SIZE, cross_counter=False):
        bots = rng.sample(self.floor, n)
        arrivals = []
        for _ in range(bundle_size * n):
            item = rng.choice(list(ITEMS))
            arrivals.append({"item": item, "counter": ITEMS[item]["counter"],
                             "room": rng.choice(self.rooms)})
        trips = self.bundle_orders(arrivals, max_room_gap, bundle_size, cross_counter)
        trips.sort(key=lambda t: min(t["orders"]))  # first come, first served
        chosen, later = trips[:n], trips[n:]
        cost = np.array([[self.trip_length(tuple(b), t) for t in chosen] for b in bots], dtype=float)
        return {"bots": bots, "orders": chosen, "cost": cost.tolist(), "arrivals": arrivals,
                "waiting": sorted(i for t in later for i in t["orders"])}


def add_priority(scenario):
    """Weight each order's column by its arrival priority (first = n ... last = 1).

    Multiplying (not adding) matters: a per-order term added to the cost is the
    same for every assignment, so it would break no ties.
    """
    dist = np.asarray(scenario["cost"], dtype=float)
    n = dist.shape[1]
    weights = np.arange(n, 0, -1, dtype=float)
    for order, w in zip(scenario["orders"], weights):
        order["priority"] = int(w)
    scenario["distance"] = dist.tolist()
    scenario["cost"] = (dist * weights).tolist()
    return scenario
