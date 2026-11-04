"""Campus map, shortest paths and random goose delivery scenarios.

Optional pre-bundling (make_scenario(..., bundle=True)): before assignment,
orders from the same counter whose rooms are close together are grouped into
one trip, so one goose can carry two orders. Off by default, so the training
data and results are unchanged.
"""

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
BUNDLE_SIZE = 2        # a goose carries at most two bundled orders


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

    def bundle_orders(self, orders, max_room_gap=MAX_ROOM_GAP):
        """Group orders first-come-first-served into trips of at most BUNDLE_SIZE.

        A new order joins the earliest open trip from the same counter whose
        rooms are all within max_room_gap steps of its room; otherwise it starts
        a new trip. Returns trips in order of their earliest order.
        """
        bundles = []
        for idx, order in enumerate(orders):
            for b in bundles:
                if (len(b["orders"]) < BUNDLE_SIZE and b["counter"] == order["counter"]
                        and all(self.room_gap(r, order["room"]) <= max_room_gap for r in b["rooms"])):
                    b["orders"].append(idx)
                    b["rooms"].append(order["room"])
                    break
            else:
                bundles.append({"item": order["item"], "counter": order["counter"],
                                "rooms": [order["room"]], "orders": [idx]})
        for b in bundles:  # visit rooms in the order that walks least from the counter
            if len(b["rooms"]) == 2:
                c = self.named[b["counter"]]
                r1, r2 = b["rooms"]
                if self.dist_from[c][self.named[r2]] < self.dist_from[c][self.named[r1]]:
                    b["rooms"].reverse()
            b["room"] = b["rooms"][-1]
        return bundles

    def trip_length(self, start, order):
        """Steps for one goose: start -> counter -> every room of the order or bundle."""
        counter = self.named[order["counter"]]
        steps = self.dist_from[counter][start]
        here = counter
        for room in order.get("rooms", [order["room"]]):
            cell = self.named[room]
            steps += self.dist_from[here][cell] if here in self.dist_from else self._dist(here, cell)
            here = cell
        return steps

    def _dist(self, a, b):
        if a not in self._room_dist:
            self._room_dist[a] = bfs_distances(a, self.grid)
        return self._room_dist[a][b]

    def make_scenario(self, n, rng=None, bundle=False, max_room_gap=MAX_ROOM_GAP):
        """n geese and n orders. Cost = walking distance (goose -> counter -> room).

        bundle=True: 2n orders arrive, nearby same-counter orders are bundled, and
        the first n trips (first come, first served) are assigned to the n geese;
        later orders wait for the next round ("waiting" in the result).
        """
        rng = rng or random.Random()
        if bundle:
            return self._make_bundled(n, rng, max_room_gap)
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

    def _make_bundled(self, n, rng, max_room_gap):
        bots = rng.sample(self.floor, n)
        arrivals = []
        for _ in range(2 * n):
            item = rng.choice(list(ITEMS))
            arrivals.append({"item": item, "counter": ITEMS[item]["counter"],
                             "room": rng.choice(self.rooms)})
        trips = self.bundle_orders(arrivals, max_room_gap)
        chosen, later = trips[:n], trips[n:]
        for t in chosen:
            t["items"] = [arrivals[i]["item"] for i in t["orders"]]
        cost = np.array([[self.trip_length(tuple(b), t) for t in chosen] for b in bots], dtype=float)
        return {"bots": bots, "orders": chosen, "cost": cost.tolist(), "arrivals": arrivals,
                "waiting": sorted(i for t in later for i in t["orders"])}
