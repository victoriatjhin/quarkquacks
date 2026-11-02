"""Campus map, shortest paths and random goose delivery scenarios.

Starter version for Person A to extend (map, demo paths, animation).
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
ITEMS = {
    "coffee": {"counter": "A", "priority": 3},      # hot drink: deliver fast
    "pizza": {"counter": "B", "priority": 2},       # hot food
    "bubble tea": {"counter": "C", "priority": 1},  # can wait a little
}


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

    def make_scenario(self, n, rng=None):
        """n bots and n orders. Cost = priority x (bot -> counter -> room)."""
        rng = rng or random.Random()
        bots = rng.sample(self.floor, n)
        orders = []
        for _ in range(n):
            item = rng.choice(list(ITEMS))
            orders.append({"item": item, "counter": ITEMS[item]["counter"],
                           "room": rng.choice(self.rooms), "priority": ITEMS[item]["priority"]})
        cost = np.zeros((n, n))
        for b, bot in enumerate(bots):
            for o, order in enumerate(orders):
                pickup = self.named[order["counter"]]
                d = self.dist_from[pickup]
                cost[b, o] = order["priority"] * (d[bot] + d[self.named[order["room"]]])
        return {"bots": bots, "orders": orders, "cost": cost.tolist()}
