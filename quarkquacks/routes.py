"""Collision-free goose routes, with bystanders and short-range sensing.

1. Plan: each goose gets a route goose -> counter -> room. Geese are planned one
   at a time in first-come-first-served order of their orders. Each planned
   route reserves (cell, time) pairs, and later geese plan around them
   (space-time BFS: every step a goose may move or wait).
2. Execute: bystanders wander the halls. Each goose only senses people within
   SENSE_RADIUS cells. A goose never enters a cell held by a person or another
   goose, never swaps cells head-on, and waits if a person is right next to its
   next cell. If it has waited PATIENCE steps, it re-plans around what it senses.

The safety rules in step 2 hold even when waits make the step-1 plans go stale.
"""

import random
from collections import deque

MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
SENSE_RADIUS = 2
PATIENCE = 2
MAX_TICKS = 400


def _walkable(grid, cell):
    r, c = cell
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "#"


def plan_path(grid, start, waypoints, t0=0, reserved=None, reserved_moves=None,
              blocked=frozenset(), max_t=MAX_TICKS):
    """Space-time BFS from `start` through `waypoints` in order.

    reserved:       {(cell, t)} cells other geese hold at time t
    reserved_moves: {(a, b, t)} another goose moves a -> b between t and t+1
    blocked:        cells treated as walls (e.g. sensed people)
    Returns the list of cells after `start`, one per time step, or None.
    """
    reserved = reserved or set()
    reserved_moves = reserved_moves or set()
    first_wp = 1 if waypoints and start == waypoints[0] else 0
    if first_wp == len(waypoints):
        return []
    frontier = {(start, first_wp)}
    parents = {(start, first_wp, t0): None}
    t = t0
    while frontier and t < max_t:
        nxt = set()
        for cell, wp in frontier:
            for dr, dc in MOVES:
                cand = (cell[0] + dr, cell[1] + dc)
                if not _walkable(grid, cand) or cand in blocked and cand != cell:
                    continue
                if (cand, t + 1) in reserved or (cand, cell, t) in reserved_moves:
                    continue
                new_wp = wp + 1 if cand == waypoints[wp] else wp
                key = (cand, new_wp, t + 1)
                if key in parents:
                    continue
                parents[key] = (cell, wp, t)
                if new_wp == len(waypoints):
                    path, node = [], key
                    while parents[node] is not None:
                        path.append(node[0])
                        node = parents[node]
                    return path[::-1]
                nxt.add((cand, new_wp))
        frontier, t = nxt, t + 1
    return None


def static_path(grid, start, waypoints, blocked=frozenset()):
    """Plain shortest path through waypoints, treating `blocked` cells as walls.

    Used for quick re-plans while executing (no time dimension). Returns the
    cells after `start`, or None if a waypoint can't be reached.
    """
    path, here = [], start
    for goal in waypoints:
        if here == goal:
            continue
        prev = {here: None}
        q = deque([here])
        while q and goal not in prev:
            cell = q.popleft()
            for dr, dc in MOVES[1:]:
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt not in prev and _walkable(grid, nxt) and (nxt not in blocked or nxt == goal):
                    prev[nxt] = cell
                    q.append(nxt)
        if goal not in prev:
            return None
        leg, node = [], goal
        while node != here:
            leg.append(node)
            node = prev[node]
        path += leg[::-1]
        here = goal
    return path


def goose_waypoints(campus, scenario, assignment):
    """{goose: [counter cell, room cell(s)]} for each assigned order or bundle."""
    out = {}
    for goose, order_idx in assignment.items():
        order = scenario["orders"][order_idx]
        rooms = order.get("rooms", [order["room"]])
        out[goose] = [campus.named[order["counter"]]] + [campus.named[r] for r in rooms]
    return out


def plan_routes(campus, scenario, assignment, waypoints=None):
    """Prioritized planning. Returns {goose: path} and the planning order.

    `waypoints` ({goose: [cells]}) overrides the default counter -> room trip,
    e.g. for carpool trips with several stops. `assignment` then maps each goose
    to the index of its earliest order, which sets first-come-first-served priority.
    """
    waypoints = waypoints or goose_waypoints(campus, scenario, assignment)
    order = sorted(assignment, key=lambda g: assignment[g])  # first-come, first-served
    reserved, reserved_moves, paths = set(), set(), {}
    for goose in order:
        start = tuple(scenario["bots"][goose])
        path = plan_path(campus.grid, start, waypoints[goose], 0, reserved, reserved_moves,
                         max_t=150)
        if path is None:  # no reservation-free route: take the plain route, safety rules still apply
            path = static_path(campus.grid, start, waypoints[goose]) or []
        paths[goose] = path
        prev = start
        reserved.add((start, 0))
        for t, cell in enumerate(path):
            reserved.add((cell, t + 1))
            reserved_moves.add((prev, cell, t))
            prev = cell
    return paths, order


def _near(a, b, radius):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= radius


def simulate(campus, scenario, assignment, n_people=3, seed=0, max_ticks=MAX_TICKS,
             waypoints=None):
    """Run the delivery tick by tick. Returns frames and stats.

    Each frame: {"t", "geese": {goose: cell or None once delivered},
                 "people": [cells], "waiting": {goose: reason},
                 "stops": {goose: number of waypoints reached so far}}

    stats["stop_times"][goose][k] is the step at which the goose reached its
    k-th waypoint (counter pickups and room drop-offs), used for wait times.
    """
    rng = random.Random(seed)
    grid = campus.grid
    paths, order = plan_routes(campus, scenario, assignment, waypoints)
    waypoints = waypoints or goose_waypoints(campus, scenario, assignment)
    pos = {g: tuple(scenario["bots"][g]) for g in order}
    remaining = {g: list(paths[g]) for g in order}
    next_wp = {g: 0 for g in order}
    waited = {g: 0 for g in order}
    done = set()
    stats = {"waits_for_goose": 0, "waits_for_person": 0, "replans": 0,
             "stop_times": {g: [] for g in order}, "paused": {g: 0 for g in order}}

    named = set(campus.named.values())
    counters = {cell for sym, cell in campus.named.items() if not sym.isdigit()}
    free = [c for c in campus.floor if c not in named and c not in pos.values()]
    people = rng.sample(free, min(n_people, len(free)))

    def snapshot(t, waiting):
        return {"t": t, "geese": {g: (None if g in done else pos[g]) for g in order},
                "people": list(people), "waiting": dict(waiting), "stops": dict(next_wp)}

    frames = [snapshot(0, {})]
    for t in range(max_ticks):
        if len(done) == len(order):
            break
        waiting, claimed, moved, claimed_by = {}, set(), {}, {}
        active = [g for g in order if g not in done]
        for i, g in enumerate(active):
            here = pos[g]
            if not remaining[g]:
                remaining[g] = static_path(grid, here, waypoints[g][next_wp[g]:]) or []
            want = remaining[g][0] if remaining[g] else here
            sensed = [p for p in people if _near(p, here, SENSE_RADIUS)]
            others_now = {pos[h] for h in active[i + 1:]}  # not yet moved this tick

            reason = None
            if want != here:
                if want in claimed or want in others_now or moved.get(want) == here:
                    reason = "goose"
                elif want in sensed:
                    reason = "person"
                elif waited[g] < PATIENCE and any(_near(p, want, 1) for p in sensed):
                    reason = "person"  # keep a one-cell buffer, unless stuck

            if reason is None:
                if want != here:
                    moved[here] = want
                pos[g] = want
                if remaining[g]:
                    remaining[g].pop(0)
                waited[g] = 0
            else:
                waiting[g] = reason
                stats["waits_for_goose" if reason == "goose" else "waits_for_person"] += 1
                stats["paused"][g] += 1
                waited[g] += 1
                blocker = claimed_by.get(want) or next(
                    (h for h in active if h != g and pos[h] == want), None)
                yields = (blocker is None or order.index(g) > order.index(blocker)
                          or waited[g] >= 3 * PATIENCE)  # stuck too long: everyone gives way
                if yields and waited[g] >= PATIENCE:  # only the later order re-routes
                    goals = waypoints[g][next_wp[g]:]
                    blocked = set(sensed) | {pos[h] for h in active if h != g}
                    new = static_path(grid, here, goals, blocked)
                    if new and new[0] != want:
                        remaining[g] = new
                        stats["replans"] += 1
                    elif waited[g] >= 2 * PATIENCE:
                        # Head-on in a narrow hall: step aside to a free cell, then continue.
                        taken = claimed | others_now | set(people) | set(moved.values())
                        spots = [(here[0] + dr, here[1] + dc) for dr, dc in MOVES[1:]]
                        spots = [c for c in spots if _walkable(grid, c) and c not in taken
                                 and moved.get(c) != here]
                        if spots:
                            step = rng.choice(spots)
                            moved[here] = step
                            pos[g] = step
                            remaining[g] = static_path(grid, step, goals) or []
                            waited[g] = 0
                            stats["sidesteps"] = stats.get("sidesteps", 0) + 1
            claimed.add(pos[g])
            claimed_by[pos[g]] = g

            wps = waypoints[g]
            while next_wp[g] < len(wps) and pos[g] == wps[next_wp[g]]:
                next_wp[g] += 1
                stats["stop_times"][g].append(t + 1)
            if next_wp[g] == len(wps):
                done.add(g)  # handed the food over and went into the room

        goose_cells = {pos[g] for g in order if g not in done}
        for k, p in enumerate(people):
            options = [(p[0] + dr, p[1] + dc) for dr, dc in MOVES]
            options = [c for c in options if _walkable(grid, c) and c not in counters
                       and c not in goose_cells and (c == p or c not in people)]
            people[k] = rng.choice(options) if options else p
        frames.append(snapshot(t + 1, waiting))

    stats["ticks"] = frames[-1]["t"]
    stats["all_delivered"] = len(done) == len(order)
    return frames, stats


def assignment_drops(scenario, assignment):
    """Drop-off stops for assignment mode: [{"order", "label", "goose", "stop"}].

    `stop` is the waypoint index of the drop-off (0 is the counter pickup). With
    bundling, each order inside a bundle gets its own entry, numbered by arrival.
    """
    drops = []
    for g, o in sorted(assignment.items(), key=lambda kv: kv[1]):
        trip = scenario["orders"][o]
        rooms = trip.get("rooms", [trip["room"]])
        items = trip.get("items", [trip["item"]])
        ids = trip.get("orders", [o])
        for k, room in enumerate(rooms):
            drops.append({"order": ids[k], "label": f"{items[k]} to room {room}", "goose": g, "stop": 1 + k})
    return sorted(drops, key=lambda d: d["order"])


def order_wait_times(stats, drops):
    """Wait per order: steps from the order being placed (step 0) to its drop-off.

    Returns drops with "wait" filled in (None if that order was never delivered).
    """
    out = []
    for d in drops:
        times = stats["stop_times"].get(d["goose"], [])
        out.append({**d, "wait": times[d["stop"]] if d["stop"] < len(times) else None})
    return out


def check_frames(frames):
    """Raise AssertionError if any two geese collide or swap, or a goose meets a person."""
    for prev, cur in zip(frames, frames[1:]):
        cells = [c for c in cur["geese"].values() if c is not None]
        assert len(cells) == len(set(cells)), f"geese share a cell at t={cur['t']}"
        assert not set(cells) & set(cur["people"]), f"goose on a person at t={cur['t']}"
        for g, a in prev["geese"].items():
            b = cur["geese"][g]
            if a is None or b is None or a == b:
                continue
            for h, c in prev["geese"].items():
                if h != g and c == b and cur["geese"][h] == a:
                    raise AssertionError(f"geese {g} and {h} swap at t={cur['t']}")
