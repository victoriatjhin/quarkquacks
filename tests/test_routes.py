import random

from quarkquacks import Campus, solve_classical
from quarkquacks.routes import check_frames, plan_path, simulate


def _run(n, seed, people=3):
    campus = Campus()
    scn = campus.make_scenario(n, random.Random(seed))
    perm, _ = solve_classical(scn["cost"])
    assignment = {g: o for g, o in enumerate(perm)}
    return simulate(campus, scn, assignment, n_people=people, seed=seed)


def test_no_collisions_across_many_scenarios():
    """Geese never share or swap cells, and never step onto a person."""
    delivered = 0
    for seed in range(60):
        frames, stats = _run(random.Random(seed).choice([2, 3, 4]), seed)
        check_frames(frames)
        delivered += stats["all_delivered"]
    assert delivered == 60  # every run delivers every order


def test_plan_path_waits_for_reserved_cell():
    """A goose waits instead of entering a cell another goose holds."""
    grid = ["#####", "#...#", "#####"]
    path = plan_path(grid, (1, 1), [(1, 3)], reserved={((1, 2), 1)})
    assert path[0] == (1, 1) and path[-1] == (1, 3)
