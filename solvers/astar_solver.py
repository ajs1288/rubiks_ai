import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import heapq
from cube.cube import RubiksCube
from solvers.heuristics import misplaced_heuristic, weighted_heuristic
import itertools
counter = itertools.count()

ALL_MOVES = [
    "U", "U'", "U2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "B", "B'", "B2"
]

def astar_solver(initial_cube: RubiksCube, heuristic_func=misplaced_heuristic, max_depth=12):
    """
    Solves the cube using A* search with a provided heuristic.

    Args:
        initial_cube (RubiksCube): starting cube
        heuristic_func (function): function to estimate cost to goal
        max_depth (int): optional safety depth limit

    Returns:
        list: sequence of moves to solve the cube
    """
    visited = set()
    heap = []

    initial_state = initial_cube.get_state()
    h = heuristic_func(initial_cube)
    g = 0
    f = g + h
    heapq.heappush(heap, (f, next(counter), g, initial_cube.copy(), []))
    visited.add(initial_state)

    while heap:
        f, _, g, cube, path = heapq.heappop(heap)

        if cube.is_solved():
            return path

        if len(path) >= max_depth:
            continue

        for move in ALL_MOVES:
            next_cube = cube.copy()
            next_cube.apply_move(move)
            state = next_cube.get_state()

            if state in visited:
                continue

            visited.add(state)
            g_new = g + 1
            h_new = heuristic_func(next_cube)
            f_new = g_new + h_new
            heapq.heappush(heap, (f_new, next(counter), g_new, next_cube, path + [move]))

    return None

# Optional test
if __name__ == "__main__":
    from solvers.heuristics import misplaced_heuristic, weighted_heuristic
    cube = RubiksCube()
    scramble = cube.scramble(length=3)
    print("Scramble:", scramble)
    solution = astar_solver(cube, heuristic_func=weighted_heuristic)
    print("Solution:", solution)