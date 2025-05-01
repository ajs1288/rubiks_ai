import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import deque
from cube.cube import RubiksCube

# Define all 18 standard face moves
ALL_MOVES = [
    "U", "U'", "U2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "B", "B'", "B2"
]

def bfs_solver(initial_cube: RubiksCube, max_depth: int = 10):
    """
    Solves the cube using Breadth-First Search. Returns the move sequence.
    Warning: BFS is only practical for very shallow scrambles.
    
    Args:
        initial_cube (RubiksCube): The starting cube state
        max_depth (int): Depth limit for BFS to avoid excessive memory usage

    Returns:
        list: List of moves to solve the cube, or None if not found
    """
    visited = set()
    queue = deque()

    queue.append((initial_cube.copy(), []))
    visited.add(initial_cube.get_state())

    while queue:
        cube, path = queue.popleft()

        if cube.is_solved():
            return path

        if len(path) >= max_depth:
            continue

        for move in ALL_MOVES:
            next_cube = cube.copy()
            next_cube.apply_move(move)
            state = next_cube.get_state()

            if state not in visited:
                visited.add(state)
                queue.append((next_cube, path + [move]))

    return None

def iterative_bfs(cube, max_total_depth=10):
    for depth in range(1, max_total_depth + 1):
        solution = bfs_solver(cube, max_depth=depth)
        if solution is not None:
            return solution
    return None

# Optional test
if __name__ == "__main__":
    cube = RubiksCube()
    scramble = cube.scramble(length=3)
    print("Scramble:", scramble)

    solution = bfs_solver(cube, max_depth=7)

    if solution:
        print("Solution found:", solution)
        cube.apply_moves(solution)
        print("Solved:", cube.is_solved())
        print("Moves to solve:", len(solution))
    else:
        print("No solution found within depth limit.")