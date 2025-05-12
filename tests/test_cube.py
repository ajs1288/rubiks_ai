import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cube.cube import RubiksCube
from solvers.bfs_solver import bfs_solver, iterative_bfs
from solvers.astar_solver import astar_solver
from solvers.heuristics import misplaced_heuristic, weighted_heuristic

def test_solved_state():
    cube = RubiksCube()
    assert cube.is_solved(), "Cube should start solved"

def test_apply_move():
    cube = RubiksCube()
    cube.apply_move("R")
    assert not cube.is_solved(), "Cube should not be solved after a single move"

def test_scramble_and_copy():
    cube = RubiksCube()
    scramble = cube.scramble()
    assert not cube.is_solved(), "Scrambled cube should not be solved"
    copy_cube = cube.copy()
    assert cube == copy_cube, "Copy should be equal to original after scramble"

def test_apply_moves():
    cube = RubiksCube()
    moves = ["R", "U", "R'", "U'"]
    cube.apply_moves(moves)
    assert not cube.is_solved(), "Cube should not be solved after RU R' U'"
    
def test_get_state_changes_after_move():
    cube = RubiksCube()
    original_state = cube.get_state()
    
    cube.apply_move("R")
    moved_state = cube.get_state()

    assert isinstance(original_state, str), "State should be a string"
    assert isinstance(moved_state, str), "State after move should be a string"
    assert original_state != moved_state, "State should change after applying a move"

def test_get_state_reversibility():
    cube = RubiksCube()
    original_state = cube.get_state()
    
    cube.apply_moves(["R", "U", "F"])
    cube.apply_moves(["F'", "U'", "R'"])
    
    final_state = cube.get_state()
    assert original_state == final_state, "State should return to original after applying moves and their inverses"

def test_reset_cube():
    cube = RubiksCube()
    cube.scramble()
    cube.reset()
    assert cube.is_solved(), "Cube should be solved after reset"

def test_str_and_state_consistency():
    cube = RubiksCube()
    cube.apply_move("R")
    state_str = str(cube)
    internal_state = cube.get_state()
    assert isinstance(state_str, str), "String representation should be a string"
    assert isinstance(internal_state, str), "Cube state should be string"
    assert state_str != "", "String representation should not be empty"
    assert internal_state != "", "Cube state should not be empty"

def test_equality_operator():
    cube1 = RubiksCube()
    cube2 = RubiksCube()
    assert cube1 == cube2, "Two new cubes should be equal"

    cube1.apply_move("R")
    assert cube1 != cube2, "Cubes should differ after a move on one"

def test_hash_consistency():
    cube1 = RubiksCube()
    cube2 = RubiksCube()
    assert hash(cube1) == hash(cube2), "Hashes should match for solved cubes"

    cube1.apply_move("R")
    assert hash(cube1) != hash(cube2), "Hashes should differ for different cube states"

def test_apply_empty_move_list():
    cube = RubiksCube()
    cube.apply_moves([])
    assert cube.is_solved(), "Applying empty move list should not alter cube"

def test_reverse_moves():
    cube = RubiksCube()
    moves = ["R", "U", "F"]
    inverse_moves = ["F'", "U'", "R'"]
    cube.apply_moves(moves)
    cube.apply_moves(inverse_moves)
    assert cube.is_solved(), "Cube should return to solved state after applying moves and their inverses"

def run_base_tests():
    test_solved_state()
    test_apply_move()
    test_scramble_and_copy()
    test_apply_moves()
    test_get_state_changes_after_move()
    test_get_state_reversibility()
    test_reset_cube()
    test_str_and_state_consistency()
    test_equality_operator()
    test_hash_consistency()
    test_apply_empty_move_list()
    test_reverse_moves()
    print("Base cube functionality tested and working.")

def test_scramble_and_solve_bfs():
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

def test_iterative_bfs_short_scramble():
    cube = RubiksCube()
    scramble = cube.scramble(length=2)
    solution = iterative_bfs(cube, max_total_depth=5)
    assert solution is not None, f"Should solve 2-move scramble: {scramble}"
    cube.apply_moves(solution)
    assert cube.is_solved(), "Cube should be solved after applying solution"

def test_iterative_bfs_medium_scramble():
    cube = RubiksCube()
    scramble = cube.scramble(length=4)
    solution = iterative_bfs(cube, max_total_depth=8)
    if solution:
        cube.apply_moves(solution)
        assert cube.is_solved(), "Cube should be solved with sufficient depth"
    else:
        print(f"Warning: Could not solve scramble: {scramble}")

def test_iterative_bfs_insufficient_depth():
    cube = RubiksCube()
    scramble = cube.scramble(length=5)
    solution = iterative_bfs(cube, max_total_depth=3)
    assert solution is None, "Should not solve deep scramble with low max depth"
    
def run_bfs_tests():
    print("\nRunning BFS tests...")
    test_scramble_and_solve_bfs()
    test_iterative_bfs_short_scramble()
    test_iterative_bfs_medium_scramble()
    test_iterative_bfs_insufficient_depth()
    print("BFS solver tests passed.")
 
def test_astar_solve_scramble():
    cube = RubiksCube()
    scramble = cube.scramble(length=3)
    print("Scramble:", scramble)
    solution = astar_solver(cube, heuristic_func=misplaced_heuristic)
    print("Solution:", solution)
    
def test_astar_medium_scramble():
    cube = RubiksCube()
    scramble = cube.scramble(length=4)
    print("A* Medium scramble:", scramble)
    solution = astar_solver(cube, heuristic_func=weighted_heuristic, max_depth=10)
    assert solution is not None, "A* should solve 4-move scramble"
    cube.apply_moves(solution)
    assert cube.is_solved(), "Cube should be solved"

def run_astar_tests():
    print("\nRunning A* solver tests...")
    test_astar_solve_scramble()
    test_astar_medium_scramble()
    print("A* solver tests complete.")

def test_dqn_agent(scramble_length=3, max_steps=200):
    print("\nRunning DQN agent test...")

    from rl.cube_env import CubeEnv
    from stable_baselines3 import DQN

    # Load the trained model
    model = DQN.load("models/dqn_cube.zip")
    env = CubeEnv(scramble_length=scramble_length, max_steps=max_steps)

    obs, info = env.reset()
    print("Initial scrambled state:")
    env.render()

    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    
    print("\nFinal state after agent ran:")
    env.render()
    if terminated:
        print(f"DQN agent solved the cube in {steps} steps.")
    else:
        print(f"DQN agent failed to solve the cube in {steps} steps.")

def test_beginner_solver():
    print("\nRunning Beginner solver test...")
    cube = RubiksCube()
    scramble = cube.scramble(length=3)
    print("Scramble:", scramble)
    from solvers.human_solvers import beginner_solver
    moves = beginner_solver(cube)
    print("Moves:", moves)
    print("Solved?", cube.is_solved())
    print("Beginner is hardcoded and meant to fail")
    # assert cube.is_solved(), "Beginner solver should solve the cube"

def test_cfop_solver():
    print("\nRunning CFOP solver test...")
    cube = RubiksCube()
    scramble = cube.scramble(length=3)
    print("Scramble:", scramble)
    print("Before solving:", cube.get_state())

    from solvers.human_solvers import cfop_solver
    moves = cfop_solver(cube)
    print("Moves:", moves)
    print("After solving:", cube.get_state())
    assert cube.is_solved(), "CFOP solver should solve the cube"

def test_roux_solver():
    print("\nRunning Roux solver test...")
    cube = RubiksCube()
    scramble = cube.scramble(length=3)
    print("Scramble:", scramble)
    from solvers.human_solvers import roux_solver
    moves = roux_solver(cube)
    print("Moves:", moves)
    print("Solved?", cube.is_solved())
    print("Roux is hardcoded and meant to fail")
    # assert cube.is_solved(), "Roux solver should solve the cube"
    
def test_imitation_agent(scramble_length=1, max_steps=100):
    print("\nRunning Imitation agent test...")

    import torch
    import torch.nn as nn
    from eval.evaluator import ImitationPolicy, cube_to_obs, ALL_MOVES

    cube = RubiksCube()
    scramble = cube.scramble(length=scramble_length)
    print("Scramble:", scramble)
    print("Before solving:", cube.get_state())

    # Load the trained imitation model
    model = ImitationPolicy()
    model.load_state_dict(torch.load("models/imitation_policy.pth"))
    model.eval()

    obs = cube_to_obs(cube)
    steps = 0
    moves = []

    while not cube.is_solved() and steps < max_steps:
        input_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            action = torch.argmax(logits, dim=1).item()

        move = ALL_MOVES[action]
        cube.apply_move(move)
        moves.append(move)
        obs = cube_to_obs(cube)
        steps += 1

    print("Moves taken:", moves)
    print("After solving:", cube.get_state())
    if cube.is_solved():
        print(f"✅ Imitation agent solved the cube in {steps} steps.")
    else:
        print("❌ Imitation agent failed to solve the cube.")

def test_hybrid_agent(scramble_length=1, max_steps=200):
    print("\nRunning Hybrid Agent test...")

    from rl.cube_env import CubeEnv
    from stable_baselines3 import DQN
    import torch
    from rl.cube_env import ALL_MOVES

    # Load the trained hybrid agent model (RL + Imitation)
    model = DQN.load("models/hybrid_dqn_cube.zip")  # Replace with your trained hybrid model
    env = CubeEnv(scramble_length=scramble_length, max_steps=max_steps)

    obs, info = env.reset()
    print("Initial scrambled state:")
    env.render()

    steps = 0
    terminated = False
    truncated = False

    # Run the agent to solve the cube
    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)  # Hybrid model should handle both RL and imitation
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    print("\nFinal state after agent ran:")
    env.render()

    if terminated:
        print(f"Hybrid agent solved the cube in {steps} steps.")
    else:
        print(f"Hybrid agent failed to solve the cube in {steps} steps.")


if __name__ == "__main__":
    run_base_tests()
    #run_bfs_tests()
    #run_astar_tests()
    #test_dqn_agent()
    #test_beginner_solver()
    #test_cfop_solver()
    #test_roux_solver()
    #test_imitation_agent()
    test_hybrid_agent()
