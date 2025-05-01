import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cube.cube import RubiksCube
from solvers.bfs_solver import bfs_solver
from solvers.astar_solver import astar_solver
from solvers.human_solvers import beginner_solver, cfop_solver, roux_solver
from rl.train_agent import MODEL_PATH
from stable_baselines3 import DQN
from rl.cube_env import ALL_MOVES
import time
import csv
import numpy as np

internal_solvers = {"CFOP", "Roux", "RL"}

def rl_solver(cube, model, max_steps=40):
    obs = cube_to_obs(cube)
    moves = []
    steps = 0
    while not cube.is_solved() and steps < max_steps:
        obs = np.array(obs)
        action, _ = model.predict(obs, deterministic=True)
        move = ALL_MOVES[action]
        cube.apply_move(move)
        moves.append(move)
        obs = cube_to_obs(cube)
        steps += 1
    return moves

def cube_to_obs(cube):
    from rl.cube_env import COLOR_MAP
    flat = []
    for face in ['U', 'D', 'F', 'B', 'L', 'R']:
        face_grid = cube.cube.get_face(face)
        for row in face_grid:
            for square in row:
                color_char = square.colour.lower()
                if color_char not in COLOR_MAP:
                    raise ValueError(f"Invalid facelet color: {color_char}")
                flat.append(COLOR_MAP[color_char])
    return np.array(flat, dtype=np.uint8)

def evaluate_all(num_scrambles=1, scramble_length=2):
    model = DQN.load(MODEL_PATH)
    results = []

    for i in range(num_scrambles):
        scramble_cube = RubiksCube()
        scramble_seq = scramble_cube.scramble(length=scramble_length)

        print(f"Scramble {i+1}: {scramble_seq}")
        solvers = [
            ("BFS", bfs_solver),
            ("A*", astar_solver),
            ("CFOP", cfop_solver),
            ("RL", lambda c: rl_solver(c, model, max_steps=100)),
            #("Beginner", beginner_solver),
            #("Roux", roux_solver),
        ]

        for name, solver in solvers:
            print(f"  ➤ Running {name} solver...")
            start = time.time()

            cube = RubiksCube()

            if name in internal_solvers:
                cube = scramble_cube.copy()  # use the same scrambled cube for CFOP and Roux
            else:
                cube = RubiksCube()
                cube.apply_moves(scramble_seq)

            try:
                moves = solver(cube)
                if name not in internal_solvers:
                    cube.apply_moves(moves)

                solved = cube.is_solved()
            except Exception as e:
                print(f"Solver {name} failed on scramble {i+1}: {e}")
                moves = []
                solved = False

            end = time.time()
            results.append({
                "scramble": i + 1,
                "solver": name,
                "moves": len(moves),
                "solved": solved,
                "time": round(end - start, 4),
            })
            print(f"    ➤ {name} completed in {round(end - start, 2)}s")

    with open("eval/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Evaluation complete. Results saved to eval/results.csv")

if __name__ == "__main__":
    evaluate_all()
