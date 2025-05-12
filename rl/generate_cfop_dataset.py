import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from cube.cube import RubiksCube
from solvers.human_solvers import cfop_solver
from rl.cube_env import COLOR_MAP, ALL_MOVES
import numpy as np
import contextlib
import io
from tqdm import tqdm


def cube_to_obs(cube):
    flat = []
    for face in ['U', 'D', 'F', 'B', 'L', 'R']:
        face_grid = cube.cube.get_face(face)
        for row in face_grid:
            for square in row:
                flat.append(COLOR_MAP[square.colour.lower()])
    return np.array(flat, dtype=np.uint8)

dataset = []

for _ in tqdm(range(10000), desc="Generating dataset"):
    cube = RubiksCube()
    scramble = cube.scramble(length=3)
    obs = cube_to_obs(cube)

    cube = RubiksCube()
    cube.apply_moves(scramble)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = cfop_solver(cube)

    for move in solution:
        action = ALL_MOVES.index(move)
        dataset.append((obs.copy(), action))
        cube.apply_move(move)
        obs = cube_to_obs(cube)

os.makedirs("imitation_data", exist_ok=True)
with open("imitation_data/cfop_expert.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("✅ Expert dataset saved.")
