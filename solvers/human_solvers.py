from cube.cube import RubiksCube
from pycuber import Cube as PyCube, Formula
from pycuber.solver.cfop import CFOPSolver

import random

ALL_MOVES = [
    "U", "U'", "U2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "B", "B'", "B2"
]

# Utility: apply a sequence of moves to a RubiksCube object
def apply_alg(cube, alg):
    for move in alg:
        cube.apply_move(move)

# Phase 1: Build white cross (mocked with a simple known alg)
def solve_white_cross(cube):
    return ["F", "R", "D", "R'", "D'", "F'"]  # basic insertion pattern

# Phase 2: Insert white corners (simplified)
def solve_white_corners(cube):
    return ["R'", "D'", "R", "D"] * 4  # basic corner insertion loop

# Phase 3: Solve middle layer edges
def solve_middle_edges(cube):
    return ["U", "R", "U'", "R'", "U'", "F'", "U", "F"] * 2  # pseudo algorithm

# Phase 4: Make yellow cross (OLL part 1)
def solve_yellow_cross(cube):
    return ["F", "R", "U", "R'", "U'", "F'"]

# Phase 5: Orient yellow face (OLL part 2)
def orient_yellow_face(cube):
    return ["R", "U", "R'", "U", "R", "U2", "R'"]

# Phase 6: Permute last layer (PLL)
def permute_last_layer(cube):
    return ["R'", "U'", "R", "U'", "R'", "U2", "R"]

def beginner_solver(cube: RubiksCube):
    moves = []
    phases = [
        solve_white_cross,
        solve_white_corners,
        solve_middle_edges,
        solve_yellow_cross,
        orient_yellow_face,
        permute_last_layer
    ]

    for phase in phases:
        alg = phase(cube)
        apply_alg(cube, alg)
        moves.extend(alg)

def cfop_solver(cube_wrapper: RubiksCube):
    scramble = cube_wrapper.get_scramble_sequence()

    # Create fresh PyCuber cube and apply scramble
    pycube = PyCube()
    pycube(Formula(scramble))

    # Solve using PyCuber's CFOP solver
    solver = CFOPSolver(pycube)
    solution = solver.solve()

    moves = [str(move) for move in solution]
    cube_wrapper.cube = pycube.copy()
    return moves

# Full state-based Roux method phases (simplified representative logic)
def solve_first_block(pcube: PyCube):
    # In real implementation, detect pieces and solve intuitively or via search
    return Formula("L D L' U F' U' F")

def solve_second_block(pcube: PyCube):
    return Formula("R B R' U' B' U B")

def solve_cmll(pcube: PyCube):
    return Formula("R U R' U R U2 R'")

def solve_lse(pcube: PyCube):
    return Formula("R2 U R2 U2 R2 U R2")

def roux_solver(cube: RubiksCube):
    scramble = cube.get_scramble_sequence()
    pcube = PyCube()
    pcube(scramble)

    total_moves = Formula()

    for phase_func in [solve_first_block, solve_second_block, solve_cmll, solve_lse]:
        alg = phase_func(pcube)
        pcube(alg)
        apply_alg(cube, alg)
        total_moves += alg

    return [str(move) for move in total_moves]
