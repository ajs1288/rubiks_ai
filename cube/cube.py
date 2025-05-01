from pycuber import Cube
import random
import copy

class RubiksCube:
    def __init__(self):
        """
        Initializes a solved Rubik's Cube using PyCuber.
        """
        self.cube = Cube()
        self._scramble_seq = []

    def apply_move(self, move: str):
        """
        Applies a single move to the cube (e.g., "R", "U'", "F2").
        """
        self.cube(move)

    def apply_moves(self, moves: list):
        """
        Applies a sequence of moves to the cube.
        """
        for move in moves:
            self.apply_move(move)

    def is_solved(self) -> bool:
        """
        Returns True if the cube is solved.
        """
        return self.cube == Cube()

    def scramble(self, length=20) -> list:
        """
        Applies a random scramble to the cube and returns the scramble sequence.
        """
        moves = ["U", "U'", "U2", "D", "D'", "D2",
                 "L", "L'", "L2", "R", "R'", "R2",
                 "F", "F'", "F2", "B", "B'", "B2"]
        self._scramble_seq = [random.choice(moves) for _ in range(length)]
        self.apply_moves(self._scramble_seq)
        return self._scramble_seq

    def get_scramble_sequence(self):
        """
        Returns the last scramble sequence if available.
        """
        return self._scramble_seq

    def get_state(self) -> str:
        """
        Returns a string representation of the cube state.
        Useful for hashing and comparison.
        """
        return str(self.cube)

    def copy(self):
        """
        Returns a deep copy of the RubiksCube object.
        """
        new_cube = RubiksCube()
        new_cube.cube = self.cube.copy()
        return new_cube

    def __str__(self):
        """
        Returns a printable string representation of the cube.
        """
        return str(self.cube)

    def __eq__(self, other):
        return isinstance(other, RubiksCube) and self.get_state() == other.get_state()

    def __hash__(self):
        return hash(self.get_state())
    
    def reset(self):
        self.cube = Cube()
        
    def distance_to_solved(self):
        """
        Heuristic: number of facelets not matching the center of their face.
        """
        distance = 0
        for face in ['U', 'D', 'F', 'B', 'L', 'R']:
            center_color = self.cube.get_face(face)[1][1].colour
            for row in self.cube.get_face(face):
                for square in row:
                    if square.colour != center_color:
                        distance += 1
        return distance

