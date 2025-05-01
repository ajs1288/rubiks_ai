from gymnasium import Env, spaces
import numpy as np
from cube.cube import RubiksCube

ALL_MOVES = [
    "U", "U'", "U2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "B", "B'", "B2"
]

COLOR_MAP = {
    'white': 0,
    'yellow': 1,
    'red': 2,
    'orange': 3,
    'blue': 4,
    'green': 5
}

class CubeEnv(Env):
    def __init__(self, scramble_length=10, max_steps=100):
        super().__init__()
        self.scramble_length = scramble_length
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ALL_MOVES))
        self.observation_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.uint8)
        self.cube = None
        self.scramble_seq = []
        self.steps_taken = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.cube = RubiksCube()
        self.scramble_seq = self.cube.scramble(self.scramble_length)
        self.steps_taken = 0
        obs = self._get_obs()
        info = {}  # you can later include scramble sequence or step count here
        return self._get_obs(), {"scramble": self.scramble_seq}

    def step(self, action):
        move = ALL_MOVES[action]
        self.cube.apply_move(move)
        self.steps_taken += 1

        obs = self._get_obs()
        terminated = self.cube.is_solved()
        truncated = self.steps_taken >= self.max_steps
        distance = self.cube.distance_to_solved()
        if terminated:
            reward = 100 + (self.max_steps - self.steps_taken)  # early bonus
        else:
            reward = 1.0 - (distance / 54.0)
            
        info = {
            "distance_to_solved": distance,
            "scramble": self.scramble_seq,
            "steps": self.steps_taken
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        print(self.cube)

    def _get_obs(self):
        flat = []
        for face in ['U', 'D', 'F', 'B', 'L', 'R']:
            face_grid = self.cube.cube.get_face(face)
            for row in face_grid:
                for square in row:
                    color_char = square.colour.lower()
                    if color_char not in COLOR_MAP:
                        raise ValueError(f"Invalid facelet color: {color_char}")
                    flat.append(COLOR_MAP[color_char])
        return np.array(flat, dtype=np.uint8)

    def set_scramble(self, scramble):
        self.cube = RubiksCube()
        self.cube.apply_moves(scramble)
        self.steps_taken = 0
        self.scramble_seq = scramble
        return self._get_obs()

    def seed(self, seed=None):
        np.random.seed(seed)
