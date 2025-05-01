import numpy as np
import csv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from rl.cube_env import CubeEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path="logs/rewards.csv", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and "r" in info["episode"]:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
            for i, r in enumerate(self.episode_rewards):
                writer.writerow([i + 1, r])
                
def make_env(scramble_length, seed=None):
    def _init():
        env = CubeEnv(scramble_length=scramble_length)
        if seed is not None:
            env.seed(seed)
        return env
    return _init

# Curriculum stages: gradually increase scramble length
curriculum = [
    {"scramble_length": 1, "timesteps": 750_000},
    {"scramble_length": 2, "timesteps": 1_000_000},
    {"scramble_length": 3, "timesteps": 1_250_000},
]

MODEL_PATH = "models/dqn_cube.zip"

def run_curriculum_training():
    model = None

    for i, stage in enumerate(curriculum):
        scramble_length = stage["scramble_length"]
        timesteps = stage["timesteps"]
        print(f"\n=== Stage {i+1}: scramble_length={scramble_length}, timesteps={timesteps} ===")

        env = VecMonitor(SubprocVecEnv([make_env(scramble_length, seed=j) for j in range(8)]))
        eval_env = VecMonitor(DummyVecEnv([make_env(scramble_length)]))

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/best_model_stage_{i+1}",
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False,
        )
        reward_logger = RewardLoggerCallback(log_path=f"logs/rewards_stage_{i+1}.csv")
        if model is None:
            model = DQN(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                tensorboard_log="./logs/tb/",
                learning_rate=1e-3,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=128,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_final_eps=0.05,
            )
        else:
            model.set_env(env)

        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=[eval_callback, reward_logger])

    model.save(MODEL_PATH)
    print(f"✅ Training complete. Final model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    run_curriculum_training()
