import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
results = pd.read_csv("eval/results.csv")

# Filter successful solves only for relevant plots
success_only = results[results["solved"] == True]

# Group by solver and compute metrics
avg_moves_success = success_only.groupby("solver")["moves"].mean()
avg_time = results.groupby("solver")["time"].mean()
solve_rate = results.groupby("solver")["solved"].mean() * 100

# # Plot 1: Average Moves per Solver (only successful solves)
# plt.figure(figsize=(8, 5))
# avg_moves_success.sort_values().plot(kind="barh", title="Average Moves per Solver (Success Only)")
# plt.xlabel("Moves")
# plt.tight_layout()
# plt.savefig("eval/avg_moves.png")
# plt.show()

# # Plot 2: Average Time per Solver (includes failures)
# plt.figure(figsize=(8, 5))
# avg_time.sort_values().plot(kind="barh", color="orange", title="Average Solve Time per Solver")
# plt.xlabel("Time (seconds)")
# plt.tight_layout()
# plt.savefig("eval/avg_time.png")
# plt.show()

# Plot 3: Solve Success Rate per Solver
plt.figure(figsize=(8, 5))
solve_rate.sort_values().plot(kind="barh", color="green", title="Solve Success Rate per Solver")
plt.xlabel("Success Rate (%)")
plt.tight_layout()
plt.savefig("eval/solve_rate.png")
plt.show()

# # Plot 4: Moves per Scramble by Solver (all data)
# plt.figure(figsize=(10, 6))
# for solver in results["solver"].unique():
#     subset = results[results["solver"] == solver]
#     plt.plot(subset["scramble"], subset["moves"], label=solver)
# plt.title("Number of Moves per Scramble by Solver")
# plt.xlabel("Scramble #")
# plt.ylabel("Moves")
# plt.legend()
# plt.tight_layout()
# plt.savefig("eval/moves_per_scramble.png")
# plt.show()

# # Plot 5: RL Reward Progression
# plt.figure(figsize=(10, 6))
# for stage in [1, 2, 3]:
#     try:
#         df = pd.read_csv(f"logs/rewards_stage_{stage}.csv")
#         plt.plot(df["episode"], df["reward"], label=f"Stage {stage}")
#     except FileNotFoundError:
#         print(f"Warning: rewards_stage_{stage}.csv not found")

# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("RL Agent Reward per Episode (All Stages)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("eval/reward_progression.png")
# plt.show()

print("Plots saved to eval/ directory.")
