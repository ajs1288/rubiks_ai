import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
results = pd.read_csv("eval/results.csv")

# Group by solver and compute averages
avg_moves = results.groupby("solver")["moves"].mean()
avg_time = results.groupby("solver")["time"].mean()
solve_rate = results.groupby("solver")["solved"].mean() * 100

# Plot 1: Average Moves per Solver
plt.figure(figsize=(8, 5))
avg_moves.sort_values().plot(kind="barh", title="Average Number of Moves per Solver")
plt.xlabel("Moves")
plt.tight_layout()
plt.savefig("eval/avg_moves.png")
plt.show()

# Plot 2: Average Time per Solver
plt.figure(figsize=(8, 5))
avg_time.sort_values().plot(kind="barh", color="orange", title="Average Solve Time per Solver")
plt.xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig("eval/avg_time.png")
plt.show()

# Plot 3: Solve Success Rate per Solver
plt.figure(figsize=(8, 5))
solve_rate.sort_values().plot(kind="barh", color="green", title="Solve Success Rate per Solver")
plt.xlabel("Success Rate (%)")
plt.tight_layout()
plt.savefig("eval/solve_rate.png")
plt.show()

# Plot 4: Moves per Scramble by Solver (line plot)
plt.figure(figsize=(10, 6))
for solver in results["solver"].unique():
    subset = results[results["solver"] == solver]
    plt.plot(subset["scramble"], subset["moves"], label=solver)
plt.title("Number of Moves per Scramble by Solver")
plt.xlabel("Scramble #")
plt.ylabel("Moves")
plt.legend()
plt.tight_layout()
plt.savefig("eval/moves_per_scramble.png")
plt.show()


df = pd.read_csv("logs/rewards.csv")
plt.plot(df["episode"], df["reward"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("RL Agent Reward per Episode")
plt.grid(True)
plt.show()

print("Plots saved to eval/ directory.")
