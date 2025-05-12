# Rubiks AI

A reinforcement learning project for solving the Rubik's Cube using a combination of traditional search algorithms, human-inspired solvers, imitation learning, and deep Q-learning.

This project was built for CS730/830 and supports evaluation and training of three types of agents:

* **Imitation Agent**: Learns from a CFOP solver via behavior cloning
* **RL Agent**: Learns from scratch via DQN and reward shaping
* **Hybrid Agent**: Starts from imitation weights and continues learning via RL

## 🛠 Setup

Requires Python 3.8+

### Install Dependencies

Run the following to install required Python packages:

```bash
pip install torch stable-baselines3 gymnasium numpy matplotlib pandas
```

Main libraries used:

* `stable-baselines3`
* `torch`
* `gymnasium`
* `matplotlib`
* `numpy`

---

## 🚀 Running the Code

### ⚠️ Quick Note:

All three agents (**Hybrid**, **Imitation**, and **Standard DQN**) have already been trained.
Their saved models are available in the `models/` directory, so **you do NOT need to retrain** them unless you want to test new settings or architectures.

---

### 🧠 1. Train the Imitation Learning Agent (Behavior Cloning)

This supervised agent learns by mimicking a CFOP expert.

```bash
python rl/train_imitation_agent.py
```

**Output:**

* Model: `models/imitation_policy.pth`
* Accuracy and loss logs printed to terminal

**To modify:**

* CFOP dataset: `generate_cfop_dataset.py`
* Epochs, architecture: `train_imitation_agent.py`

---

### 🤖 2. Train the Standard RL Agent (DQN only)

This agent trains from scratch using curriculum learning and reward shaping.

```bash
python rl/train_agent.py
```

**Output:**

* Model: `models/dqn_cube.zip`
* Rewards: `logs/rewards_stage_*.csv`
* TensorBoard: `logs/tb/`

**To modify:**

* Curriculum: `curriculum` list in `train_agent.py`
* Reward shaping: `CubeEnv.step()` in `rl/cube_env.py`
* Network: change MLP policy or architecture

---

### 🧪 3. Train the Hybrid RL Agent (Imitation + DQN)

Loads weights from the imitation model, then continues training via reinforcement learning.

```bash
python rl/train_hybrid_agent.py
```

**Output:**

* Model: `models/hybrid_dqn_cube.zip`
* Logs: `logs/hybrid_rewards_stage_*.csv`, `logs/tb/`

**To modify:**

* Network: `CustomMLPExtractor` in `train_hybrid_agent.py`
* Transfer logic: `load_imitation_into_dqn()`
* Curriculum or DQN hyperparameters

---

### 📊 4. Evaluate All Agents + Solvers

Runs each solver (BFS, A\*, RL, CFOP, Imitation, etc.) on generated scrambles.

```bash
python evaluator.py
```

**Output:**

* Results printed to console
* CSV: `eval/results.csv`

**To modify:**

* Active solvers: add/remove from `solvers = [...]` (Currently only CFOP, RL, and Hybrid are active)
* Scramble config: ex: `evaluate_all(num_scrambles=10, scramble_length=3)`

---

## ✅ Status

* [x] Fully working cube environment and solvers
* [x] Curriculum training for RL
* [x] Imitation learning from CFOP
* [x] Hybrid agent bootstrapped with behavior cloning
* [x] Automated evaluation and plotting scripts

---

## 👨‍🎓 Author

Developed by Anthony Santos for CS730 @ University of New Hampshire
Feel free to fork, improve, and experiment!
