import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cube_env import ALL_MOVES
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
with open("imitation_data/cfop_expert.pkl", "rb") as f:
    dataset = pickle.load(f)

X = torch.tensor([x for x, _ in dataset], dtype=torch.float32)
y = torch.tensor([y for _, y in dataset], dtype=torch.long)

loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

# Simple MLP policy
class ImitationPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(54, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(ALL_MOVES))
        )


    def forward(self, x):
        return self.net(x)

policy = ImitationPolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    for xb, yb in loader:
        logits = policy(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct = (torch.argmax(logits, dim=1) == yb).sum().item()
        accuracy = correct / len(yb)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}, Accuracy: {accuracy:.2%}")

torch.save(policy.state_dict(), "models/imitation_policy.pth")
print("✅ Imitation model saved.")
