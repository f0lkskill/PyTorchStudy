import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()
# print(model)

print("fc1 权重:\n", model.fc1.weight)
print("fc2 权重:\n", model.fc2.weight)

x = torch.tensor([[1.0, 2.0]])
y = model(x)
print("输出:\n", y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
optimizer.step()
print("优化后 fc1 权重:\n", model.fc1.weight)
print("优化后 fc2 权重:\n", model.fc2.weight)