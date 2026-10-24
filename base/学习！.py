import torch
import torch.nn as nn
import torch.optim as optim

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

optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

# 造 1000 组数据，任务是学加法
x = (torch.rand(1000, 2) - 0.5) * 10
answer = x.sum(dim=1, keepdim=True)

for epoch in range(500):
    y = model(x)
    loss = loss_fn(y, answer)

    optimizer.zero_grad()   # 清空上轮梯度
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"第 {epoch} 轮  误差 = {loss.item():.4f}")

# 循环结束后，验收一下
test = torch.tensor([[1.0, 2.0], [5.0, 5.0], [-3.0, 1.0], [2.0, 2.0]])
with torch.no_grad():          # 只预测，不需要算梯度
    pred = model(test)

print("网络预测:", pred.flatten().tolist())
print("正确答案:", test.sum(dim=1).tolist())