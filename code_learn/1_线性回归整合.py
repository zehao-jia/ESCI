import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

def create_dataset():
    # 生成数据集
    x = torch.linspace(1, 50, 50).reshape(-1, 1)
    y = 2.0 * x + 3.0 + torch.randn(x.size()) * 8.0
    return x, y

def train():
    x, y = create_dataset()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = nn.Linear(in_features = 1, out_features = 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 初始化训练参数
    num_epochs = 100

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final parameters: weight = {model.weight.item()}, bias = {model.bias.item()}')
if __name__ == "__main__":
    train()  