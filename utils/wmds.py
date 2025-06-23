import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm  # 导入 tqdm

def get_embeddings(X, weights, n_components=2, epochs=1000, lr=1.0, step_size=30, gamma=0.1, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将输入转换为 PyTorch 张量
    X = torch.as_tensor(X, dtype=dtype, device=device)
    weights = torch.as_tensor(weights, dtype=dtype, device=device)

    # 定义一个简单的全连接神经网络
    class FCNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(32, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            return x

    model = FCNN(X.shape[1], n_components).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # 学习率调度器
    criterion = nn.MSELoss()

    # 计算加权距离矩阵
    distances_X = torch.cdist(X * weights, X * weights, p=2)

    # 训练模型
    model.train()
    progress_bar = tqdm(range(epochs), desc="Training Embeddings", unit="epoch")  # tqdm 进度条
    for epoch in progress_bar:
        optimizer.zero_grad()
        projection = model(X)
        loss = criterion(distances_X, torch.cdist(projection, projection, p=2))
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

        # 打印当前 epoch 的 loss
        progress_bar.set_postfix({"loss": loss.item()})

    # 返回嵌入结果
    return projection


def get_weights(X, embeddings, epochs=1000, lr=1.0, step_size=300, gamma=0.1, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将输入转换为 PyTorch 张量
    X = torch.as_tensor(X, dtype=dtype, device=device)
    embeddings = torch.as_tensor(embeddings, dtype=dtype, device=device)

    # model = FCNN(embeddings.shape[1], X.shape[1]).to(device)
    weights = torch.ones(X.shape[1], dtype=dtype, device=device, requires_grad=True)
    optimizer = optim.Adam([weights], lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # 学习率调度器
    criterion = nn.MSELoss()

    # 计算嵌入的距离矩阵
    distances_embeddings = torch.cdist(embeddings, embeddings, p=2)

    progress_bar = tqdm(range(epochs), desc="Training Weights", unit="epoch")  # tqdm 进度条
    for epoch in progress_bar:
        optimizer.zero_grad()
        diff = torch.cdist(X * weights, X * weights, p=2) - distances_embeddings
        loss = criterion(diff, torch.zeros_like(diff))
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

        # 打印当前 epoch 的 loss
        progress_bar.set_postfix({"loss": loss.item()})

    # 返回权重结果
    return weights
