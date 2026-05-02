# PyTorch模块

本模块介绍PyTorch深度学习框架。

## 核心概念

```
PyTorch = Tensor + Autograd + nn.Module + Optim

核心组件：
├── Tensor: 多维数组，支持GPU
├── Autograd: 自动微分
├── nn.Module: 神经网络模块
├── Optim: 优化器
└── DataLoader: 数据加载
```

---

## 基础操作

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 创建tensor
x = torch.randn(3, 4)                    # 随机
x = torch.zeros(3, 4)                     # 零
x = torch.ones(3, 4)                      # 一
x = torch.tensor([[1, 2], [3, 4]])       # 从数据

# GPU
x = x.cuda()                              # 移到GPU
x = x.cpu()                               # 移回CPU

# 运算
y = x + 2                                 # 加
y = x @ x.T                               # 矩阵乘法
y = F.softmax(x, dim=-1)                 # softmax
y = F.relu(x)                             # relu
```

---

## nn.Module

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Net()
output = model(input)
```

---

## 数据加载

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch_x, batch_y in loader:
    # 训练
    pass
```

---

## 训练流程

```python
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()           # 清梯度
        output = model(batch_x)         # 前向
        loss = criterion(output, batch_y)  # 损失
        loss.backward()                 # 反向
        optimizer.step()                # 更新
```

---

## GPU训练

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    # 训练
    pass
```

---

## 面试高频问题

### Q: torch.Tensor和torch.nn.Parameter的区别？

**答**：
- Tensor是需要训练的参数，但不自动注册到模块
- Parameter是Tensor的子类，会自动注册到模块的parameters()

---

## 相关知识点

- → [HuggingFace](../HuggingFace/README.md)
- → [BP反向传播](../3-深度学习核心/BP反向传播/README.md)
