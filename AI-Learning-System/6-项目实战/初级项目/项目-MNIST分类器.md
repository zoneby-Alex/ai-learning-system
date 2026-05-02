# 项目：MNIST分类器

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐☆☆☆☆ |
| 预估时间 | 1-2天 |
| 学习目标 | 理解训练流程 |
| 前置知识 | PyTorch基础 |

---

## 项目目标

1. 理解神经网络训练完整流程
2. 掌握数据加载和预处理
3. 学会使用GPU训练
4. 实现训练可视化

---

## 技术栈

```python
PyTorch
torchvision
matplotlib
tqdm
```

---

## 代码实现

### 1. 数据加载

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # (0-255) -> (0-1)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值方差
])

# 加载数据
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

### 2. 模型定义（使用Conv2d，不使用Linear）

```python
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Conv2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # FC layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1: 28x28 -> 14x14
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Conv Block 2: 14x14 -> 7x7
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

### 3. 训练循环

```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    print(f'Epoch {epoch}: Loss={running_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    print(f'Test Accuracy: {100.*correct/total:.2f}%')
```

### 4. 完整训练脚本

```python
if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
        test()
```

---

## 训练曲线可视化

```python
import matplotlib.pyplot as plt

# 记录训练过程
train_losses = []
train_accs = []
test_accs = []

def train_and_record(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_losses.append(running_loss/len(train_loader))
    train_accs.append(100.*correct/total)

# 绘图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Test')
plt.title('Accuracy')
plt.legend()
plt.savefig('training_curve.png')
```

---

## 面试可能问到的问题

### Q1: 为什么用Conv2d而不是直接Flatten后用Linear？

**答**：
- Conv利用局部连接和权重共享，比Flatten+Linear参数量更少
- Conv能捕获图像的空间结构（边缘、纹理）
- 减少过拟合风险

### Q2: Max Pooling的作用？

**答**：
- 减少特征图尺寸，降低计算量
- 提供平移不变性
- 增大感受野

### Q3: Dropout的作用和原理？

**答**：
- 训练时随机丢弃神经元（断开的比例为p）
- 迫使网络不依赖某些特定神经元，提高泛化能力
- 推理时使用所有神经元（输出乘以1-p）

---

## 验收标准

- [ ] 测试集准确率 > 99%
- [ ] 能解释每层的作用
- [ ] 能回答上面的面试问题
- [ ] 绘制了训练曲线

---

## 进阶挑战

1. **不用任何Conv层**，仅用Linear实现，看准确率差异
2. **实现VGG风格的网络**
3. **添加数据增强**（旋转、缩放）
4. **使用不同的优化器**对比效果
