# 知识点卡片：PyTorch核心

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | PyTorch深度学习框架核心 |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 10小时 |

---

## 核心概念

### 1. Tensor操作

```python
import torch

# 创建
x = torch.randn(3, 4)
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
eye = torch.eye(3)
arange = torch.arange(0, 10, 2)

# 形状操作
x = torch.randn(2, 3, 4)
x.view(2, 12)        # reshape
x.unsqueeze(0)       # (1, 2, 3, 4)
x.squeeze()          # 移除维度为1的轴
x.permute(2, 0, 1)   # 转置
x.transpose(0, 1)    # 交换两个维度

# 索引
x[0, :, :]           # 取第一个batch
x[:, 1:, :]          # 切片

# GPU
x = x.cuda()         # 或 x.to('cuda')
x = x.cpu()
```

### 2. Autograd自动微分

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x ** 2
y.backward()
print(x.grad)  # 3x² + 4x = 12 + 8 = 20

# 梯度清零
optimizer.zero_grad()  # 或 model.zero_grad()

# 推理时禁用梯度
with torch.no_grad():
    output = model(input)
```

### 3. nn.Module

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

model = MyModel(784, 256, 10)
# 获取所有参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 4. DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CustomDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
```

### 5. 完整训练模板

```python
model = MyModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Acc={acc:.4f}")
```

### 6. 混合精度训练

```python
scaler = torch.cuda.amp.GradScaler()

for x, y in train_loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = criterion(model(x.cuda()), y.cuda())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 常用调试技巧

```python
# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")

# 检查NaN
assert not torch.isnan(loss), "Loss is NaN!"

# Profile
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    output = model(input)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 相关知识点

- → [HuggingFace](../HuggingFace/知识点-Transformers使用.md)
- → [BP反向传播](../../3-深度学习核心/BP反向传播/README.md)
