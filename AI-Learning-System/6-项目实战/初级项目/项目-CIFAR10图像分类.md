# 项目：CIFAR-10 图像分类

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐☆☆☆ |
| 预估时间 | 1-2天 |
| 学习目标 | CNN实践 + 数据增强 |
| 前置知识 | PyTorch基础 |
| 技术栈 | ResNet18 + 数据增强 + CosineAnnealingLR |

---

## 项目目标

1. 掌握CNN图像分类完整流程
2. 实践数据增强策略
3. 使用迁移学习（预训练ResNet）
4. 实现学习率调度

---

## 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# ResNet18 + CIFAR-10适配
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)  # 修改最后一层

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 训练
for epoch in range(50):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # 评估
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(1) == target).sum().item()
    print(f"Epoch {epoch+1}: Accuracy={100.*correct/len(test_dataset):.2f}%")
```

---

## 验收标准

- [ ] 测试集准确率 > 95%
- [ ] 使用了至少3种数据增强
- [ ] 实现了CosineAnnealingLR
- [ ] 能解释ResNet迁移学习

---

## 面试Q&A

### Q: 为什么使用预训练ResNet？
**答**：ImageNet预训练的特征提取能力可迁移到CIFAR-10，加速收敛并提升准确率。这是迁移学习的核心思想。

---

## 进阶挑战

1. 从零训练ResNet（不用预训练），对比效果
2. 尝试Cutout/RandomErasing增强
3. 使用MixUp/CutMix
4. 对比不同模型（ResNet18 vs VGG vs MobileNet）
