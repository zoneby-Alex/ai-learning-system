# 手撕代码：ResNet Block & BatchNorm

## 面试要求

- 难度：★★★☆☆
- 能在5分钟内手写
- 理解训练/推理模式区别

---

## ResNet Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut（维度不匹配时用1x1卷积调整）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # 残差连接：F(x) + x
        out = F.relu(out)

        return out
```

---

## BatchNorm手写

```python
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 统计量（不参与梯度）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x: (B, C, H, W)
        if self.training:
            # 在 (B, H, W) 维度上计算
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            # 更新全局统计量（指数移动平均）
            self.running_mean = self.running_mean * (1-self.momentum) + mean.squeeze() * self.momentum
            self.running_var = self.running_var * (1-self.momentum) + var.squeeze() * self.momentum
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        # 归一化 + 缩放
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
```

---

## 面试要点

### ResNet核心
- 残差连接：H(x) = F(x) + x
- 当F(x)→0时，H(x)≈x（恒等映射）
- 缓解梯度消失，使深层网络可训练
- Shortcut：1x1卷积处理维度不匹配

### BatchNorm核心
- 训练：用batch统计量 + 更新running stats
- 推理：用固定的running stats
- γ/β：恢复表达能力
- momentum：控制running stats更新速度

---

## 验证代码

```python
# 验证ResNet Block
block = ResNetBlock(64, 128, stride=2)
x = torch.randn(4, 64, 32, 32)
out = block(x)
print(f"Input: {x.shape}, Output: {out.shape}")  # (4,128,16,16)

# 验证BatchNorm
bn = BatchNorm2d(64)
bn.train()
out_train = bn(x)
bn.eval()
out_eval = bn(x)
print(f"Train/Test diff: {(out_train-out_eval).abs().mean():.6f}")
```
