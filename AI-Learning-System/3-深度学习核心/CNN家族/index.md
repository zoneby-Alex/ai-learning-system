# CNN家族模块

本模块包含卷积神经网络（CNN）的核心知识点。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| 卷积运算 | ★★★★★ | P0 | ★★★★☆ |
| 池化 | ★★★★☆ | P0 | ★★★☆☆ |
| LeNet/AlexNet | ★★★☆☆ | P1 | ★★☆☆☆ |
| VGG/GoogLeNet | ★★★☆☆ | P1 | ★★☆☆☆ |
| ResNet | ★★★★★ | P0 | ★★★★★ |
| EfficientNet | ★★★☆☆ | P2 | ★★☆☆☆ |

---

## 卷积运算

### 原理

```python
# 输入：(B, C_in, H, W)
# 卷积核：(C_out, C_in, kH, kW)
# 输出：(B, C_out, H_out, W_out)

# 其中：
# H_out = (H - kH + 2*padding) / stride + 1
# W_out = (W - kW + 2*padding) / stride + 1
```

### PyTorch实现

```python
import torch
import torch.nn as nn

# 2D卷积
conv = nn.Conv2d(
    in_channels=3,      # 输入通道
    out_channels=64,    # 输出通道
    kernel_size=3,      # 或 (3, 3)
    stride=1,           # 或 (1, 1)
    padding=1,          # 或 'same'
    bias=False          # 通常设为False，配合BatchNorm
)

x = torch.randn(1, 3, 224, 224)
out = conv(x)  # (1, 64, 224, 224)

# 深度可分离卷积（MobileNet）
depthwise = nn.Conv2d(3, 3, 3, groups=3)  # 每个通道单独卷积
pointwise = nn.Conv2d(3, 64, 1)            # 1x1卷积融合
```

---

## 池化操作

```python
# 最大池化
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化
pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 全局池化（常用于分类头）
gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出 (B, C, 1, 1)
gap = nn.AdaptiveMaxPool2d((1, 1))
```

---

## CNN架构演化

```
LeNet(1998) → AlexNet(2012) → VGG(2014) → GoogLeNet(2014) → ResNet(2015) → DenseNet → EfficientNet
   5层         8层          16-19层      22层            152层          121层       移动端
```

### 经典架构对比

| 架构 | 年份 | 创新点 | 参数量 |
|------|------|--------|--------|
| LeNet | 1998 | 卷积+池化+FC | ~60K |
| AlexNet | 2012 | ReLU/Dropout/GPU | 60M |
| VGG | 2014 | 3x3卷积堆叠 | 133M |
| GoogLeNet | 2014 | Inception模块 | 5M |
| ResNet | 2015 | 残差连接 | 25-60M |
| DenseNet | 2017 | 密集连接 | 20M |
| EfficientNet | 2019 | 复合缩放 | 5-30M |

---

## ResNet

### 核心：残差连接

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        return F.relu(out)
```

### 为什么残差有效？

```
普通网络：深层次的网络容易退化（degradation）
残差网络：恒等映射更容易学习

数学上：
- 普通：H(x) = F(x)
- 残差：H(x) = F(x) + x

当F(x)=0时，H(x)=x（恒等映射）
这比学习H(x)=x更容易
```

---

## 面试高频问题

### Q1: 卷积和全连接的区别？

**答**：
| 方面 | 全连接 | 卷积 |
|------|--------|------|
| 连接方式 | 所有神经元相连 | 局部连接 |
| 参数 | O(H×W) | O(k²×C) |
| 权重共享 | 无 | 有 |
| 平移不变性 | 无 | 有 |
| 适用数据 | 固定尺寸 | 任意尺寸 |

---

### Q2: 为什么用3x3卷积？

**答**：
- 3x3是最小的能捕获上下左右方向的卷积核
- 两个3x3堆叠的感受野 = 5x5
- 三个3x3堆叠的感受野 = 7x7
- 参数量更少（2层3x3 vs 1层5x5：18 vs 25）

---

### Q3: 残差连接的作用？

**答**：
1. **缓解梯度消失**：提供梯度的直接通道
2. **缓解退化问题**：使恒等映射更容易学习
3. **提升训练稳定性**：使深层网络的训练更稳定

---

## 相关知识点

- → [ResNet论文](../9-学习资料/论文/README.md) - 原始论文
- → [BatchNorm](../正则化/README.md) - 归一化层
- → [DenseNet](../CNN家族/README.md) - 密集连接
