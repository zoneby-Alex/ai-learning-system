# 激活函数模块

本模块包含深度学习中常用激活函数的知识点。

## 知识点列表

| 激活函数 | 掌握程度 | 优先级 | 面试频率 |
|---------|---------|--------|---------|
| Sigmoid | ★★★★☆ | P0 | ★★★★☆ |
| Tanh | ★★★☆☆ | P1 | ★★☆☆☆ |
| ReLU | ★★★★★ | P0 | ★★★★☆ |
| Leaky ReLU | ★★★☆☆ | P1 | ★★☆☆☆ |
| GELU | ★★★★☆ | P0 | ★★★★☆ |
| SiLU/Swish | ★★★☆☆ | P1 | ★★☆☆☆ |

---

## 激活函数对比

| 函数 | 公式 | 值域 | 优点 | 缺点 |
|------|------|------|------|------|
| Sigmoid | 1/(1+e^{-x}) | (0,1) | 概率输出 | 梯度消失/计算慢 |
| Tanh | (e^x-e^{-x})/(e^x+e^{-x}) | (-1,1) | 零中心 | 梯度消失 |
| ReLU | max(0,x) | [0,∞) | 计算快/不梯度消失 | Dying ReLU |
| Leaky ReLU | max(0.01x,x) | ℝ | 不 Dying | 不常用 |
| GELU | xΦ(x) | ℝ | Transformer标配 | 计算复杂 |
| SiLU | x·sigmoid(x) | ℝ | 自门控 | 计算复杂 |

---

## 数学公式

### Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 导数：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
# 在x=0处最大≈0.25，在x→±∞时→0
```

### ReLU

```python
def relu(x):
    return np.maximum(0, x)

# 导数：relu'(x) = 1 if x > 0 else 0
# 不涉及指数运算，计算快
# 但x<0时梯度为0（Dead ReLU问题）
```

### GELU

```python
def gelu(x):
    """Gaussian Error Linear Unit"""
    # 近似：0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# GELU ≈ x * Φ(x)，其中Φ是正态分布CDF
# 在Transformer中广泛使用（BERT/GPT等）
```

### SiLU (Swish)

```python
def silu(x):
    return x * sigmoid(x)

# SiLU(x) = x · sigmoid(x)
# 有自门控特性，效果好于ReLU
# Swish-β=1时等于SiLU
```

---

## PyTorch实现

```python
import torch.nn as nn

# 常用激活函数
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(0.01)
gelu = nn.GELU()
silu = nn.SiLU()  # 或 Swish
tanh = nn.Tanh()
sigmoid = nn.Sigmoid()

# 在网络中使用的例子
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## 面试高频问题

### Q1: 为什么ReLU比Sigmoid/Tanh更好？

**答**：
1. **计算效率**：ReLU只是max(0,x)，不涉及指数运算
2. **梯度消失**：Sigmoid/Tanh在两端梯度趋近于0，ReLU梯度要么0要么1
3. **收敛速度**：ReLU收敛更快
4. **生物学合理性**：更接近神经元的稀疏激活特性

---

### Q2: Dying ReLU问题及解决方案？

**答**：
**问题**：某些神经元输出恒为0，不再更新

**原因**：负输入梯度为0，权重无法更新

**解决方案**：
1. **Leaky ReLU**：让负区间也有小梯度
2. **PReLU**：让负斜率可学习
3. **ELU**：负区间用指数，输出接近0
4. **GELU**：概率角度平滑近似

---

### Q3: GELU为什么在Transformer中广泛使用？

**答**：
1. **形式上**：GELU ≈ x · Φ(x)，类似自门控
2. **效果上**：在BERT/GPT等Transformer中效果好
3. **平滑性**：比ReLU更平滑，梯度流动更好
4. **无上界**：不会像Sigmoid一样饱和

---

## 代码练习

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# 绘图对比
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, gelu(x), label='GELU')
plt.legend()
plt.grid(True)
plt.savefig('activation_functions.png')
```

---

## 相关知识点

- → [梯度消失/爆炸](../3-深度学习核心/BP反向传播/README.md) - 激活函数的影响
- → [Transformer](../Transformer/README.md) - GELU的应用
