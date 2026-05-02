# BP反向传播模块

反向传播（Backpropagation）是神经网络训练的核心算法。

## 知识点卡片

| 属性 | 内容 |
|------|------|
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 面试必考 | 是 |
| 必须手推 | 是 |

---

## 核心原理

BP算法利用链式法则计算梯度：
1. **Forward Pass**：计算每层的输出，保存中间结果
2. **Backward Pass**：从后向前计算每个参数的梯度

---

## 手推BP算法

以最简单的MLP为例：输入 → Linear → Sigmoid → Output

### 符号定义

```
输入: x (d维)
第一层: W¹ (d×h), b¹ (h)
隐藏层: h = σ(xW¹ + b¹)
第二层: W² (h×o), b² (o)
输出层: y = softmax(hW² + b²)

Loss: L = CrossEntropy(y, y_true)
```

### Forward Pass

```python
# Forward Pass
z1 = x @ W1 + b1           # (batch, h)
a1 = sigmoid(z1)           # (batch, h)
z2 = a1 @ W2 + b2          # (batch, o)
a2 = softmax(z2)           # (batch, o)
loss = cross_entropy(a2, y_true)
```

### Backward Pass

**Step 1: 输出层梯度**

```
∂L/∂z2 = a2 - y_true      # (batch, o)
# 这是交叉熵+softmax的梯度，简洁形式

∂L/∂W2 = a1.T @ (∂L/∂z2)  # (h, o)
∂L/∂b2 = sum(∂L/∂z2)       # (o,)
∂L/∂a1 = (∂L/∂z2) @ W2.T   # (batch, h)
```

**Step 2: 隐藏层梯度**

```
∂L/∂z1 = ∂L/∂a1 * σ'(z1)   # (batch, h)
# σ'(z) = σ(z) * (1 - σ(z))

∂L/∂W1 = x.T @ (∂L/∂z1)    # (d, h)
∂L/∂b1 = sum(∂L/∂z1)      # (h,)
```

---

## PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# 使用PyTorch自动微分验证手算梯度
model = MLP(784, 256, 10)
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

logits = model(x)
loss = F.cross_entropy(logits, y)
loss.backward()

# 检查梯度
print(model.fc1.weight.grad.shape)  # (256, 784)
print(model.fc2.weight.grad.shape)  # (10, 256)
```

---

## 从零实现自动微分

```python
import numpy as np

class Tensor:
    def __init__(self, data, grad_fn=None):
        self.data = np.array(data)
        self.grad_fn = grad_fn
        self.grad = None

    def backward(self, grad=None):
        if self.grad_fn is None:
            return
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad
        if self.grad_fn:
            self.grad_fn.backward(grad)

class Operation:
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Add(Operation):
    def forward(self, a, b):
        return Tensor(a.data + b.data, grad_fn=self)

    def backward(self, grad):
        return grad, grad

class MatMul(Operation):
    def forward(self, a, b):
        self.a, self.b = a, b
        return Tensor(a.data @ b.data, grad_fn=self)

    def backward(self, grad):
        return grad @ self.b.T, self.a.T @ grad
```

---

## 梯度消失与梯度爆炸

### 问题原因

```
梯度消失：
- 原因：sigmoid/tanh的导数最大0.25/1，连乘后趋近0
- 影响：靠近输入的层几乎学不到东西

梯度爆炸：
- 原因：权重初始化过大，梯度连乘后指数增长
- 影响：训练不稳定，loss变NaN
```

### 解决方案

| 方案 | 针对问题 | 方法 |
|------|---------|------|
| 激活函数 | 梯度消失 | ReLU (导数=1) |
| 残差连接 | 梯度消失 | ResNet |
| 梯度裁剪 | 梯度爆炸 | clip_grad_norm_ |
| 权重初始化 | 两者 | Xavier/He初始化 |
| BatchNorm | 梯度消失/爆炸 | 归一化激活值 |
| LSTM/GRU | 梯度消失 | 门控机制 |

---

## 面试高频问题

### Q1: 推导BP算法的链式法则

**答**：BP利用链式法则计算每个参数的梯度。假设 loss = f(g(h(x)))，则：
```
∂loss/∂x = ∂loss/∂f · ∂f/∂g · ∂g/∂h · ∂h/∂x
```
计算时从后往前，每层只需要知道：
1. 本层的梯度（来自后一层）
2. 本层的输入和输出（forward时保存）
3. 本层激活函数的导数

---

### Q2: 为什么需要保存forward时的中间值？

**答**：
计算梯度时需要用到forward时的中间结果：
- z = Wx + b
- ∂z/∂W = x
如果重新计算x，会增加计算量且可能数值不稳定。

---

### Q3: Adam优化器如何修正梯度？

**答**：
```python
# m = β1*m + (1-β1)*g  # 动量估计（一阶矩）
# v = β2*v + (1-β2)*g² # 二阶矩估计
# m_hat = m / (1-β1^t)  # 偏差修正
# v_hat = v / (1-β2^t)  # 偏差修正
# θ = θ - lr * m_hat / (√v_hat + ε)
```
偏差修正解决初始化时m、v为0的问题，使得早期梯度估计更准确。
