# 正则化模块

本模块包含深度学习中正则化技术的知识点。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| L1/L2正则化 | ★★★★☆ | P0 | ★★★★☆ |
| Dropout | ★★★★★ | P0 | ★★★★★ |
| BatchNorm | ★★★★★ | P0 | ★★★★★ |
| LayerNorm | ★★★★★ | P0 | ★★★★★ |
| 数据增强 | ★★★★☆ | P1 | ★★★☆☆ |

---

## 正则化方法对比

| 方法 | 原理 | 作用 |
|------|------|------|
| L2正则化 | 权重衰减 | 限制权重大小 |
| L1正则化 | 稀疏化 | 特征选择 |
| Dropout | 随机失活 | 防止过拟合 |
| BatchNorm | 归一化 | 内部协变量偏移 |
| LayerNorm | 归一化 | 稳定训练 |
| 数据增强 | 数据扩充 | 增加多样性 |

---

## L1/L2正则化

```python
# L2正则化（权重衰减）
# loss = task_loss + λ * Σ w²
# 梯度：∂L/∂w = ∂task_loss/∂w + 2λw

# PyTorch中
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# L1正则化
# loss = task_loss + λ * Σ |w|
# 产生稀疏解
```

---

## Dropout

### 原理

```python
# 训练时：随机将p比例的神经元置零
# 推理时：使用所有神经元，输出乘以(1-p)

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand_like(x) > self.p
            return x * mask / (1 - self.p)  # Inverted dropout
        else:
            return x
```

### Dropout变体

```python
# 1. Dropout (p=0.5)
# 2. Dropout2d / Spatial Dropout
# 3. DropPath / Stochastic Depth
# 4. MC Dropout (用于不确定性估计)
```

---

## BatchNorm

### 原理

```python
# BatchNorm: 在batch维度上归一化
# y = γ * (x - μ) / √(σ² + ε) + β

# 训练时：使用batch的均值和方差
# 推理时：使用训练时统计的moving mean/variance

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3))
            # 更新running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
```

---

## LayerNorm

### 原理

```python
# LayerNorm: 在feature维度上归一化
# 不依赖于batch

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta
```

---

## BatchNorm vs LayerNorm

| 方面 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | Batch | Feature |
| 依赖batch | 是 | 否 |
| 适用场景 | CV（固定batch） | NLP/动态batch |
| 序列模型 | RNN | Transformer |

```python
# Transformer中
# Pre-LN比Post-LN更稳定
# LN在残差之前：x = LayerNorm(x + Attention(x))
```

---

## 面试高频问题

### Q1: BatchNorm训练和推理的区别？

**答**：
- **训练**：使用当前batch的均值/方差，同时更新running mean/var
- **推理**：使用固定的running mean/var，不更新

---

### Q2: 为什么NLP用LayerNorm不用BatchNorm？

**答**：
1. NLP中batch大小经常变化
2. 序列长度不固定
3. BatchNorm依赖batch统计量，不稳定

---

### Q3: Dropout为什么防止过拟合？

**答**：
1. 强迫网络不依赖特定神经元
2. 近似模型集成
3. 减少神经元间的共适应性
