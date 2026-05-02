# 最优化方法模块

本模块包含深度学习所需的最优化理论知识。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| 梯度下降 | ★★★★★ | P0 | ★★★★★ |
| SGD/Momentum | ★★★★☆ | P0 | ★★★★☆ |
| Adam/RMSprop | ★★★★★ | P0 | ★★★★☆ |
| 学习率调度 | ★★★★☆ | P1 | ★★★★☆ |
| 梯度裁剪 | ★★★☆☆ | P1 | ★★★☆☆ |
| 拉格朗日乘子 | ★★★☆☆ | P2 | ★★☆☆☆ |

---

## 优化器对比

| 优化器 | 公式 | 优点 | 缺点 | 适用场景 |
|--------|------|------|------|---------|
| SGD | θ = θ - lr·∇θ | 简单稳定 | 收敛慢 | 经典场景 |
| Momentum | v = βv + (1-β)∇θ | 加速收敛 | 需调参 | 一般场景 |
| Adagrad | θ = θ - lr·∇θ/√(G+ε) | 自适应学习率 | 后期学习率小 | 稀疏特征 |
| RMSprop | θ = θ - lr·∇θ/√(E[g²]+ε) | 自适应学习率 | 需调参 | RNN/NLP |
| Adam | m = β1m + (1-β1)g; v = β2v + (1-β2)g² | 快收敛/易调参 | 可能不收敛 | 通用 |

---

## 核心公式

### 1. 梯度下降

```python
# θ_{t+1} = θ_t - lr * ∇θ L(θ_t)
# lr: 学习率
# ∇θ L: 损失函数梯度
```

### 2. Momentum

```python
# v_t = β * v_{t-1} + (1-β) * ∇θ L
# θ_{t+1} = θ_t - lr * v_t
# 物理意义：引入动量，类似物理中的惯性
```

### 3. Adam

```python
# m_t = β1 * m_{t-1} + (1-β1) * g_t        # 一阶矩估计（动量）
# v_t = β2 * v_{t-1} + (1-β2) * g_t²       # 二阶矩估计
# m_hat = m_t / (1 - β1^t)                   # 偏差修正
# v_hat = v_t / (1 - β2^t)                   # 偏差修正
# θ_{t+1} = θ_t - lr * m_hat / (√v_hat + ε) # 更新
```

---

## PyTorch实现

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (带权重衰减的Adam)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 学习率调度
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=10)
```

---

## 学习率调度

```python
# 1. Step Decay
# 每隔固定epoch降低学习率
lr = lr * gamma

# 2. Cosine Annealing
# 余弦曲线下降
lr = min_lr + (max_lr - min_lr) * (1 + cos(π * t / T)) / 2

# 3. Warmup + Cosine
# 先warmup预热，再cosine下降
if t < warmup_steps:
    lr = max_lr * t / warmup_steps
else:
    lr = max_lr * (1 - (t - warmup_steps) / (T - warmup_steps)) ** k
```

---

## 面试高频问题

### Q1: Adam优化器的β1和β2参数作用？

**答**：
- **β1 (通常=0.9)**：一阶矩估计的衰减率，控制动量
  - m_t = β1 * m_{t-1} + (1-β1) * g
  - 越大则历史梯度影响越大，收敛越平滑
- **β2 (通常=0.999)**：二阶矩估计的衰减率，控制自适应学习率
  - v_t = β2 * v_{t-1} + (1-β2) * g²
  - 越大则历史梯度平方影响越大，学习率越小

---

### Q2: 为什么Adam需要偏差修正？

**答**：
初始化时m_0=0, v_0=0，导致早期估计有偏差：
- 修正后：m_hat = m_t / (1 - β1^t)
- 当t很小时，β1^t接近1，修正幅度大
- 当t很大时，β1^t接近0，修正趋于0
- 这使得早期梯度估计更准确

---

### Q3: SGD with Momentum vs Adam？

| 方面 | SGD+Momentum | Adam |
|------|-------------|------|
| 收敛速度 | 慢 | 快 |
| 泛化能力 | 通常更好 | 可能略差 |
| 调参难度 | 需手动调学习率 | 默认参数效果还行 |
| 理论性质 | 更成熟 | 经验上好 |

---

### Q4: 梯度裁剪的作用？

**答**：
防止梯度爆炸：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 将梯度范数裁剪到max_norm以内
```
在RNN/LSTM/Transformer训练中常用，因为容易出现梯度爆炸。

---

## 相关知识点

- → [BP反向传播](../3-深度学习核心/BP反向传播/README.md) - 梯度计算
- → [激活函数](../3-深度学习核心/激活函数/README.md) - 梯度消失/爆炸
- → [Adam论文](../9-学习资料/论文/README.md) - 原始论文
