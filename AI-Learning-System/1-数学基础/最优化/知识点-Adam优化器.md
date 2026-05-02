# 知识点卡片：Adam优化器

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | Adam优化器及其变体 |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 6小时 |
| 面试频率 | ★★★★★ |

---

## 核心原理

Adam = Momentum + RMSProp + Bias Correction

```
Adam更新公式：

m_t = β₁ m_{t-1} + (1-β₁) g_t          # 一阶矩（动量）
v_t = β₂ v_{t-1} + (1-β₂) g_t²          # 二阶矩（自适应学习率）

m̂_t = m_t / (1 - β₁^t)                   # 一阶矩偏差修正
v̂_t = v_t / (1 - β₂^t)                   # 二阶矩偏差修正

θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)     # 参数更新

默认参数：α=0.001, β₁=0.9, β₂=0.999, ε=1e-8
```

---

## 从零实现

```python
import numpy as np
import torch
import torch.nn as nn

class Adam:
    """Adam优化器从零实现"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None   # 一阶矩
        self.v = None   # 二阶矩
        self.t = 0      # 时间步

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            # 更新矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 参数更新
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## PyTorch实现

```python
import torch.optim as optim

# 基础Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# AdamW（解耦权重衰减）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# AdamW把weight_decay从loss中解耦到参数更新中
# θ = θ - α*(m̂/(√v̂+ε) + λθ) 而不是 loss = task_loss + λ‖θ‖²

# Adam with amsgrad
optimizer = optim.Adam(model.parameters(), amsgrad=True)

# 稀疏梯度场景
optimizer = optim.SparseAdam(model.parameters(), lr=0.001)
```

---

## 关键参数分析

### β₁（一阶矩衰减率 = 0.9）

- 控制动量的衰减速度
- 越大：历史梯度影响越久，收敛越平滑
- 越小：更注重近期梯度，反应更快
- 典型值：0.9（视觉/语言模型通用）

### β₂（二阶矩衰减率 = 0.999）

- 控制自适应学习率的更新速度  
- 越大：历史梯度平方影响越久，学习率越小
- 越小：更注重近期梯度大小，适应更快
- 典型值：0.999（视觉）、0.95（GAN训练）

### ε（数值稳定项）

```python
# ε在分母√v̂+ε中，防止除以0
# 太小：数值不稳定
# 太大：影响优化效果
# 默认1e-8通常有效
```

---

## 偏差修正详解

```python
"""
为什么需要偏差修正？

初始化m₀=0, v₀=0

t=1: m₁ = (1-β₁)g₁，期望 E[m₁] = (1-β₁)E[g] ≠ E[g]
t=2: m₂ = β₁(1-β₁)g₁ + (1-β₁)g₂，期望也偏小

修正后：
m̂_t = m_t / (1 - β₁^t)

当t很小时：
1 - β₁^t ≈ 0（小），修正幅度大
当t很大时：
1 - β₁^t ≈ 1（接近1），修正趋于0

例如 β₁=0.9：
t=1: 1-β₁^t = 0.1, m̂₁ = m₁/0.1 (放大10倍!)
t=10: 1-β₁^t ≈ 0.65
t=100: 1-β₁^t ≈ 1.0
"""

# 可视化偏差修正
beta1 = 0.9
for t in [1, 2, 5, 10, 50, 100]:
    correction = 1 - beta1 ** t
    print(f"t={t:3d}: correction factor={correction:.4f}, scale={1/correction:.2f}x")
```

---

## 优化器对比

| 方面 | SGD+Momentum | Adam | AdamW |
|------|-------------|------|-------|
| 收敛速度 | 慢 | 快 | 快 |
| 泛化能力 | 通常更好 | 可能略差 | 改善 |
| 调参难度 | 需手动调LR | 默认参数还行 | 默认参数还行 |
| 显存占用 | 最低 | 需存m和v | 需存m和v |
| 权重衰减 | L2正则化 | L2正则化 | 解耦权重衰减 |
| 适用场景 | CV | NLP/LLM | LLM训练 |

---

## 面试高频问题

### Q1: Adam为什么有效？

**答**：
1. **Momentum（一阶矩）**：加速收敛，平滑梯度
2. **自适应学习率（二阶矩）**：每个参数有自己的学习率，适合稀疏梯度
3. **偏差修正**：解决初始估计偏差
4. **组合优势**：兼具SGD+Momentum和RMSProp的优点

### Q2: AdamW相比Adam的改进？

**答**：
```
Adam（L2正则化）：
loss = task_loss + λ‖θ‖²/2
这个λ会通过m和v影响更新，不够直接

AdamW（解耦权重衰减）：
θ = θ - α*(m̂/(√v̂+ε) + λθ)
权重衰减直接作用于参数，与自适应学习率解耦

结果：AdamW泛化能力更好，成为LLM训练标配
```

### Q3: 为什么有时候SGD比Adam泛化好？

**答**：
- Adam的自适应学习率可能过早收敛到"尖锐"的局部最优
- SGD的均匀学习率倾向于收敛到"平坦"的局部最优
- 平坦的最优通常泛化更好（对参数扰动不敏感）
- 因此有时在CV任务中会用SGD，在NLP/LLM中用AdamW

---

## 训练技巧

```python
# 1. Warmup + Adam
# 开始用很小的学习率，逐渐增大到目标值
warmup_steps = 1000
def get_lr(step, d_model, warmup_steps):
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

# 2. 梯度裁剪配合Adam
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 混合精度训练中Adam的稳定性
scaler = torch.cuda.amp.GradScaler()
# Adam在混合精度下可能需要调整ε（增大到1e-7或1e-6）
```

---

## 相关知识点

- → [梯度下降](./知识点-梯度下降.md) - GD/SGD基础
- → [学习率调度](./知识点-学习率调度.md) - LR策略
- → [BP反向传播](../../3-深度学习核心/BP反向传播/README.md) - 梯度计算
