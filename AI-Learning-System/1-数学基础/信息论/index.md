# 信息论模块

本模块包含深度学习所需的信息论基础知识。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| 熵 | ★★★★☆ | P0 | ★★★★☆ |
| 交叉熵 | ★★★★★ | P0 | ★★★★★ |
| KL散度 | ★★★★☆ | P0 | ★★★★☆ |
| 最大互信息 | ★★★☆☆ | P1 | ★★☆☆☆ |

---

## 核心公式

### 1. 熵 (Entropy)

```python
# 信息熵：H(p) = -Σ p(x) log p(x)
# 衡量随机变量的不确定性

# 二进制熵：H(p) = -p log p - (1-p) log(1-p)

import numpy as np

def entropy(p):
    """计算熵"""
    p = np.array(p)
    return -np.sum(p * np.log2(p + 1e-10))

# 熵的性质：
# - 非负：H(p) ≥ 0
# - 当p均匀分布时，熵最大
# - 当p确定性分布时，熵最小(=0)
```

---

### 2. 联合熵与条件熵

```python
# 联合熵：H(X,Y) = -ΣΣ P(x,y) log P(x,y)
# 条件熵：H(Y|X) = Σ P(x) H(Y|X=x)
#                      = -ΣΣ P(x,y) log P(y|x)

# 链式法则：H(X,Y) = H(X) + H(Y|X)
```

---

### 3. 互信息

```python
# 互信息：I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
#           = ΣΣ P(x,y) log P(x,y) / (P(x)P(y))
# 衡量两个随机变量之间的相关性

def mutual_info(Pxy):
    """计算互信息"""
    Pxy = np.array(Pxy)
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    return np.sum(Pxy * np.log2(Pxy / (Px * Py + 1e-10) + 1e-10))
```

---

### 4. KL散度与交叉熵

```python
# KL散度：KL(p||q) = Σ p(x) log(p(x)/q(x))
# 交叉熵：H(p,q) = -Σ p(x) log q(x)

# 关系：H(p,q) = H(p) + KL(p||q)
# 因此：最小化H(p,q) ⇔ 最小化KL(p||q)（因为H(p)是常数）
```

---

## 在深度学习中的应用

```
┌─────────────────────────────────────────────────────────────┐
│                    信息论在DL中的应用                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  熵 → 衡量分布的不确定性                                     │
│  交叉熵 → 分类损失函数（负对数似然）                         │
│  KL散度 → VAE损失/知识蒸馏/隐式正则化                        │
│  互信息 → 对比学习（InfoNCE）                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## PyTorch实现

```python
import torch
import torch.nn.functional as F

# 交叉熵（内置）
loss = F.cross_entropy(logits, targets)

# KL散度
p = torch.distributions.Normal(0, 1)
q = torch.distributions.Normal(0.5, 1.5)
kl = torch.distributions.kl_divergence(p, q)  # KL(p||q)

# 手写KL
def kl_divergence(p, q, eps=1e-10):
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)

# InfoNCE（对比学习）
def info_nce(pos_score, neg_scores, temperature=0.1):
    # pos_score: 正样本分数
    # neg_scores: 负样本分数
    logits = torch.cat([pos_score.unsqueeze(-1), neg_scores], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    return F.cross_entropy(logits / temperature, labels)
```

---

## 面试高频问题

### Q1: 熵、交叉熵、KL散度的关系？

**答**：
```
H(p,q) = -Σ p(x) log q(x)        # 交叉熵
H(p) = -Σ p(x) log p(x)          # 熵
KL(p||q) = Σ p(x) log(p(x)/q(x)) # KL散度

关系：H(p,q) = H(p) + KL(p||q)

最小化交叉熵 = 最小化KL散度（因为H(p)是常数）
```

---

### Q2: KL散度的非对称性在工程中的应用？

**答**：
- **VAE**：使用KL散度约束隐变量分布，使q(z|x)接近p(z)
- **知识蒸馏**：用大模型（teacher）指导小模型（student），KL(teacher||student)
- **RLHF**：KL约束确保新策略不会偏离旧策略太远

---

## 相关知识点

- → [交叉熵损失](知识点-交叉熵.md) - 分类损失
- → [VAE](../3-深度学习核心/Diffusion/README.md) - 变分推断
- → [对比学习](../3-深度学习核心/Transformer/README.md) - InfoNCE
