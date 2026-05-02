# 概率统计模块

本模块包含深度学习所需的概率统计基础知识。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| 概率分布 | ★★★★☆ | P0 | ★★★★☆ |
| 条件概率/贝叶斯 | ★★★★☆ | P0 | ★★★★☆ |
| 期望/方差/协方差 | ★★★★★ | P0 | ★★★★★ |
| 最大似然估计(MLE) | ★★★★☆ | P0 | ★★★★☆ |
| KL散度 | ★★★★☆ | P1 | ★★★☆☆ |
| 中心极限定理 | ★★★☆☆ | P2 | ★★☆☆☆ |

---

## 核心知识点

### 1. 概率分布

| 分布 | 类型 | 公式 | DL应用 |
|------|------|------|--------|
| 伯努利 | 离散 | P(X=1)=p | 二分类/sigmoid |
| Categorical | 离散 | P(X=k)=p_k | 多分类/softmax |
| 高斯 | 连续 | N(μ,σ²) | 回归/初始化/噪声 |
| Laplace | 连续 | Lap(μ,b) | L1损失 |

---

### 2. 贝叶斯公式

```python
# P(A|B) = P(B|A) * P(A) / P(B)
# 后验 = 似然 * 先验 / 归一化常数

# 在ML中的应用：
# - 朴素贝叶斯分类器
# - 贝叶斯优化
# - 变分推断
```

---

### 3. 最大似然估计 (MLE)

```python
# 似然函数：L(θ|D) = P(D|θ)
# MLE: θ* = argmax_θ L(θ|D)

# 对数似然：log L(θ|D) = Σ log P(x_i|θ)
# 最大化对数似然 = 最小化负对数似然

# 交叉熵 = 负对数似然
# 因此：最小化CE = MLE
```

---

### 4. KL散度

```python
# KL散度：衡量两个分布的差异
# KL(p||q) = Σ p(x) log(p(x)/q(x))
#           = Σ p(x) log p(x) - Σ p(x) log q(x)
#           = -H(p) - Σ p(x) log q(x)
#           = H(p,q) - H(p)

# 性质：
# - 非负：KL(p||q) ≥ 0
# - 非对称：KL(p||q) ≠ KL(q||p)
# - 不满足三角不等式

# 在DL中的应用：
# - VAE的损失函数
# - 知识蒸馏
# - 隐式正则化
```

---

## PyTorch概率分布

```python
import torch.distributions as dist
import torch

# 高斯分布
p = dist.Normal(loc=0.0, scale=1.0)
x = p.sample()           # 采样
log_prob = p.log_prob(x) # 对数概率密度

# Bernoulli分布
p = dist.Bernoulli(probs=0.5)
x = p.sample()

# Categorical分布（多分类）
p = dist.Categorical(probs=torch.tensor([0.2, 0.5, 0.3]))
x = p.sample()

# 变分分布（VAE用）
q_z = dist.Normal(loc=mu, scale=sigma)
z = q_z.rsample()  # 重参数化采样
```

---

## 面试高频问题

### Q1: MLE和MAP的区别？

**答**：
- **MLE (最大似然估计)**：θ* = argmax P(D|θ)
  - 只考虑似然，不考虑先验
- **MAP (最大后验估计)**：θ* = argmax P(θ|D) ∝ P(D|θ) P(θ)
  - 考虑先验，相当于加了正则化的MLE

```python
# MAP = MLE + log P(θ)
# 如果先验P(θ)是高斯分布，则MAP等价于L2正则化
# 如果先验P(θ)是Laplace分布，则MAP等价于L1正则化
```

---

### Q2: 为什么用交叉熵而不是MSE？

**答**：
1. **概率解释**：CE是分类的自然损失（负对数似然）
2. **梯度特性**：CE梯度是pred-label，Sigmoid+MSE梯度是pred*(1-pred)*(pred-label)
3. **收敛速度**：CE更快，MSE容易梯度消失

---

### Q3: KL散度的非对称性说明什么？

**答**：
- KL(p||q)：用q去近似p的信息损失（forward KL）
- KL(q||p)：用p去近似q的信息损失（reverse KL）

```python
# Forward KL: min KL(p||q) = Σ p log(p/q)
# - 强迫q在p高概率的地方也高概率
# - 忽略p低概率的地方

# Reverse KL: min KL(q||p) = Σ q log(q/p)
# - 强迫q只在p高概率的地方有概率
# - 忽略p低概率的地方（mode-seeking）
```

---

## 相关知识点

- → [交叉熵](../信息论/知识点-交叉熵.md) - 损失函数
- → [KL散度](../信息论/知识点-KL散度.md) - 分布差异
- → [VAE](../3-深度学习核心/Diffusion/README.md) - 变分推断
