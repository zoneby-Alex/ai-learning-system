# 知识点卡片：KL散度

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | KL散度 (Kullback-Leibler Divergence) |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P1 |
| 预估时间 | 5小时 |
| 面试频率 | ★★★★☆ |

---

## 核心原理

### 定义

```
KL(p‖q) = Σ_x p(x) log(p(x) / q(x))
        = E_{x~p} [log p(x) - log q(x)]

性质：
1. 非负性：KL(p‖q) ≥ 0（当且仅当p=q时取0）
2. 非对称性：KL(p‖q) ≠ KL(q‖p)
3. 不满足三角不等式
```

### Forward KL vs Reverse KL

```
Forward KL: KL(p‖q) = Σ p log(p/q)
- 强迫q在p高概率的地方也高（mean-seeking）
- 可能导致q过于分散

Reverse KL: KL(q‖p) = Σ q log(q/p)
- 强迫q只在p高概率的地方有概率（mode-seeking）
- 导致q可能只覆盖p的少数模式
```

---

## 代码实现

```python
import numpy as np
import torch
import torch.nn.functional as F

def kl_divergence(p, q, eps=1e-15):
    """KL(p||q)"""
    p = np.asarray(p)
    q = np.asarray(q)
    q = np.clip(q, eps, 1 - eps)
    return np.sum(p * np.log(p / q))

# PyTorch
def kl_div_torch(log_p, log_q):
    """KL散度（输入为log概率）"""
    return torch.sum(torch.exp(log_p) * (log_p - log_q))

# 高斯分布的KL散度
def kl_gaussian(mu1, sigma1, mu2, sigma2):
    """KL(N(μ₁,σ₁²) || N(μ₂,σ₂²))"""
    return (np.log(sigma2 / sigma1)
            + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
            - 0.5)

# 测试
print(f"KL(相同分布): {kl_divergence([0.5, 0.5], [0.5, 0.5]):.6f}")  # 0
print(f"KL(不同分布): {kl_divergence([0.9, 0.1], [0.5, 0.5]):.4f}")   # 0.3680
print(f"KL(相反): {kl_divergence([0.5, 0.5], [0.9, 0.1]):.4f}")        # 0.5108
# 注意不对称性！
```

---

## 在深度学习中的应用

### 1. VAE的损失函数

```python
"""
VAE损失 = 重构误差 + KL散度正则项

L = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

其中 p(z) ~ N(0, 1)（标准正态先验）
     q(z|x) ~ N(μ(x), σ²(x))（编码器输出）

KL项确保隐变量分布接近标准正态分布
"""

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, logvar):
        # 重构损失（MSE 或 BCE）
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL散度：KL(N(μ,σ²) || N(0,1))
        # = -0.5 * Σ (1 + log σ² - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss
```

### 2. 知识蒸馏

```python
"""
知识蒸馏中使用KL散度让学生模型的输出分布
逼近教师模型的输出分布

L = CE(y, student(x)) + α * T² * KL(teacher(x)/T || student(x)/T)
"""

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    知识蒸馏损失
    T: temperature（温度参数）
    alpha: 蒸馏损失权重
    """
    # 硬标签损失（交叉熵）
    hard_loss = F.cross_entropy(student_logits, labels)

    # 软标签损失（KL散度）
    soft_teacher = F.log_softmax(teacher_logits / T, dim=-1)  # 注意：教师用log_softmax
    soft_student = F.softmax(student_logits / T, dim=-1)      # 学生用softmax

    # KL(teacher||student) = Σ teacher/T * log(teacher/student) / T
    # PyTorch KLDivLoss: input=log-prob, target=prob
    soft_loss = F.kl_div(soft_teacher, soft_student, reduction='batchmean')
    soft_loss = soft_loss * T * T  # 温度缩放

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 3. Diffusion Model

```python
"""
Diffusion训练等价于最小化以下KL散度：

KL(q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t))

其中 q 是真实的后验（已知），p_θ 是模型学习的分布
最终简化为预测噪声的MSE损失
"""
```

---

## 面试高频问题

### Q1: KL散度为什么非对称？

**答**：
```
KL(p‖q) ≠ KL(q‖p) 的原因在于log(p/q)的权重不同：

KL(p‖q) = Σ p(x) log(p(x)/q(x))
→ 当p(x)大而q(x)小时，惩罚很大（重要区域不能忽略）
→ 当q(x)大而p(x)小时，惩罚很小（可以容忍多余的q）

KL(q‖p) = Σ q(x) log(q(x)/p(x))
→ 当q(x)大而p(x)小时，惩罚很大（不能凭空添加）
→ 当p(x)大而q(x)小时，惩罚很小

因此两者优化行为不同：
- Forward KL：保守、分布广
- Reverse KL：激进、分布窄（mode-seeking）
```

### Q2: KL散度与交叉熵的关系？

**答**：
```
CE(p, q) = H(p) + KL(p‖q)

当p固定时（如训练数据固定），H(p)是常数
因此：min CE(p,q) = min KL(p‖q)

特殊情况：p是one-hot分布
→ H(p) = 0
→ CE(p,q) = KL(p‖q) = -log q(correct_class)
```

### Q3: 为什么VAE用KL散度？

**答**：
1. KL散度作为正则项，约束隐变量分布接近标准正态
2. 这使得隐空间连续且光滑（可以在隐空间插值）
3. 便于采样生成（直接从N(0,1)采样即可）
4. KL(N(μ,σ²)||N(0,1))有闭式解，计算高效

---

## 练习题

```python
# 1. 验证KL散度非负性
# KL(p||q) ≥ 0，等号仅当p=q
p = np.array([0.3, 0.3, 0.4])
q = np.array([0.3, 0.3, 0.4])
assert kl_divergence(p, q) < 1e-10  # 相等时为0

q2 = np.array([0.5, 0.2, 0.3])
assert kl_divergence(p, q2) > 0  # 不相等时大于0

# 2. 验证不对称性
print(f"KL(p||uniform): {kl_divergence(p, [1/3, 1/3, 1/3]):.4f}")
print(f"KL(uniform||p): {kl_divergence([1/3, 1/3, 1/3], p):.4f}")
# 两者不同！

# 3. 高斯的KL散度
mu1, sigma1 = 0.0, 1.0
mu2, sigma2 = 2.0, 0.5
kl_12 = kl_gaussian(mu1, sigma1, mu2, sigma2)
kl_21 = kl_gaussian(mu2, sigma2, mu1, sigma1)
print(f"KL(N1||N2): {kl_12:.4f}")
print(f"KL(N2||N1): {kl_21:.4f}")
# 也非对称
```

---

## 相关知识点

- → [熵](./知识点-熵.md) - 信息量度量
- → [交叉熵](./知识点-交叉熵.md) - CE与KL的关系
- → [VAE](../../3-深度学习核心/Diffusion/README.md) - KL应用
- → [最大似然估计](../概率统计/知识点-最大似然估计.md) - CE=MLE
