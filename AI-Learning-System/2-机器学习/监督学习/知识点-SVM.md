# 知识点卡片：支持向量机 (SVM)

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | 支持向量机 (SVM) |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P1 |
| 预估时间 | 8小时 |
| 面试频率 | ★★★★☆ |

---

## 核心原理

SVM找最大间隔的分类超平面：

```
硬间隔SVM：
min ½‖w‖²  s.t.  y_i(w^T x_i + b) ≥ 1

软间隔SVM（允许一些错误）：
min ½‖w‖² + C Σ ξ_i  s.t.  y_i(w^T x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0
```

### 对偶问题

通过拉格朗日乘子法转化为对偶问题：

```
max Σ α_i - ½ Σ Σ α_i α_j y_i y_j (x_i^T x_j)
s.t. α_i ≥ 0, Σ α_i y_i = 0

核技巧：将内积 x_i^T x_j 替换为 K(x_i, x_j)
```

---

## 核函数

```python
# 线性核：K(x,z) = x^T z（原始空间）
# 多项式核：K(x,z) = (γ x^T z + r)^d
# RBF核：K(x,z) = exp(-γ‖x-z‖²)（最常用，能拟合任意边界）
# Sigmoid核：K(x,z) = tanh(γ x^T z + r)
```

---

## sklearn实现

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM对特征缩放敏感，必须先标准化！
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])

# 线性SVM（数据量大时更快）
svm_linear = LinearSVC(C=1.0, max_iter=10000, dual=False)

# 参数说明
# C: 惩罚系数，C越大→过拟合风险大；C越小→欠拟合风险大
# gamma: RBF核的宽度，gamma越大→决策边界越复杂
# kernel: 'linear'/'poly'/'rbf'/'sigmoid'
```

---

## SVM vs 逻辑回归

| 方面 | 逻辑回归 | SVM |
|------|---------|-----|
| 损失函数 | 交叉熵 | Hinge Loss |
| 输出 | 概率 | 分类边界 |
| 对异常值 | 敏感（所有点都有影响） | 鲁棒（仅支持向量影响） |
| 核方法 | 不天然支持 | 天然支持（核技巧） |
| 大规模数据 | 更快（线性时间） | O(n²)或O(n³) |
| 多分类 | 天然支持 | 需要OvR/OvO |

---

## 面试高频问题

### Q1: SVM为什么用Hinge Loss？

**答**：
Hinge Loss: L = max(0, 1 - y(w^T x + b))

- 正确分类且间隔大于1时loss为0（稀疏性）
- 只关注"难分"样本（支持向量）
- 这是SVM"最大间隔"目标的直接翻译

### Q2: 核技巧的核心思想？

**答**：
不需要显式地将数据映射到高维空间，只需计算低维空间中的核函数值 K(x,z)。这使得可以在无限维空间中进行分类（如RBF核），同时保持计算在低维进行。

### Q3: C参数的作用？

**答**：
C控制"间隔最大化"和"分类错误最小化"之间的权衡：
- C大：严格要求正确分类 → 更小的间隔 → 容易过拟合
- C小：允许更多错误 → 更大的间隔 → 更好的泛化

---

## 相关知识点

- → [逻辑回归](./知识点-逻辑回归.md) - 对比分类器
- → [拉格朗日乘子](../../1-数学基础/最优化/知识点-拉格朗日乘子.md) - 对偶推导
