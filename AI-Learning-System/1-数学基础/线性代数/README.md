# 线性代数模块

本模块包含深度学习所需的线性代数基础知识。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| 向量/矩阵运算 | ★★★★★ | P0 | ★★★★★ |
| 范数 | ★★★★★ | P0 | ★★★★☆ |
| 特征值/特征向量 | ★★★☆☆ | P1 | ★★★☆☆ |
| SVD分解 | ★★★☆☆ | P1 | ★★★☆☆ |
| 矩阵微分 | ★★★☆☆ | P1 | ★★☆☆☆ |

---

## 核心知识点

### 1. 向量与矩阵运算

```python
import numpy as np

# 向量：形状 (n,)
# 矩阵：形状 (m, n)

# 矩阵乘法：(m, n) @ (n, k) -> (m, k)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B  # (3, 5)

# 逐元素乘法：形状必须相同
A * B  # (3, 4) * (3, 4) -> (3, 4)

# 转置
A.T  # (4, 3)

# 求逆（方阵）
np.linalg.inv(A @ A.T)  # (A @ A.T)^-1
```

---

### 2. 范数

```python
x = np.array([3, 4])

# L1范数：曼哈顿距离
l1 = np.abs(x).sum()  # 7

# L2范数：欧氏距离
l2 = np.linalg.norm(x)  # 5

# L∞范数：最大绝对值
linf = np.abs(x).max()  # 4

# 矩阵F范数
A = np.random.randn(3, 4)
fro = np.linalg.norm(A, 'fro')  # sqrt(sum(A_ij^2))
```

---

### 3. 特征值与特征向量

```python
# A @ v = λ @ v
# v是特征向量，λ是特征值

A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

# 特征值分解：A = V @ diag(λ) @ V^-1
# 用途：PCA、谱范数、稳定性分析
```

---

### 4. SVD分解

```python
# A = U @ S @ V^T
# U, V是正交矩阵，S是对角矩阵（奇异值）

A = np.random.randn(3, 4)
U, S, Vt = np.linalg.svd(A)

# 用途：
# - 降维（保留前k个奇异值）
# - 伪逆计算
# - 图像压缩
# - 协同过滤

# 用SVD做PCA
def pca_svd(X, k):
    U, S, Vt = np.linalg.svd(X - X.mean(axis=0))
    return U[:, :k] * S[:k], Vt[:k, :]
```

---

## 在神经网络中的应用

```python
# 神经网络 = 线性变换 + 非线性激活

# 全连接层：y = W @ x + b
# 卷积层：y = Conv(W, x)
# 注意力：attn = softmax(Q @ K^T) @ V

# 权重正则化
loss = task_loss + 0.01 * (W ** 2).sum()  # L2正则化

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 面试高频问题

### Q1: 矩阵乘法和逐元素乘法的区别？

**答**：
- 矩阵乘法：(m,n) @ (n,k) → (m,k)，涉及行-列点积
- 逐元素乘法：形状相同，对应位置相乘

---

### Q2: L1和L2正则化的几何解释？

**答**：
- L1（菱形）：顶点处解更稀疏
- L2（球形）：解更平滑

---

## 相关知识点

- → [范数](知识点-范数.md) - 矩阵范数
- → [特征值与SVD](知识点-特征值与SVD.md) - 矩阵分解
