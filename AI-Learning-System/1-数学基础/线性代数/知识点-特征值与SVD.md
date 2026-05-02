# 知识点卡片：特征值与SVD

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | 特征值分解与奇异值分解(SVD) |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P1 |
| 预估时间 | 8小时 |
| 面试频率 | ★★★★☆ |

---

## 核心原理

### 特征值分解

对于方阵 A ∈ R^{n×n}，若存在 λ 和非零向量 v 使得：

```
A v = λ v
```

则 λ 为特征值，v 为特征向量。

**几何意义**：矩阵A作用于v时，只改变v的长度(λ倍)，不改变方向。

### SVD分解

对于任意矩阵 A ∈ R^{m×n}：

```
A = U Σ V^T

其中：
- U ∈ R^{m×m}：左奇异向量矩阵（正交）
- Σ ∈ R^{m×n}：奇异值对角矩阵（降序排列）
- V ∈ R^{n×n}：右奇异向量矩阵（正交）
```

---

## 代码实现

### NumPy实现

```python
import numpy as np

# 特征值分解
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证 A v = λ v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"Av = {A @ v}")
    print(f"λv = {lam * v}")
    print(f"误差: {np.linalg.norm(A @ v - lam * v):.10f}")
```

### SVD实现

```python
# NumPy SVD
A = np.random.randn(5, 3)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"U shape: {U.shape}")   # (5, 3)
print(f"S: {S}")               # 奇异值
print(f"Vt shape: {Vt.shape}") # (3, 3)

# 重构
A_reconstructed = U @ np.diag(S) @ Vt
print(f"重构误差: {np.linalg.norm(A - A_reconstructed):.10f}")

# 截断SVD（降维）
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"截断SVD误差: {np.linalg.norm(A - A_approx):.4f}")
```

### PyTorch实现

```python
import torch

A = torch.randn(5, 3)
U, S, Vt = torch.linalg.svd(A, full_matrices=False)

# 截断SVD
k = 2
U_k = U[:, :k]
S_k = S[:k]
Vt_k = Vt[:k, :]
A_approx = U_k @ torch.diag(S_k) @ Vt_k
```

---

## 在深度学习中的应用

### 1. PCA降维

```python
def pca_via_svd(X, k):
    """通过SVD实现PCA"""
    # 中心化
    X_centered = X - X.mean(axis=0)

    # SVD分解
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 前k个主成分
    components = Vt[:k, :]
    X_reduced = X_centered @ components.T

    # 解释方差比例
    explained_variance = (S ** 2) / (X_centered.shape[0] - 1)
    ratio = explained_variance[:k].sum() / explained_variance.sum()

    return X_reduced, components, ratio

# 应用到MNIST
from sklearn.datasets import load_digits
X, _ = load_digits(return_X_y=True)
X_reduced, _, ratio = pca_via_svd(X, 50)
print(f"保留方差比例: {ratio:.4f}")
```

### 2. 谱归一化（SN-GAN）

```python
def spectral_norm(W, num_iters=10):
    """幂迭代法计算谱范数（最大奇异值）"""
    u = torch.randn(W.shape[0], 1)
    u = u / torch.norm(u)

    for _ in range(num_iters):
        v = W.T @ u
        v = v / torch.norm(v)
        u = W @ v
        u = u / torch.norm(u)

    sigma = (u.T @ W @ v).squeeze()
    return W / sigma  # 谱归一化
```

### 3. 权重初始化分析

```python
# 正交初始化：使用SVD确保权重矩阵正交
def orthogonal_init(weight):
    """正交初始化"""
    U, _, Vt = torch.linalg.svd(weight, full_matrices=False)
    if U.shape == weight.shape:
        return U
    else:
        return Vt
```

---

## 面试高频问题

### Q1: 特征值分解与SVD的区别？

**答**：

| 方面 | 特征值分解 | SVD |
|------|-----------|-----|
| 适用矩阵 | 方阵 | 任意矩阵 |
| 分解形式 | A = VΛV^{-1} | A = UΣV^T |
| 正交性 | V不一定正交 | U和V都正交 |
| 数值稳定性 | 可能不稳定 | 非常稳定 |
| 应用 | 对称矩阵分析 | PCA/推荐系统/NLP |

**关键**：SVD适用于任意矩阵，且U和V都是正交矩阵，数值稳定性远优于特征值分解。PCA底层实际通过SVD计算。

### Q2: PCA的推导过程？

**答**：

```
目标：找到投影方向w，使得投影后方差最大
max_w Var(Xw) = w^T Σ w, s.t. w^T w = 1

拉格朗日乘子法：
L(w, λ) = w^T Σ w - λ(w^T w - 1)
∂L/∂w = 2Σw - 2λw = 0
→ Σw = λw

结论：w是协方差矩阵Σ的特征向量，λ是对应的特征值
方差最大方向 = 最大特征值对应的特征向量
```

实际计算通过SVD更稳定：对中心化数据X进行SVD，V^T的行就是主成分。

### Q3: 为什么正交初始化有效？

**答**：
- 正交矩阵保持向量长度：‖Wx‖ = ‖x‖
- 防止梯度爆炸/消失：每层输入输出范数不变
- 信息保持：正交变换是可逆的，不丢失信息
- 训练初期稳定性：特征值全为1，条件数为1

---

## 奇异值的含义

```python
# 奇异值代表矩阵在对应方向上的"拉伸因子"
# σ₁ = max_{‖x‖=1} ‖Ax‖ （最大拉伸）
# σₙ = min_{‖x‖=1} ‖Ax‖ （最小拉伸）

# 条件数 = σ₁ / σₙ，衡量矩阵的数值稳定性
# 条件数大 → 对输入变化敏感 → 训练不稳定
```

---

## 练习题

```python
import numpy as np

# 1. 验证SVD重构
A = np.random.randn(5, 4)
U, S, Vt = np.linalg.svd(A)
Sigma = np.zeros((5, 4))
np.fill_diagonal(Sigma, S)
A_recon = U @ Sigma @ Vt
print(f"重构误差: {np.linalg.norm(A - A_recon):.10f}")

# 2. 用SVD实现图像压缩
def svd_compress(image, k):
    """用前k个奇异值压缩图像"""
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    compression_ratio = (image.size) / (k * (U.shape[0] + Vt.shape[1] + 1))
    return compressed, compression_ratio

# 3. PCA手写推导验证
X = np.random.randn(100, 10)
X_centered = X - X.mean(axis=0)
cov = X_centered.T @ X_centered / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# 最大特征值对应的特征向量是第一主成分
print(f"第一主成分方向: {eigenvectors[:, -1][:5]}")
```

---

## 相关知识点

- → [向量与矩阵运算](./知识点-向量与矩阵运算.md) - 矩阵基础
- → [范数](./知识点-范数.md) - 谱范数与SVD
- → [PCA](../无监督学习/知识点-PCA.md) - 降维应用
- → [矩阵微分](./知识点-矩阵微分.md) - 梯度计算
