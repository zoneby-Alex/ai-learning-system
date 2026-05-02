# 无监督学习模块

本模块包含无监督学习的核心算法和原理。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| KMeans | ★★★★☆ | P0 | ★★★★☆ |
| PCA | ★★★★☆ | P0 | ★★★★☆ |
| t-SNE | ★★★☆☆ | P1 | ★★★☆☆ |
| GMM/EM | ★★★☆☆ | P2 | ★★☆☆☆ |
| DBSCAN | ★★★☆☆ | P2 | ★★☆☆☆ |

---

## 算法速查

| 算法 | 类型 | 面试重点 |
|------|------|---------|
| KMeans | 聚类 | 迭代优化/收敛性 |
| PCA | 降维 | 方差最大化/手推 |
| t-SNE | 降维 | 概率解释/KL散度 |
| GMM | 聚类 | EM算法 |
| DBSCAN | 聚类 | 密度概念 |

---

## KMeans

### 原理

```python
# 目标：min Σ ||x_i - μ_{c_i}||²
# 迭代：
#   1. 分配：c_i = argmin_k ||x_i - μ_k||
#   2. 更新：μ_k = mean(x_i where c_i = k)
```

### sklearn实现

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5, n_init=10, max_iter=300)
labels = model.fit_predict(X)

# 评估
inertia = model.inertia_  # 簇内平方和
```

### 手写实现

```python
import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-6):
    """KMeans实现"""
    # 随机初始化中心
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]

    for _ in range(max_iters):
        # 分配
        distances = ((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)

        # 更新中心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 收敛判断
        if np.abs(new_centroids - centroids).max() < tol:
            break
        centroids = new_centroids

    return labels, centroids
```

---

## PCA

### 原理

```python
# 目标：找到方差最大的正交方向
# 步骤：
#   1. 中心化：X = X - mean(X)
#   2. 协方差矩阵：S = X^T X / (n-1)
#   3. 特征分解：S = V Λ V^T
#   4. 选择前k个主成分
```

### sklearn实现

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # 保留95%方差
X_reduced = pca.fit_transform(X)

# 评估
explained_var = pca.explained_variance_ratio_
cumsum = np.cumsum(explained_var)
```

### 手推实现

```python
def pca(X, k):
    """PCA实现"""
    # 中心化
    X = X - X.mean(axis=0)

    # SVD分解
    U, S, Vt = np.linalg.svd(X)

    # 选择前k个主成分
    components = Vt[:k, :]
    X_reduced = X @ components.T

    return X_reduced, components
```

---

## t-SNE

### 原理

```python
# 高维相似性：p_{j|i} = exp(-||x_i - x_j||² / 2σ_i²) / Σ_{k≠i} exp(-||x_i - x_k||² / 2σ_i²)
# 低维相似性：q_{ij} = (1 + ||y_i - y_j||²)^{-1} / Σ_{k≠l} (1 + ||y_k - y_l||²)^{-1}
# 目标：min KL(p || q)
```

### sklearn实现

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_2d = tsne.fit_transform(X)
```

---

## DBSCAN

### 原理

```python
# 核心点：ε邻域内至少有min_samples个点
# 边界点：ε邻域内有核心点但不足min_samples
# 噪声点：既不是核心点也不是边界点
```

### sklearn实现

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
```

---

## 面试高频问题

### Q1: KMeans的收敛性？

**答**：
- 每次迭代都减小簇内平方和
- 下界为0，因此收敛
- 但可能陷入局部最优（不同初始化结果不同）
- 解决：多次初始化（k-means++）

---

### Q2: PCA和SVD的关系？

**答**：
- PCA通过SVD实现
- 对于中心化数据，PCA的 eigenvectors = V（来自SVD的右奇异向量）
- SVD更数值稳定，是实际计算PCA的方法

---

## 相关知识点

- → [监督学习](../监督学习/README.md) - 分类/回归
- → [集成学习](../集成学习/README.md) - 降维用于特征工程
