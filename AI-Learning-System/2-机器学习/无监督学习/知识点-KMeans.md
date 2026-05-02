# 知识点卡片：KMeans聚类

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | KMeans聚类算法 |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P0 |
| 预估时间 | 4小时 |
| 面试频率 | ★★★★☆ |

---

## 核心原理

KMeans的目标：将n个点分到k个簇，使簇内平方和最小：

```
目标：min Σ_{i=1}^{n} ‖x_i - μ_{c_i}‖²

EM式迭代：
1. E步（分配）：c_i = argmin_k ‖x_i - μ_k‖²
2. M步（更新）：μ_k = mean({x_i : c_i = k})
```

---

## 从零实现

```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # k-means++ 初始化
        self.centroids = self._kmeans_plus_plus_init(X, rng)

        for _ in range(self.max_iters):
            # 1. 分配：计算每个点到每个中心的距离
            distances = ((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # 2. 更新：重新计算中心
            new_centroids = np.array([
                X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)
            ])

            # 3. 收敛判断
            shift = np.abs(new_centroids - self.centroids).max()
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.inertia_ = ((X - self.centroids[self.labels_]) ** 2).sum()
        return self

    def _kmeans_plus_plus_init(self, X, rng):
        """k-means++ 初始化：让初始中心点尽量分散"""
        centroids = [X[rng.randint(len(X))]]
        for _ in range(1, self.n_clusters):
            dists = np.min([((X - c) ** 2).sum(axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            centroids.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centroids)

    def predict(self, X):
        distances = ((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1)
```

---

## sklearn实现

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5, n_init='auto', random_state=42)
labels = model.fit_predict(X)

# 评估
inertia = model.inertia_  # 簇内平方和（越小越好）
centers = model.cluster_centers_

# 选择最优k（肘部法则）
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init='auto')
    km.fit(X)
    inertias.append(km.inertia_)
# 曲线拐点处为最优k
```

---

## 面试高频问题

### Q1: KMeans的收敛性证明？

**答**：
- 每次迭代都减小（或不增）目标函数
- 目标函数有下界（≥0）
- 单调递减有下界 → 必定收敛
- 但可能收敛到局部最优（依赖初始化）

### Q2: k-means++ 初始化的原理？

**答**：
- 随机选第一个中心
- 后续中心按距离已有中心的平方距离作为概率选取
- 越远的点被选中的概率越大
- 使初始中心尽量分散，减少陷入差局部最优的概率

### Q3: KMeans的局限性？

**答**：
1. 需要预设k值
2. 对初始中心敏感
3. 只能发现凸形簇（不能处理环形等非凸簇）
4. 对异常值敏感
5. 假设各簇大小均衡

---

## 相关知识点

- → [PCA](./知识点-PCA.md) - 降维后聚类
- → [GMM](../无监督学习/README.md) - 软聚类
