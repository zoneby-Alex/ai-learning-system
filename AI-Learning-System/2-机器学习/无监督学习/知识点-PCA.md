# 知识点卡片：主成分分析 (PCA)

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | 主成分分析 (PCA) |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P0 |
| 预估时间 | 6小时 |
| 面试频率 | ★★★★☆ |

---

## 核心原理

PCA找到数据方差最大的正交方向：

```
步骤：
1. 中心化：X = X - mean(X)
2. 协方差矩阵：Σ = X^T X / (n-1)
3. 特征分解：Σ v = λ v
4. 选择前k个最大特征值对应的特征向量作为主成分
5. 投影：X_reduced = X @ V_k
```

**实际通过SVD实现更稳定**：
```
X = U Σ V^T
主成分 = V^T 的前k行
降维数据 = U_k Σ_k（或 X @ V_k^T）
```

---

## 从零实现

```python
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        # 中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD分解
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 主成分
        if self.n_components is None:
            self.n_components = X.shape[1]
        self.components_ = Vt[:self.n_components]

        # 解释方差
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components_ + self.mean_
```

---

## sklearn实现

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 标准化后PCA（推荐，尤其是特征量纲不同时）
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))  # 保留95%方差
])
X_reduced = pipeline.fit_transform(X)

# 或直接指定主成分数
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# 解释方差
print(f"各主成分方差比: {pca.explained_variance_ratio_[:10]}")
print(f"累积方差: {pca.explained_variance_ratio_.cumsum()[:10]}")
print(f"总保留方差: {pca.explained_variance_ratio_.sum():.4f}")
```

---

## PCA与自编码器的对比

| 方面 | PCA | 自编码器 |
|------|-----|---------|
| 变换 | 线性 | 非线性 |
| 可解释性 | 高（主成分有几何意义） | 低 |
| 计算效率 | 快 | 慢 |
| 表达能力 | 有限 | 强 |
| 参数 | 无（确定性的） | 有（需训练） |

---

## 面试高频问题

### Q1: PCA数据为什么需要标准化？

**答**：
PCA是对协方差矩阵做特征分解，而协方差受量纲影响极大。如果某特征的值域是[0,100]而另一个是[0,1]，PCA会认为第一个特征方差更大（10000倍），完全主导主成分方向。标准化使所有特征在同等量级上比较。

### Q2: PCA和SVD的关系？

**答**：
PCA等价于对中心化数据X做SVD：
- 右奇异向量V^T的行 = 主成分方向
- 奇异值的平方/(n-1) = 特征值（方差）
- 实际PCA实现都用SVD，因为数值更稳定

### Q3: 如何选择主成分数量？

**答**：
1. **方差阈值法**：保留方差比例达到阈值（如95%）
2. **肘部法则**：画方差曲线找拐点
3. **下游任务**：根据分类/回归任务的效果选择

---

## 相关知识点

- → [SVD](../../1-数学基础/线性代数/知识点-特征值与SVD.md) - PCA的数学基础
- → [KMeans](./知识点-KMeans.md) - 降维+聚类组合
- → [t-SNE](./README.md) - 非线性降维对比
