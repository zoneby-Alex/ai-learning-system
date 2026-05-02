# 监督学习模块

本模块包含监督学习的核心算法和原理。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| 线性回归 | ★★★★★ | P0 | ★★★★★ |
| 逻辑回归 | ★★★★★ | P0 | ★★★★★ |
| SVM | ★★★★☆ | P1 | ★★★★☆ |
| 决策树 | ★★★★☆ | P1 | ★★★★☆ |
| KNN | ★★★☆☆ | P2 | ★★☆☆☆ |

---

## 算法速查

| 算法 | 分类 | 回归 | 面试重点 |
|------|------|------|---------|
| 线性回归 | ✗ | ✓ | 推导/正则化 |
| 逻辑回归 | ✓ | ✗ | 梯度推导/softmax |
| SVM | ✓ | ✗ | 对偶问题/核函数 |
| 决策树 | ✓ | ✓ | 分裂准则/剪枝 |
| KNN | ✓ | ✓ | 距离度量/k选择 |

---

## 线性回归

### 原理

```python
# 模型：y = X @ w + b
# 损失：MSE = (1/n) * ||y - Xw||²
# 解：w* = (X^T X)^-1 X^T y
```

### sklearn实现

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# 普通线性回归
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 正则化
model = Ridge(alpha=1.0)  # L2正则化
model = Lasso(alpha=1.0)   # L1正则化
model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # L1+L2
```

### 手推梯度

```python
import numpy as np

def linear_regression_gradient(X, y, w, b, lr=0.01):
    """梯度下降"""
    n = len(y)
    pred = X @ w + b
    error = pred - y

    dw = (2/n) * X.T @ error  # w的梯度
    db = (2/n) * error.sum()  # b的梯度

    w -= lr * dw
    b -= lr * db
    return w, b
```

---

## 逻辑回归

### 原理

```python
# 模型：p = sigmoid(X @ w + b)
# 损失：Binary Cross Entropy = -[y log p + (1-y) log(1-p)]
```

### sklearn实现

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
pred_proba = model.predict_proba(X_test)
```

### 手推梯度

```python
def logistic_gradient(X, y, w, b, lr=0.01):
    """逻辑回归梯度"""
    n = len(y)
    z = X @ w + b
    pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # 数值稳定

    error = pred - y
    dw = (1/n) * X.T @ error
    db = (1/n) * error.sum()

    w -= lr * dw
    b -= lr * db
    return w, b
```

---

## SVM

### 原理

```python
# 硬间隔：min ||w||²/2  s.t. y_i(wx_i + b) >= 1
# 软间隔：min ||w||²/2 + C Σξ_i  s.t. y_i(wx_i + b) >= 1 - ξ_i
```

### sklearn实现

```python
from sklearn.svm import SVC, LinearSVC

# 核SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# 线性SVM（更快）
model = LinearSVC(C=1.0, max_iter=10000)
```

### 核函数

```python
# 线性核：K(x, z) = x^T z
# 多项式核：K(x, z) = (γ x^T z + r)^d
# RBF核：K(x, z) = exp(-γ ||x - z||²)
```

---

## 决策树

### 原理

```python
# 分裂准则：
# - Gini不纯度：1 - Σ p_k²
# - 信息增益：IG = H(parent) - Σ (N_child/N) * H(child)
# - MSE（回归）：方差减少量
```

### sklearn实现

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 分类
model = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
model.fit(X_train, y_train)

# 回归
model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5)
```

---

## 面试高频问题

### Q1: 逻辑回归和SVM的区别？

| 方面 | 逻辑回归 | SVM |
|------|---------|-----|
| 损失函数 | 交叉熵 | Hinge loss |
| 输出 | 概率 | 分类边界 |
| 对异常值 | 敏感 | 鲁棒 |
| 扩展性 | 好 | 差（核SVM慢） |

---

### Q2: L1和L2正则化的区别？

**答**：
- L1产生稀疏解（特征选择）
- L2使参数都趋近于零但不为零（训练更稳定）

---

## 相关知识点

- → [集成学习](../集成学习/README.md) - 随机森林/XGBoost
- → [无监督学习](../无监督学习/README.md) - 聚类/降维
