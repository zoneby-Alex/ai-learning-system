# 集成学习模块

本模块包含集成学习的核心算法和原理。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| Bagging | ★★★★☆ | P0 | ★★★☆☆ |
| Boosting | ★★★★★ | P0 | ★★★★☆ |
| 随机森林 | ★★★★☆ | P1 | ★★★★☆ |
| XGBoost/LightGBM | ★★★★★ | P0 | ★★★★★ |
| Stacking | ★★★☆☆ | P1 | ★★★☆☆ |

---

## 集成方法对比

| 方法 | 基学习器 | 学习方式 | 减少偏差 | 减少方差 |
|------|---------|---------|----------|----------|
| Bagging | 强 | 并行 | ✗ | ✓ |
| Boosting | 弱 | 串行 | ✓ | ✗ |
| Random Forest | 决策树 | 并行 | ✗ | ✓ |
| XGBoost/LightGBM | 决策树 | 串行 | ✓ | ✓ |

---

## Bagging

### 原理

```python
# Bootstrap Aggregating
# 1. 有放回采样n个样本
# 2. 在每个样本上训练基学习器
# 3. 聚合预测（投票/平均）
```

### sklearn实现

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=100,
    bootstrap=True
)
model.fit(X_train, y_train)
```

---

## 随机森林

### 原理

```python
# Bagging + 随机特征选择
# 每棵树：
#   - 随机采样n个样本
#   - 每次分裂时随机选k个特征
#   - 选最优特征分裂
```

### sklearn实现

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',  # 随机选特征
    n_jobs=-1
)
model.fit(X_train, y_train)
```

---

## Boosting

### 原理

```python
# 串行集成，逐步修正错误
# 1. 训练第一个基学习器
# 2. 计算错误率
# 3. 增加错误样本的权重
# 4. 训练下一个基学习器
# 5. 加权投票/平均
```

### AdaBoost实现

```python
import numpy as np

def adaboost(X, y, n_estimators=10):
    """AdaBoost实现"""
    n = len(y)
    w = np.ones(n) / n  # 初始权重
    models = []
    alphas = []

    for t in range(n_estimators):
        # 训练决策树桩
        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y, sample_weight=w)
        pred = model.predict(X)

        # 计算错误率
        error = (w * (pred != y)).sum()
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

        # 更新权重
        w = w * np.exp(alpha * (pred != y) * 2)
        w = w / w.sum()

        models.append(model)
        alphas.append(alpha)

    return models, alphas
```

---

## XGBoost

### 原理

```python
# 目标函数：L = Σ l(y_i, pred_i) + Σ Ω(f_k)
#       = Σ l(y_i, pred_i) + Σ [0.5 * ||w||² + γT]
# 新树：pred_i^(t) = pred_i^(t-1) + f_t(x_i)
# 近似贪心最优分裂
```

### sklearn实现

```python
from xgboost import XGBClassifier, XGBRegressor

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic'
)
model.fit(X_train, y_train)
```

### 参数说明

```python
# 重要参数
n_estimators: 树的数量
max_depth: 树的最大深度
learning_rate: 学习率（ shrinkage）
subsample: 行采样比例
colsample_bytree: 列采样比例
min_child_weight: 叶子节点的最小权重和
gamma: 最小损失减少量
reg_alpha/lambda: L1/L2正则化
```

---

## LightGBM

### 原理

```python
# 基于梯度的单边采样（GOSS）
# 互斥特征捆绑（EFB）
# 叶子优先（Leaf-wise）而非深度优先（Level-wise）
```

### sklearn实现

```python
from lightgbm import LGBMClassifier, LGBMRegressor

model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)
model.fit(X_train, y_train)
```

---

## 面试高频问题

### Q1: XGBoost和LightGBM的区别？

| 方面 | XGBoost | LightGBM |
|------|---------|----------|
| 分裂策略 | 深度优先 | 叶子优先 |
| 分裂点 | 精确贪心 | 直方图近似 |
| 处理类别 | 需one-hot | 原生支持 |
| 速度 | 较慢 | 快 |
| 内存 | 较高 | 较低 |

---

### Q2: Bagging vs Boosting？

**答**：
- Bagging：并行训练，减少方差，基学习器独立
- Boosting：串行训练，减少偏差，基学习器相关

---

## 相关知识点

- → [监督学习](../监督学习/README.md) - 基学习器
- → [决策树](../监督学习/README.md) - 决策树分裂
