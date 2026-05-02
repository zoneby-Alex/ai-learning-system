# 机器学习模块

本模块包含机器学习的核心算法和原理。

## 目录结构

```
2-机器学习/
├── README.md                # 本文件
├── 监督学习/                # 分类/回归
├── 无监督学习/              # 聚类/降维
└── 集成学习/               # Bagging/Boosting
```

---

## 学习路径

```
机器学习
├── 监督学习
│   ├── 线性回归/逻辑回归 ★★★★★
│   ├── SVM ★★★★☆
│   ├── 决策树 ★★★★☆
│   └── KNN ★★★☆☆
│
├── 无监督学习
│   ├── KMeans ★★★★☆
│   ├── PCA ★★★★☆
│   ├── t-SNE ★★★☆☆
│   └── GMM ★★★☆☆
│
└── 集成学习
    ├── Bagging ★★★★☆
    ├── Random Forest ★★★★☆
    ├── Boosting ★★★★★
    └── XGBoost/LightGBM ★★★★★
```

---

## 算法速查表

| 算法 | 类型 | 损失函数 | 正则化 | 面试频率 |
|------|------|---------|--------|---------|
| 线性回归 | 监督-回归 | MSE | L1/L2 | ★★★★☆ |
| 逻辑回归 | 监督-分类 | CE | L1/L2 | ★★★★★ |
| SVM | 监督-分类 | Hinge | L2 | ★★★★☆ |
| 决策树 | 监督-两者 | Gini/Entropy | 剪枝 | ★★★★☆ |
| KMeans | 无监督-聚类 | 簇内平方和 | - | ★★★★☆ |
| PCA | 无监督-降维 | 方差最大化 | - | ★★★★☆ |
| XGBoost | 监督-两者 | 自定义 | L1/L2 | ★★★★★ |

---

## sklearn API速查

```python
# 监督学习
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier

# 无监督学习
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 集成学习
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 评估
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

---

## 与深度学习的关系

| ML方法 | DL对应/关系 | 面试重点 |
|--------|-------------|---------|
| 逻辑回归 | softmax回归 | 梯度推导 |
| SVM | 支持向量机→对比学习 | 对偶问题/核函数 |
| 决策树 | CART→GBDT | 信息增益/过拟合 |
| PCA | 自编码器 | 降维原理/方差最大化 |
| KMeans | VQ-VAE/GAN | 迭代优化/收敛性 |

---

## 面试高频考点

### 1. 逻辑回归
- Sigmoid函数推导
- 梯度下降推导
- 与softmax的关系

### 2. SVM
- 对偶问题
- 核函数
- 软间隔

### 3. XGBoost/LightGBM
- GBDT原理
- 梯度提升
- 正则化

### 4. 聚类/降维
- KMeans迭代过程
- PCA与SVD的关系
- t-SNE原理

---

## 学习建议

```
1. 先掌握sklearn API
2. 再理解算法原理
3. 最后手推公式
```

---

## 相关知识点

- → [数学基础](../1-数学基础/README.md) - 推导基础
- → [深度学习核心](../3-深度学习核心/README.md) - 神经网络基础
