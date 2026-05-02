# 知识点卡片：LightGBM

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | LightGBM |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 5小时 |
| 面试频率 | ★★★★★ |

---

## 核心创新

### 1. Leaf-wise 树生长（vs Level-wise）

```
XGBoost (Level-wise)：逐层生长，同一层所有节点都分裂
  → 浪费计算（有些节点增益很小也分裂）

LightGBM (Leaf-wise)：选择增益最大的叶子先分裂
  → 同样叶子数下loss更低
  → 但可能过深，需要max_depth限制
```

### 2. GOSS (Gradient-based One-Side Sampling)

保留大梯度样本，随机采样小梯度样本：
- 大梯度样本 → 全部保留（信息量大）
- 小梯度样本 → 按比例随机采样
- 在采样上计算增益时乘上放大系数补偿

### 3. EFB (Exclusive Feature Bundling)

将互斥的特征（不同时取非零值）捆绑在一起，减少特征数量，加速训练。

### 4. 直方图算法

将连续特征离散化到bins（如256个桶），在bins上计算分裂，大幅减少计算量。

---

## sklearn实现

```python
from lightgbm import LGBMClassifier, LGBMRegressor

model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    num_leaves=31,          # 叶子节点数（关键参数！）
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=20,   # 叶节点最小样本数
    verbose=-1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

---

## 面试高频问题

### Q1: num_leaves和max_depth的关系？

**答**：
在leaf-wise生长下，树可能很深。num_leaves控制叶子数，max_depth限制深度。
- 一棵深度为d的完全二叉树有 2^d 个叶节点
- num_leaves ≈ 2^max_depth
- 实践中 num_leaves 不要设为 2^max_depth，会导致过拟合
- 建议：num_leaves < 2^max_depth，如 depth=6, leaves=31

### Q2: LightGBM比XGBoost快在哪里？

**答**：
1. **直方图算法**：离散化后寻找最佳分裂点是O(#bin)而非O(#data)
2. **Leaf-wise生长**：计算更少的分裂
3. **GOSS采样**：减少需要计算梯度的样本
4. **EFB**：减少特征数量
5. **内存优化**：离散特征只需存uint8

### Q3: 什么时候选LightGBM而非XGBoost？

- 数据量大（>10万行）→ LightGBM更快
- 有大量类别特征 → LightGBM原生支持
- 需要快速迭代 → LightGBM
- 需要更稳定的结果 → XGBoost（level-wise更保守）

---

## 相关知识点

- → [XGBoost](./知识点-XGBoost.md) - 对比
- → [Bagging与Boosting](./知识点-Bagging与Boosting.md) - 集成原理
