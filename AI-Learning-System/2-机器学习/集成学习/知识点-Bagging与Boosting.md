# 知识点卡片：Bagging与Boosting

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | Bagging、Boosting与集成学习 |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 6小时 |
| 面试频率 | ★★★★★ |

---

## 核心原理

### Bagging (Bootstrap Aggregating)

并行训练多个独立基学习器，通过投票/平均聚合：

```
方法：
1. 从训练集有放回采样B个子集
2. 在每个子集上训练基学习器
3. 分类：多数投票；回归：取平均

代表：随机森林 (Random Forest)
```

### Boosting

串行训练多个弱学习器，每个纠正前一个的错误：

```
方法：
1. 训练第一个弱学习器
2. 根据错误调整样本权重
3. 训练下一个弱学习器
4. 加权投票/求和

代表：AdaBoost, GBDT, XGBoost, LightGBM
```

---

## 核心区别

| 方面 | Bagging | Boosting |
|------|---------|----------|
| 训练方式 | 并行 | 串行 |
| 基学习器 | 强（深度树） | 弱（浅树） |
| 偏差 | 与基学习器相同 | 逐渐降低 |
| 方差 | 显著降低 | 相对较低 |
| 主要解决 | 过拟合（高方差） | 欠拟合（高偏差） |
| 过拟合风险 | 低 | 中（可加正则化） |
| 代表算法 | Random Forest | XGBoost/LightGBM |

---

## Bias-Variance分解

```python
"""
预期泛化误差 = Bias² + Variance + Noise

Bagging：主要减少Variance
- 独立模型的平均 → 方差降低 √B 倍
- Bias不变（基学习器相同）
- 适合高Variance的模型（深决策树）

Boosting：主要减少Bias
- 从弱学习器逐步加强 → 偏差降低
- 串行训练 → 方差不会显著降低
- 适合高Bias的模型（浅决策树）
"""
```

---

## 代码实现

### Bagging

```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=100,
    max_samples=0.8,  # Bootstrap采样比例
    bootstrap=True,   # 有放回采样
    n_jobs=-1
)

# Random Forest = Bagging + 随机特征选择
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',  # 关键！每次分裂随机选特征
    n_jobs=-1
)
```

### AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 决策树桩
    n_estimators=100,
    learning_rate=0.1
)
```

---

## 面试高频问题

### Q1: 随机森林中"随机"体现在哪里？

**答**：
1. **样本随机**：Bootstrap采样（每棵树看到的样本不同）
2. **特征随机**：每次分裂时只从随机子集（√d个）中选择最佳特征
3. 这两重随机性使树之间独立性强，方差降低更多

### Q2: GBDT、XGBoost和LightGBM的关系？

**答**：
- **GBDT**：Gradient Boosting的基础算法，用一阶梯度的残差近似
- **XGBoost**：在GBDT基础上加入二阶梯度、正则化、并行化
- **LightGBM**：在GBDT基础上加入GOSS采样、EFB压缩、leaf-wise生长

三者都是Boosting算法，逐步改进工程效率和效果。

### Q3: Bagging为什么能减少方差？

**答**：
设B个独立基学习器的方差都为σ²，则均值的方差为：
```
Var(平均值) = Var((1/B) Σ f_i) = σ²/B（如果完全独立）

实际不完全独立（但Bagging尽力使其独立），所以：
σ²/B < Var(平均值) < σ²
```
这就是Bagging降低方差的数学基础。

---

## 相关知识点

- → [随机森林](./README.md) - Bagging代表
- → [XGBoost](./知识点-XGBoost.md) - Boosting代表
- → [决策树](../监督学习/知识点-决策树.md) - 基学习器
