# 知识点卡片：XGBoost

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | XGBoost (Extreme Gradient Boosting) |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 8小时 |
| 面试频率 | ★★★★★ |

---

## 核心原理

XGBoost是Gradient Boosting的高效实现：

```
目标函数：L = Σ l(y_i, ŷ_i) + Σ Ω(f_k)

其中 Ω(f) = γT + ½λ‖w‖²（正则化项）
      T = 叶子节点数
      w = 叶子权重

新树 f_t 最小化：
L̃(t) = Σ [g_i f_t(x_i) + ½h_i f_t²(x_i)] + γT + ½λ Σ w_j²

其中 g_i = ∂l/∂ŷ(t-1), h_i = ∂²l/∂ŷ(t-1)（一阶和二阶梯度）
```

---

## sklearn实现

```python
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

# 分类
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,         # 行采样
    colsample_bytree=0.8,  # 列采样
    reg_alpha=0,           # L1正则化
    reg_lambda=1,          # L2正则化
    gamma=0,               # 最小损失减少（越大越保守）
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=10,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 特征重要性
importance = model.feature_importances_
for i, imp in enumerate(importance):
    print(f"Feature {i}: {imp:.4f}")
```

---

## 关键参数

| 参数 | 作用 | 推荐范围 |
|------|------|---------|
| n_estimators | 树数量 | 100-1000 |
| max_depth | 树深度 | 3-10 |
| learning_rate | 学习率 | 0.01-0.3 |
| subsample | 行采样比例 | 0.6-1.0 |
| colsample_bytree | 列采样比例 | 0.6-1.0 |
| reg_lambda | L2正则化 | 0-10 |
| reg_alpha | L1正则化 | 0-10 |
| gamma | 分裂最小损失减少 | 0-5 |
| min_child_weight | 叶节点最小权重和 | 1-10 |

---

## XGBoost vs LightGBM vs CatBoost

| 方面 | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| 树生长策略 | Level-wise | Leaf-wise | Symmetric |
| 类别特征 | 需编码 | 原生支持 | 原生支持(最好) |
| 速度 | 中等 | 快 | 中等 |
| 内存 | 中等 | 低 | 较高 |
| 缺失值 | 自动处理 | 自动处理 | 自动处理 |
| GPU支持 | 好 | 好 | 好 |

---

## 面试高频问题

### Q1: XGBoost为什么使用二阶梯度？

**答**：
XGBoost在目标函数中使用了泰勒二阶展开：
```
f(x+Δx) ≈ f(x) + f'(x)Δx + ½f''(x)Δx²
```
二阶信息可以：
1. 更精确地估计分裂增益
2. 天然支持自定义损失函数（只需一阶+二阶导）
3. 相比GBDT只使用一阶梯度的残差近似更为准确

### Q2: XGBoost的正则化体现在哪里？

**答**：
1. **γT**：叶节点数惩罚（越大树越简单）
2. **λ‖w‖²**：叶权重L2正则化
3. **subsample / colsample**：类似于随机森林的Bagging策略
4. **shrinkage (learning_rate)**：学习率衰减

### Q3: 为什么XGBoost这么受欢迎？

**答**：
- 效果非常好（Kaggle竞赛多年霸榜）
- 训练速度快（多种工程优化）
- 自动处理缺失值
- 内置交叉验证和早停
- 特征重要性可解释

---

## 相关知识点

- → [LightGBM](./知识点-LightGBM.md) - 对比
- → [Bagging与Boosting](./知识点-Bagging与Boosting.md) - 集成原理
- → [决策树](../监督学习/知识点-决策树.md) - 基学习器
