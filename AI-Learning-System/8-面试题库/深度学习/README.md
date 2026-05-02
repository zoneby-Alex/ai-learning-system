# 深度学习面试题库

本模块收集深度学习方向的高频面试题。

---

## 1. Transformer

### Q1: 画出Transformer架构图

**答**：
```
输入
  ↓
Embedding + 位置编码
  ↓
N× Encoder Block
  ├─ Multi-Head Self-Attention
  ├─ Add & Norm
  ├─ Feed Forward
  └─ Add & Norm
  ↓
N× Decoder Block
  ├─ Masked Multi-Head Self-Attention
  ├─ Add & Norm
  ├─ Multi-Head Cross-Attention
  ├─ Add & Norm
  ├─ Feed Forward
  └─ Add & Norm
  ↓
Linear + Softmax
  ↓
输出
```

### Q2: Self-Attention计算过程

**答**：
```python
# Q = X @ W_q
# K = X @ W_k
# V = X @ W_v
# Attention = softmax(Q @ K.T / √d_k) @ V
```

---

## 2. 优化器

### Q3: Adam优化器公式

**答**：
```
m_t = β₁m_{t-1} + (1-β₁)g_t        # 动量
v_t = β₂v_{t-1} + (1-β₂)g_t²      # RMSProp
m̂ = m_t / (1-β₁^t)                  # 偏差修正
v̂ = v_t / (1-β₂^t)                  # 偏差修正
θ_{t+1} = θ_t - lr * m̂ / (√v̂ + ε)
```

---

## 3. 正则化

### Q4: BatchNorm和LayerNorm的区别

| 方面 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | Batch | Feature |
| 依赖batch | 是 | 否 |
| 适用场景 | CV | NLP |

---

## 4. Attention变体

### Q5: MHA vs GQA vs MQA

| 方面 | MHA | MQA | GQA |
|------|-----|-----|-----|
| K/V头数 | H | 1 | G (< H) |
| Q头数 | H | H | H |
| 显存 | 高 | 低 | 中 |
| 效果 | 好 | 差 | 接近MHA |

---

## 更多题目

- BP反向传播推导
- 梯度消失/爆炸原因和解决方案
- 残差连接作用
- Dropout原理
- 激活函数对比

---

## 相关知识点

- → [BP反向传播](../3-深度学习核心/BP反向传播/README.md)
- → [Self-Attention](../3-深度学习核心/Transformer/知识点-Self-Attention.md)
