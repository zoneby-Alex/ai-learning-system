# Transformer 面试题集

---

## Q1: 画出Transformer架构并解释各组件

**答**：
```
Encoder                          Decoder
  ↓                                ↓
Input Embedding               Output Embedding
  ↓                                ↓
Positional Encoding           Positional Encoding
  ↓                                ↓
┌──────────────┐              ┌──────────────┐
│ Multi-Head   │              │ Masked MHA   │← 因果mask
│ Attention    │              │              │
├──────────────┤              ├──────────────┤
│ Add & Norm   │              │ Add & Norm   │
├──────────────┤              ├──────────────┤
│ Feed Forward │              │ Cross-Attn   │← 接收Encoder输出
├──────────────┤              ├──────────────┤
│ Add & Norm   │              │ Add & Norm   │
└──────────────┘ (×N)        ├──────────────┤
                              │ Feed Forward │
                              ├──────────────┤
                              │ Add & Norm   │
                              └──────────────┘ (×N)
                                   ↓
                              Linear + Softmax
```

---

## Q2: Self-Attention的计算过程（手写）

**答**：
```python
# 输入: x (B, T, C)
# 1. 线性变换
Q = x @ W_q  # (B, T, C)
K = x @ W_k
V = x @ W_v

# 2. 注意力分数
scores = Q @ K.T / sqrt(d_k)  # (B, T, T)

# 3. Softmax
attn = softmax(scores, dim=-1)

# 4. 加权求和
output = attn @ V  # (B, T, C)
```

---

## Q3: 为什么用Layer Norm而不是Batch Norm？

**答**：

| Batch Norm | Layer Norm |
|-----------|------------|
| 在batch维度归一化 | 在feature维度归一化 |
| 依赖batch统计量 | 独立于batch |
| 序列长度不一难处理 | 每个样本独立归一化 |
| 推理时需running stats | 推理时行为相同 |

NLP中序列长度不固定，Batch Norm效果不稳定。Layer Norm归一化在特征维度，与序列长度和batch size无关，更适合Transformer。

---

## Q4: Pre-LN vs Post-LN

```python
# Post-LN (原始Transformer)：LN在残差之后
x = LN(x + Attention(x))
# 梯度可能直接通过残差传到前面，但通过LN的部分受限制
# 训练难，需要warmup

# Pre-LN (Llama/Qwen等)：LN在残差之前
x = x + Attention(LN(x))
# 梯度有"高速公路"（残差连接），训练更稳定
# 现代LLM几乎都用Pre-LN
```

---

## Q5: MHA vs MQA vs GQA

| 方面 | MHA | MQA | GQA |
|------|-----|-----|-----|
| K/V头数 | H | 1 | G (1<G<H) |
| 显存 | 最高 | 最低 | 中等 |
| 质量 | 最好 | 略低 | 接近MHA |
| 代表 | GPT-3 | PaLM | Llama 3 |

GQA是MHA和MQA的折中，保持较好的质量同时大幅减少KV Cache显存。

---

## Q6: RoPE位置编码原理

**答**：
RoPE通过旋转矩阵施加位置信息：
- 将Q和K按位置m和n分别旋转
- QK^T的点积自动包含相对位置信息（m-n）
- 优点是自然编码相对位置，支持外推到训练长度之外

---

## Q7: Transformer的时间/空间复杂度

```
Self-Attention: O(N²·d) 时间, O(N²) 空间
FFN: O(N·d²) 时间

当 N << d（大多数情况）:
瓶颈是FFN，不是Attention

当 N >> d（长上下文）:
瓶颈是Attention的O(N²)
```
