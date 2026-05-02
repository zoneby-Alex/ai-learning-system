# 知识点卡片：Self-Attention 与 Multi-Head Attention

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | Self-Attention / Multi-Head Attention |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 面试必考 | 是 |
| 必须手写 | 是 |

---

## 核心原理

Self-Attention（自注意力）通过Query-Key-Value机制实现序列内各位置的交互：

```
每个位置 i 对所有位置 j 计算注意力：
- Query (Q): 我想查询什么
- Key (K): 我包含什么信息
- Value (V): 我提供什么信息

注意力分数 = Q · K^T / √d_k
注意力权重 = softmax(注意力分数)
输出 = 注意力权重 · V
```

---

## 数学公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

---

## PyTorch实现

### 1. 标准Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """标准Self-Attention"""
    def __init__(self, d_model):
        super().__init__()
        self.d_k = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, C = x.shape

        # 线性变换得到Q, K, V
        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力分数
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        # (B, T, T) @ (B, T, C) -> (B, T, C)
        context = torch.matmul(attn_weights, V)

        # 输出变换
        output = self.W_o(context)
        return output, attn_weights
```

### 2. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 合并为一个大线性层（效率更高）
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # 同时计算Q, K, V
        qkv = self.W_qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dk)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = attn @ V  # (B, H, T, Dk)

        # 合并多头
        context = context.transpose(1, 2).reshape(B, T, C)
        return self.W_o(context), attn
```

### 3. Flash Attention（简化版原理）

```python
def flash_attention(Q, K, V, scale=1.0):
    """
    Flash Attention的核心思想：分块计算 + 数值稳定
    实际实现需要CUDA kernel，这里展示原理
    """
    # 标准attention: O(N²) 显存
    # Flash attention: O(N) 显存，通过分块避免 materialize 整个 attention matrix

    B, H, N, D = Q.shape
    block_size = 64

    # 分块计算
    output = torch.zeros_like(Q)

    # 只需保存 row_max 和 row_sum，实现 online softmax
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            # 计算当前块
            q_block = Q[..., i:i+block_size, :]
            k_block = K[..., j:j+block_size, :]
            v_block = V[..., j:j+block_size, :]

            # ... 分块注意力计算 ...
            # 利用数值稳定的 online softmax
            pass

    return output
```

---

## 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 标准Attention | O(N²·d) | O(N²) |
| Flash Attention | O(N²·d) | O(N·d) |

```
N = 序列长度
d = 每头维度
h = 注意力头数

标准Attention需要保存 N×N 的注意力矩阵 → O(N²) 空间
Flash Attention通过分块计算，只需保存对角线元素 → O(N) 空间
```

---

## 面试高频问题

### Q1: Self-Attention的计算过程（手写）

**答**：
```python
def attention(Q, K, V):
    # 1. 计算注意力分数
    scores = Q @ K.T / math.sqrt(d_k)  # (T, T)

    # 2. softmax归一化
    attn_weights = F.softmax(scores, dim=-1)  # (T, T)

    # 3. 加权求和
    output = attn_weights @ V  # (T, d)

    return output, attn_weights
```

---

### Q2: 为什么要除以 √d_k？

**答**：
1. **防止梯度消失**：当d_k较大时，Q·K的点积可能很大，使softmax进入饱和区，梯度接近0
2. **方差归一化**：假设Q和K的每个元素是独立随机变量，均值为0，方差为1，则Q·K的方差为d_k，除以√d_k使其方差恢复到1

---

### Q3: Multi-Head Attention vs Single Head？

**答**：

| 方面 | Single Head | Multi-Head |
|------|-------------|------------|
| 表示子空间 | 只有一个 | 多个 |
| 并行计算 | 简单 | 可并行 |
| 参数量 | 较少 | 略多（但每头维度减小） |
| 效果 | 可能欠拟合 | 通常更好 |

```
Multi-Head的本质：让每个头学习不同的表示子空间

例如：
- 头1可能关注语法结构
- 头2可能关注语义相似性
- 头3可能关注位置信息
```

---

### Q4: Self-Attention vs RNN/LSTM？

| 方面 | Self-Attention | RNN/LSTM |
|------|---------------|----------|
| 距离 | O(1) | O(N)（距离远梯度消失） |
| 并行 | 易并行 | 难并行（顺序依赖） |
| 捕获依赖 | 全局 | 局部（长期依赖困难） |
| 计算量 | O(N²·d) | O(N·d) |
| 内存 | O(N²) | O(N·d) |

---

### Q5: Pre-LN vs Post-LN？

**答**：
```
Post-LN: 
    x → Attention → + → LN → FFN → + → LN → output
    (层Norm在残差之后，更难训练，需要warmup)

Pre-LN:
    x → LN → Attention → + → LN → FFN → + → output
    (层Norm在残差之前，更稳定，已成主流)
```

| 方面 | Post-LN | Pre-LN |
|------|---------|--------|
| 训练稳定性 | 差（需要warmup） | 好 |
| 理论效果 | 可能更好 | 略差 |
| 实际使用 | 少 | 主流（Llama/Qwen等都用Pre-LN） |

---

## 代码练习

```python
import torch

# 验证Multi-Head Attention实现
d_model = 256
num_heads = 8
batch_size = 2
seq_len = 10

x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)

output, attn = mha(x)
print(f"Output shape: {output.shape}")  # (2, 10, 256)
print(f"Attention shape: {attn.shape}")  # (2, 8, 10, 10)

# 检查注意力权重
print(f"Attention weights sum: {attn[0, 0].sum().item():.4f}")  # 应接近1.0
```

---

## 相关知识点

- → [位置编码](./知识点-位置编码.md) - 位置信息的注入
- → [LayerNorm](../正则化/LayerNorm.md) - 归一化层
- → [FFN](./知识点-FeedForward.md) - 前馈网络
- → [Flash Attention](./知识点-FlashAttention.md) - 注意力优化
