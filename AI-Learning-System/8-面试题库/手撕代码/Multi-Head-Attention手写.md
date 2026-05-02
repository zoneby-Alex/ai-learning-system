# 手撕代码：Multi-Head Attention

## 面试要求

```
难度：★★★★☆
要求：能在5分钟内完成
评分：手写无错 + 能解释每一步
```

---

## 标准答案

### 版本1: 最简洁版（面试快速写）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        # 合并为一个大线性层（效率）
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # 1. 计算Q, K, V
        qkv = self.W_qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dk)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 2. 计算注意力分数
        scores = Q @ K.transpose(-2, -1) / self.scale  # (B, H, T, T)

        # 3. Mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax
        attn = F.softmax(scores, dim=-1)

        # 5. 加权求和
        context = attn @ V  # (B, H, T, Dk)

        # 6. 合并多头
        context = context.transpose(1, 2).reshape(B, T, C)

        return self.W_o(context), attn
```

---

### 版本2: 分步注释版（更详细）

```python
class MultiHeadAttentionDetailed(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 初始化QKV和输出的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """将embedding分成多个头"""
        B, T, C = x.shape
        x = x.reshape(B, T, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (B, H, T, Dk)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)

        # Step 1: 线性变换
        Q = self.split_heads(self.W_q(Q))  # (B, H, T, Dk)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Step 2: 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, T, T)
        scores = scores / math.sqrt(self.d_k)  # 缩放

        # Step 3: 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step 4: Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Step 5: 加权求和
        context = torch.matmul(attn_weights, V)  # (B, H, T, Dk)

        # Step 6: 合并多头
        context = context.transpose(1, 2).contiguous()  # (B, T, H, Dk)
        context = context.reshape(B, -1, self.d_model)  # (B, T, C)

        return self.W_o(context), attn_weights
```

---

### 版本3: 仅QKV分离版（最容易理解）

```python
def multi_head_attention(Q, K, V, num_heads):
    """
    Q: (batch, seq_len, d_model)
    K: (batch, seq_len, d_model)
    V: (batch, seq_len, d_model)
    """
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # 1. Linear投影
    W_q = nn.Linear(d_model, d_model)
    W_k = nn.Linear(d_model, d_model)
    W_v = nn.Linear(d_model, d_model)
    W_o = nn.Linear(d_model, d_model)

    Q = W_q(Q)  # (B, T, C)
    K = W_k(K)
    V = W_v(V)

    # 2. Reshape: (B, T, C) -> (B, T, H, Dk) -> (B, H, T, Dk)
    Q = Q.reshape(batch, seq_len, num_heads, d_k).permute(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_k).permute(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).permute(0, 2, 1, 3)

    # 3. 注意力计算
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, T, T)
    attn = F.softmax(scores, dim=-1)
    context = attn @ V  # (B, H, T, Dk)

    # 4. 合并多头: (B, H, T, Dk) -> (B, T, H, Dk) -> (B, T, C)
    context = context.transpose(1, 2).reshape(batch, seq_len, d_model)

    return W_o(context)
```

---

## 面试问答

### Q1: 为什么要除以 √d_k？

**答**：
- 防止点积结果过大，使softmax进入饱和区
- 假设Q、K每个元素独立同分布，均值0方差1，则Q·K的方差为d_k
- 除以√d_k使点积的方差恢复到1，梯度更稳定

---

### Q2: Multi-Head的作用是什么？

**答**：
- 允许每个头关注不同类型的信息（如语法、语义、位置）
- 增加模型的表示能力
- 相比单头，参数量相近但效果更好

---

### Q3: 复杂度是多少？

**答**：
```
时间复杂度：O(N² · d · h) = O(N² · C)
- 每个头的维度是d_k = d/h
- QK^T: (N, d_k) @ (d_k, N) = O(N² · d_k) per head
- h heads: O(N² · d)

空间复杂度：O(N² · h) 用于存储attention matrix
```

---

### Q4: 如何处理mask？

**答**：
```python
# Padding mask（处理不同长度）
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

# Future mask（Decoder中防止看到未来）
future_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
scores = scores.masked_fill(future_mask, -1e9)
```

---

## 常见错误

```python
# 错误1: 没有缩放
scores = Q @ K.transpose(-2, -1)  # ❌

# 错误2: 多头合并方式错误
context = context.reshape(B, T, C)  # ❌ reshape顺序错了

# 错误3: mask位置错误
scores = scores.masked_fill(mask == 0, -1e9)  # mask可能是(B,T)而不是(B,T,T)

# 正确做法：
# mask应该是 (B, T, T) 或 (B, 1, T) 可以broadcast
scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)  # ✓
```

---

## 验证代码

```python
import torch

# 创建模型
d_model = 256
num_heads = 8
model = MultiHeadAttention(d_model, num_heads)

# 测试
x = torch.randn(2, 10, 256)  # (batch, seq_len, d_model)
output, attn = model(x)

print(f"Output shape: {output.shape}")  # (2, 10, 256)
print(f"Attention shape: {attn.shape}")  # (2, 8, 10, 10)
print(f"Attention sum: {attn[0, 0].sum().item():.4f}")  # 应该≈1.0
```
