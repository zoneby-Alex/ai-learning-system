# RNN家族模块

本模块包含循环神经网络（RNN）的核心知识点。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| Vanilla RNN | ★★★☆☆ | P1 | ★★★☆☆ |
| LSTM | ★★★★★ | P0 | ★★★★★ |
| GRU | ★★★★☆ | P1 | ★★★★☆ |
| seq2seq | ★★★★☆ | P1 | ★★★★☆ |
| 梯度消失/爆炸 | ★★★★☆ | P0 | ★★★★☆ |

---

## RNN原理

### 标准RNN

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x, h=None):
        # x: (batch, input_size)
        # h: (batch, hidden_size)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size)
        h = self.tanh(self.W_xh(x) + self.W_hh(h))
        return h, h
```

### 问题：梯度消失/爆炸

```
RNN的梯度：
∂h_t/∂h_{t-1} = W^T · diag(tanh'(z))

当tanh'(z) ∈ (0,1)时：
- 若|W|>1，梯度指数增长 → 梯度爆炸
- 若|W|<1，梯度指数衰减 → 梯度消失

LSTM通过门控机制解决这个问题
```

---

## LSTM

### 核心：门控机制

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 遗忘门、输入门、输出门
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h, c = state
        # 一次性计算所有门
        gates = self.W(torch.cat([x, h], dim=1))
        # 分割为4份
        f, i, o, g = gates.chunk(4, dim=1)
        # LSTM核心公式
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, (h_next, c_next)
```

### LSTM门的作用

| 门 | 公式 | 作用 |
|----|------|------|
| 遗忘门 f | σ(W_f·[h_{t-1}, x_t]) | 决定丢弃什么信息 |
| 输入门 i | σ(W_i·[h_{t-1}, x_t]) | 决定添加什么新信息 |
| 输出门 o | σ(W_o·[h_{t-1}, x_t]) | 决定输出什么 |

---

## GRU

### 简化版LSTM

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        gates = self.W(torch.cat([x, h], dim=1))
        r, z, g = gates.chunk(3, dim=1)
        # 重置门、更新门、候选隐藏状态
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        g = torch.tanh(r * h + g)
        h_next = (1 - z) * h + z * g
        return h_next
```

### LSTM vs GRU

| 方面 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 记忆单元 | 有 | 无 |
| 参数量 | 较多 | 较少 |
| 效果 | 通常更好 | 在小数据上可能更好 |

---

## seq2seq

### 编码器-解码器架构

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # encoder
        encoder_outputs, hidden = self.encoder(src)

        # decoder
        decoder_hidden = hidden
        decoder_outputs = []
        decoder_input = tgt[:, 0:1]  # <start>

        for t in range(1, tgt.size(1)):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            decoder_outputs.append(output)

            # teacher forcing
            if random.random() < teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1]
            else:
                decoder_input = output.argmax(dim=-1)

        return torch.cat(decoder_outputs, dim=1)
```

---

## 面试高频问题

### Q1: RNN为什么容易梯度消失/爆炸？

**答**：
RNN中相同权重矩阵被循环共享，导致：
- ∂h_t/∂h_{t-1} = W^T · diag(tanh'(z))
- 长期依赖时，梯度需要连乘W多次
- W的奇异值决定梯度变化：>1爆炸，<1消失

---

### Q2: LSTM如何解决梯度消失？

**答**：
通过门控机制：
- 遗忘门f控制信息保留比例
- 当f≈1时，c_t ≈ c_{t-1}，梯度可以直接传回
- 关键：c的路径是线性的，不像tanh一样有压缩

---

### Q3: Transformer相比RNN的优势？

| 方面 | RNN | Transformer |
|------|-----|-------------|
| 距离 | O(N)，距离远梯度消失 | O(1)，注意力全局 |
| 并行 | 顺序依赖 | 高度并行 |
| 捕获依赖 | 困难（长期） | 容易 |
| 内存 | O(N) | O(N²) |

---

## 相关知识点

- → [Attention](../Transformer/README.md) - 替代RNN的方式
- → [BP反向传播](../BP反向传播/README.md) - 梯度计算
