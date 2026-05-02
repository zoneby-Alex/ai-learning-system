# 项目：IMDB 文本情感分类

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐☆☆☆ |
| 预估时间 | 1-2天 |
| 学习目标 | RNN/Transformer文本分类 |
| 技术栈 | LSTM / Transformer + torchtext |

---

## 项目目标

1. 掌握文本分类完整流程
2. 实现LSTM和Transformer两种模型
3. 对比RNN vs Transformer效果
4. 实现位置编码

---

## 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# LSTM模型
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden)).squeeze()

# Transformer模型
class TransformerSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1)).squeeze()

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## 验收标准

- [ ] 测试准确率 > 85%
- [ ] 实现了LSTM和Transformer两种模型
- [ ] 能解释两种模型的效果差异
- [ ] 实现了位置编码

---

## LSTM vs Transformer 对比要点

| 方面 | LSTM | Transformer |
|------|------|-------------|
| 训练速度 | 慢（顺序） | 快（并行） |
| 长文本 | 梯度消失问题 | 全局注意力 |
| 参数量 | 较少 | 较多 |
| 准确率 | 略低 | 通常更高 |
