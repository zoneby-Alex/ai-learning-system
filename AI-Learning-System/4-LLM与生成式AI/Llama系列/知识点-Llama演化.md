# 知识点卡片：Llama系列演化

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | Llama系列 (Llama 1→2→3→4) |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 4小时 |
| 面试频率 | ★★★★★ |

---

## 演化路线

```
Llama 1(2023.02) → Llama 2(2023.07) → Llama 3(2024.04) → Llama 4(2025)
```

---

## 各代核心创新

### Llama 1 (2023)
- 参数：7B / 13B / 33B / 65B
- **核心发现**：更小的模型+更多的数据 > 更大的模型+更少的数据
- 1T tokens训练，超越GPT-3(175B)
- 使用：Pre-Norm, SwiGLU, RoPE

### Llama 2 (2023)
- 参数：7B / 13B / 70B
- **核心**：开源+商用许可 → 引爆开源LLM生态
- 2T tokens训练
- 引入GQA (Grouped Query Attention) 加速推理
- 70B版本加入RLHF对齐

### Llama 3 (2024)
- 参数：8B / 70B (初期)，405B (后续)
- **核心**：超大规模训练数据(15T+ tokens)
- 8192个token的词汇表（更大更高效）
- 128K上下文长度
- 8B模型使用GQA，70B也使用GQA

### Llama 4 (2025)
- MoE架构的探索
- 更强的多语言和多模态能力

---

## 技术亮点

### RoPE (Rotary Position Embedding)

```python
"""
Llama使用RoPE代替绝对位置编码：
- 通过旋转矩阵施加位置信息
- 自然编码相对位置
- 外推性好，支持更长上下文
"""
```

### SwiGLU 激活函数

```python
# SwiGLU(x) = (xW_1 ⊙ SiLU(xW_2)) W_3
# 其中 SiLU(x) = x * sigmoid(x)
# 比ReLU/GELU在LLM中效果更好

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

### GQA (Grouped Query Attention)

```python
"""
MHA: Q、K、V各有H个头
MQA: Q有H个头，K、V只有1个头（共享）
GQA: Q有H个头，K、V有G个头（1 < G < H）

GQA = MHA的速度 + MQA的质量折中
Llama 3 8B用GQA，推理更快
"""
```

---

## 面试高频问题

### Q1: Llama和GPT的主要区别？

**答**：

| 方面 | GPT | Llama |
|------|-----|-------|
| 开放 | 闭源（GPT-3后） | 开源 |
| 生态 | API调用 | 自由部署/微调 |
| 激活函数 | GELU | SwiGLU |
| 位置编码 | Learned | RoPE |
| 归一化 | Post-LN→Pre-LN | Pre-Norm(RMSNorm) |
| 注意力 | MHA | GQA(Llama2+) |

### Q2: 为什么Llama选择SwiGLU而非GELU？

**答**：
SwiGLU在多个LLM训练实验中表现优于GELU和ReLU。其"门控"机制提供了选择性信息流动：
- 一部分计算激活强度
- 另一部分计算门控系数
- 两者相乘决定输出
这使得网络能学会"选择性传递"信息。

---

## 相关知识点

- → [GPT演化](../GPT系列/知识点-GPT演化.md) - 闭源路线
- → [国产模型](../国内模型/知识点-国产模型对比.md) - 中文生态
