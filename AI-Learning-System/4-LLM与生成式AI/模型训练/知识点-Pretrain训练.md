# 知识点卡片：LLM预训练

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | LLM预训练 (Tokenizer/Scaling Law/分布式) |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P0 |
| 预估时间 | 8小时 |
| 面试频率 | ★★★★☆ |

---

## 核心组件

### 1. Tokenizer (分词器)

```python
"""
主流分词方法：

BPE (Byte-Pair Encoding)：
- GPT-2/GPT-3/Llama使用
- 从字符开始，逐步合并高频字节对
- 平衡了词级和字符级的优势

WordPiece：
- BERT使用
- 类似BPE但基于概率合并

SentencePiece：
- T5/Llama使用
- 直接在原始文本上训练（不需要预分词）
- 支持byte-fallback

Unigram：
- 从大词汇表开始，逐步删除低频token
- 保持最优的分词质量
"""

# HuggingFace中查看tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
text = "深度学习是人工智能的核心技术"

tokens = tokenizer.tokenize(text)
print(f"分词结果: {tokens}")
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")
print(f"词汇表大小: {tokenizer.vocab_size}")
```

### 2. Scaling Law (扩展法则)

```
核心结论（Kaplan et al., 2020）：
L(N, D) = (N_c/N)^{α_N} + (D_c/D)^{α_D} + L_min

其中：
- N: 模型参数量
- D: 训练数据量
- L: 损失函数值

实际意义：
- 模型大小和训练数据量应同步增长
- 给定计算预算C，最优的N和D满足：N ∝ C^{0.73}, D ∝ C^{0.27}
- Chinchilla修正：N ∝ C^{0.5}, D ∝ C^{0.5}（数据和模型同等重要）

Chinchilla Optimal：
- 对于给定计算量，模型大小和数据量应均衡
- 大多数现有模型训练不足（数据量不够）
```

---

## 分布式训练

```python
"""
三种并行策略：

1. 数据并行 (DP/FSDP)：
   - 每个GPU持有完整模型副本
   - 数据分片到不同GPU
   - FSDP：进一步将模型参数和优化器状态分片

2. 张量并行 (Tensor Parallelism)：
   - 单个层的权重矩阵切分到多个GPU
   - 适合大矩阵乘法（如Attention的QKV投影）
   - 通信开销大，通常在节点内使用

3. 流水线并行 (Pipeline Parallelism)：
   - 不同层放在不同GPU上
   - 数据像流水线一样通过各GPU
   - 减少通信开销（只在层边界通信）

4. 混合并行：
   - 三者组合：PP + TP + DP
   - 例如Megatron-LM: TP在节点内，PP跨节点，DP复制
"""

# DeepSpeed配置示例
deepspeed_config = {
    "train_batch_size": 1024,
    "gradient_accumulation_steps": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,           # ZeRO-2: 优化器+梯度分片
        "offload_optimizer": {"device": "cpu"},
        "allgather_partitions": True,
    }
}
```

---

## 数据处理Pipeline

```
原始数据
  └─→ 去重（URL/文档级精确去重）
      └─→ 质量过滤（语言检测/困惑度过滤/规则过滤）
          └─→ 隐私过滤（PII检测/移除）
              └─→ Tokenization
                  └─→ Packing（将短文档拼接到序列长度）
                      └─→ 打乱 + Batching
```

---

## 面试高频问题

### Q1: BPE和WordPiece的区别？

| 方面 | BPE | WordPiece |
|------|-----|-----------|
| 合并策略 | 频率最高的一对 | 似然增量最大的一对 |
| 训练 | 统计字节对频率 | 基于语言模型概率 |
| 代表 | GPT-2/3, Llama | BERT |
| 差异 | 确定性的 | 有一定随机性 |

### Q2: Scaling Law的实际指导意义？

**答**：
1. 不要盲目扩大模型而忽略数据量（Chinchilla修正）
2. 给定预算时，可以通过公式估算最优的模型和数据配比
3. 小模型+多数据可能优于大模型+少数据
4. 解释了大模型涌现能力的原因：规模带来的质变

### Q3: FSDP vs DDP的区别？

| 方面 | DDP | FSDP |
|------|-----|------|
| 模型副本 | 每GPU完整副本 | 参数分片 |
| 显存效率 | 低 | 高 |
| 通信 | allreduce梯度 | allgather参数+reduce梯度 |
| 适用 | 小模型 | 大模型(>1B) |

---

## 相关知识点

- → [SFT与对齐](./知识点-SFT与对齐.md) - 后续微调
- → [推理优化](../推理优化/知识点-推理优化技术.md) - 部署
- → [Scaling Law论文](../../LLM必读论文清单.md)
