# LLM必读论文清单

## 按学习阶段分类

### 第一阶段：基础必读

| 序号 | 论文 | 年份 | 重要性 | 读后能回答 |
|------|------|------|--------|-----------|
| 1 | Attention Is All You Need | 2017 | ⭐⭐⭐⭐⭐ | Transformer架构 |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers | 2018 | ⭐⭐⭐⭐⭐ | BERT原理 |
| 3 | GPT-3: Language Models are Few-Shot Learners | 2020 | ⭐⭐⭐⭐⭐ | In-context Learning |
| 4 | Layer Normalization | 2016 | ⭐⭐⭐⭐ | LN原理 |
| 5 | RMSProp | 2012 | ⭐⭐⭐ | 优化器基础 |

---

### 第二阶段：LLM核心

| 序号 | 论文 | 年份 | 重要性 | 读后能回答 |
|------|------|------|--------|-----------|
| 6 | LLaMA: Open and Efficient Foundation Language Models | 2023 | ⭐⭐⭐⭐⭐ | 开源LLM基础 |
| 7 | InstructGPT: Training language models to follow instructions | 2022 | ⭐⭐⭐⭐⭐ | RLHF原理 |
| 8 | LoRA: Low-Rank Adaptation of Large Language Models | 2021 | ⭐⭐⭐⭐⭐ | 高效微调 |
| 9 | FlashAttention: Fast and Memory-Efficient Exact Attention | 2022 | ⭐⭐⭐⭐ | 注意力优化 |
| 10 | GPT-4 Technical Report | 2023 | ⭐⭐⭐⭐ | 多模态/Agent |

---

### 第三阶段：进阶技术

| 序号 | 论文 | 年份 | 重要性 | 读后能回答 |
|------|------|------|--------|-----------|
| 11 | DPO: Direct Preference Optimization | 2023 | ⭐⭐⭐⭐ | DPO vs RLHF |
| 12 | QLoRA: Efficient Finetuning | 2023 | ⭐⭐⭐⭐ | 量化微调 |
| 13 | RAG: Retrieval-Augmented Generation | 2020 | ⭐⭐⭐⭐ | RAG原理 |
| 14 | LangChain | 2023 | ⭐⭐⭐ | LLM应用框架 |
| 15 | ReAct: Synergizing Reasoning and Acting | 2023 | ⭐⭐⭐⭐ | Agent原理 |

---

## 论文精读指南

### Attention Is All You Need

```
摘要重点：
- Transformer完全基于Attention，不使用RNN/CNN
- 可并行化，训练速度大幅提升

核心贡献：
1. Self-Attention
2. Multi-Head Attention
3. Positional Encoding
4. Encoder-Decoder架构

必须理解：
- Q, K, V的作用
- 为什么除以√d_k
- Multi-Head的好处
- 位置编码的选择
```

### LLaMA

```
摘要重点：
- 开源高效的基础模型
- 训练数据：CommonCrawl, C4, Wikipedia, etc.
- 参数量：7B/13B/33B/65B

核心贡献：
1. 开源模型权重
2. 高效训练
3. 性能媲美GPT-3

必须理解：
- 与GPT的区别（开源vs闭源）
- Scaling Law的应用
```

### InstructGPT / RLHF

```
摘要重点：
- 让语言模型与人类意图对齐
- 使用人类反馈进行微调

核心步骤：
1. SFT（有监督微调）
2. RM（奖励模型）
3. PPO（强化学习）

必须理解：
- 为什么需要RLHF
- PPO算法流程
- 与SFT/DPO的区别
```

### LoRA

```
摘要重点：
- 通过低秩分解减少参数量
- 冻结原模型，只训练A和B矩阵

核心公式：
ΔW = BA
其中 A ∈ R^{r×k}, B ∈ R^{d×r}, r << min(d,k)

必须理解：
- 为什么低秩有效
- 与全量微调的效果对比
- 秩r的选择
```

---

## 论文速读模板

```markdown
## 论文速读：[论文名称]

### 一句话总结
[用一句话描述论文的核心贡献]

### 背景/动机
[为什么要做这个？之前的方法有什么问题？]

### 核心方法
[用图或公式描述核心算法]

### 关键创新点
1. ...
2. ...
3. ...

### 实验结果
[关键数据]

### 局限性
[论文自己提到的limitations]

### 应用场景
[这篇论文的方法可以用在哪里？]
```

---

## 面试常问的论文问题

```
1. Attention Is All You Need
   - 画出Transformer架构图
   - 解释QKV的作用
   - 解释位置编码

2. BERT vs GPT
   - 两者有什么区别
   - 各自的优缺点

3. LoRA
   - 解释低秩适应的原理
   - 为什么有效

4. RLHF
   - 三个步骤是什么
   - PPO的优势

5. Flash Attention
   - 解决了什么问题
   - 核心思想
```

---

## 如何高效读论文

```
1. 先看代码实现，再看论文
   → 很多论文已经有开源代码

2. 使用papers-with-code
   → 论文+代码+基准测试

3. 精读vs泛读
   → 核心论文精读，其他泛读

4. 做笔记
   → 用Notion/Obsidian整理
```
