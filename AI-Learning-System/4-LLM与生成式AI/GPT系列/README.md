# GPT系列模块

本模块包含GPT系列模型的发展历程和技术特点。

## GPT演化路线

```
GPT-1(2018) → GPT-2(2019) → GPT-3(2020) → InstructGPT(2022) → GPT-4(2023) → GPT-4o(2024) → o1/o3(2024)
```

## 模型对比

| 模型 | 年份 | 参数量 | 上下文 | 特点 |
|------|------|--------|--------|------|
| GPT-1 | 2018 | 117M | 512 | 开创性 |
| GPT-2 | 2019 | 1.5B | 1024 | 开源/多任务 |
| GPT-3 | 2020 | 175B | 2048 | In-context learning |
| InstructGPT | 2022 | 175B | 4096 | RLHF对齐 |
| GPT-4 | 2023 | ? | 128K | 多模态/长上下文 |
| GPT-4o | 2024 | ? | 128K | 实时语音/原生多模态 |
| o1 | 2024 | ? | 200K | 推理模型/CoT |

---

## 核心技术

### GPT-1: 开创性工作

```python
"""
核心贡献：证明了语言模型可以通过预训练+微调范式泛化到下游任务

架构：Transformer Decoder
训练：BooksCorpus (5GB)
微调：针对每个下游任务微调
"""
```

### GPT-2: 多任务学习

```python
"""
核心贡献：语言模型是无监督的多任务学习器

关键观点：
- 不需要监督信号来学习任务
- 联合表示比条件表示更好
- 扩大模型规模和数据集

架构变化：
- 更大更深（48层，1542M参数）
- 注意力机制优化
- 更大的上下文窗口
"""
```

### GPT-3: In-Context Learning

```python
"""
核心贡献：证明了大规模语言模型可以通过few-shot完成新任务

三种推理范式：
1. Few-shot：在prompt中提供几个示例
2. One-shot：提供一个示例
3. Zero-shot：不提供示例，只给指令

示例：
Prompt: "Translate to French: hello world"
Output: "bonjour monde"
"""
```

### InstructGPT: RLHF对齐

```python
"""
核心贡献：让语言模型与人类意图对齐

三步训练：
1. SFT：有监督微调
2. RM：训练奖励模型
3. PPO：强化学习优化

损失函数：
L = E[r(x,y)] - β * log π(y|x) / π_ref(y|x)

其中r是奖励模型，π是策略，π_ref是参考模型
"""
```

---

## 面试高频问题

### Q1: GPT和BERT的区别？

| 方面 | GPT | BERT |
|------|-----|------|
| 架构 | Decoder-only | Encoder-only |
| 训练目标 | Next token prediction | Masked LM |
| 双向性 | 单向 | 双向 |
| 适用任务 | 生成 | 理解 |
| 微调方式 | 轻量微调 | 全量微调 |

---

### Q2: In-Context Learning是什么？

**答**：
语言模型通过在prompt中提供几个示例来学习新任务，不需要梯度更新：
```
示例：
Input: "great movie ! [positive]"
Input: "boring film [negative]"
Input: "I loved it !"
Output: "[positive]"
```

---

## 相关知识点

- → [Transformer](../Transformer/README.md) - 基础架构
- → [RLHF](../模型训练/README.md) - 对齐技术
