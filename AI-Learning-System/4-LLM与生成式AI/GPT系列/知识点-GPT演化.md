# 知识点卡片：GPT系列演化

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | GPT系列演化史 (GPT-1 → GPT-4o → o1) |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 5小时 |
| 面试频率 | ★★★★★ |

---

## 演化路线

```
GPT-1(2018) → GPT-2(2019) → GPT-3(2020) → InstructGPT(2022)
→ ChatGPT(2022) → GPT-4(2023) → GPT-4o(2024) → o1/o3(2024)
```

---

## 各代核心创新

### GPT-1 (2018) - 证明预训练有效
- 117M参数，12层Decoder
- 核心：Generative Pre-Training + Task-specific Fine-tuning
- 证明在大规模无标注文本上预训练 + 下游微调的有效性

### GPT-2 (2019) - Zero-shot能力
- 1.5B参数，48层
- 核心：更大的模型 + 更多数据 → Zero-shot能力
- 争议：因"风险太大"一度不开源

### GPT-3 (2020) - In-context Learning
- 175B参数
- 核心：Few-shot/Zero-shot In-context Learning
- 不需要微调，通过prompt中示例就能完成新任务
- 涌现能力的里程碑

### InstructGPT / ChatGPT (2022)
- 核心：RLHF对齐技术
- 三步：SFT → 奖励模型 → PPO
- 从"预测下一个token"到"遵循人类指令"

### GPT-4 (2023) - 多模态+更强推理
- ~1.8T参数（MoE架构推测）
- 支持图像输入（多模态）
- 在各基准测试上大幅超越GPT-3.5

### GPT-4o (2024) - 全模态
- 原生多模态：文本+图像+语音
- 更低延迟、更低成本
- 端到端训练（非pipeline拼接）

### o1 / o3 (2024) - 推理增强
- Chain-of-Thought推理链
- 用更多推理时间换取更好结果
- 在数学、编程等需多步推理的任务上突破

---

## 面试高频问题

### Q1: GPT系列的核心架构是什么？

**答**：
GPT使用Transformer Decoder-Only架构：
- 堆叠多层Transformer Decoder Block
- 每层包含：Masked Self-Attention + FFN
- 使用因果mask（只看前面的token）
- 自回归生成：逐个token预测下一个

### Q2: GPT-1到GPT-3的核心变化是什么？

| 方面 | GPT-1 | GPT-2 | GPT-3 |
|------|-------|-------|-------|
| 参数 | 117M | 1.5B | 175B |
| 层数 | 12 | 48 | 96 |
| 创新 | 预训练范式 | Zero-shot | In-context Learning |
| 微调 | 需要 | 需要 | 可选 |
| 关键发现 | - | 规模带来zero-shot | 涌现能力 |

### Q3: InstructGPT相比GPT-3做了什么？

**答**：
GPT-3虽然是优秀的基础模型，但不懂"遵循指令"。InstructGPT通过RLHF：
1. 收集人类编写的(prompt, answer)对做SFT
2. 收集人类排序偏好训练奖励模型
3. 用PPO优化模型使其生成的内容让奖励模型给高分
这使得模型真正变得"有帮助且无害"。

---

## 相关知识点

- → [Llama演化](../Llama系列/知识点-Llama演化.md) - 开源路线
- → [RLHF与DPO](../../3-深度学习核心/训练技术/知识点-RLHF与DPO.md) - 对齐技术
- → [Transformer](../../3-深度学习核心/Transformer/知识点-Self-Attention.md) - 基础架构
