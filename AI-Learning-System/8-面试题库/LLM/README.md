# LLM面试题库

本模块收集LLM方向的高频面试题。

---

## 1. 基础概念

### Q1: GPT和BERT的区别

| 方面 | GPT | BERT |
|------|-----|------|
| 架构 | Decoder-only | Encoder-only |
| 目标 | Next token | Masked LM |
| 双向性 | 单向 | 双向 |
| 适用 | 生成 | 理解 |

---

## 2. 训练技术

### Q2: RLHF的三个步骤

**答**：
1. **SFT**：有监督微调，用人工标注的问答对微调模型
2. **RM**：训练奖励模型，用人类偏好数据
3. **PPO**：强化学习优化，使回答符合人类偏好

### Q3: DPO和RLHF的区别

**答**：
- RLHF需要三步（RM+PPO）
- DPO直接用偏好数据优化，两步
- DPO更简单稳定

---

## 3. 高效微调

### Q4: LoRA原理

**答**：
- 冻结预训练权重W₀
- 添加低秩分解ΔW = BA
- 前向：h = W₀x + BAx
- 参数量从d×k减少到(d+r)×r

---

## 4. RAG

### Q5: RAG流程

**答**：
```
文档 → 加载 → 切分 → 向量化 → 存储
                              ↓
用户查询 → 向量化 → 检索 → 重排 → Prompt → LLM → 回答
```

---

## 5. Agent

### Q6: ReAct和CoT的区别

| 方面 | CoT | ReAct |
|------|-----|-------|
| 推理 | 只思考 | 思考+行动+观察 |
| 工具 | 无 | 有 |
| 适用 | 简单推理 | 需交互任务 |

---

## 更多题目

- Flash Attention原理
- PagedAttention原理
- 量化原理（INT8/FP16/GPTQ）
- 位置编码对比（RoPE/ALiBi）
- 长上下文处理方法

---

## 相关知识点

- → [RAG](../4-LLM与生成式AI/RAG系统/README.md)
- → [Agent](../4-LLM与生成式AI/Agent/README.md)
- → [模型训练](../4-LLM与生成式AI/模型训练/README.md)
