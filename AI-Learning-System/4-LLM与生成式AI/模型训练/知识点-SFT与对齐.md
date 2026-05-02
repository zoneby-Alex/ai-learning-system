# 知识点卡片：SFT与对齐技术

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | SFT / RLHF / DPO / 对齐技术 |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 8小时 |
| 面试频率 | ★★★★★ |

---

## 从Pretrain到Chat模型

```
Pretrain（基座模型）
  │ 学习：预测下一个token
  │ 能力：续写文本，但不一定遵循指令
  ↓
SFT（有监督微调）
  │ 学习：遵循指令格式
  │ 能力：理解"请帮我..."并给出有用回答
  ↓
RLHF / DPO（对齐）
  │ 学习：符合人类偏好
  │ 能力：生成有帮助、诚实、无害的回复
```

---

## SFT (Supervised Fine-Tuning)

```python
# SFT数据格式
sft_data = [
    {
        "instruction": "解释什么是机器学习",
        "input": "",
        "output": "机器学习是人工智能的一个分支..."
    },
    {
        "instruction": "将以下句子翻译成英文",
        "input": "今天天气真好",
        "output": "The weather is really nice today."
    }
]

# 训练时使用标准的自回归损失
# 但只在"output"部分计算loss（mask掉instruction部分）

# 关键：
# 1. 数据质量 > 数据数量
# 2. 多样化的任务类型（翻译/总结/编码/推理等）
# 3. 格式一致性很重要（chat template）
```

### Chat Template

```python
# Llama 3 的 chat template
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_response}<|eot_id|>"""
```

---

## 对齐方法对比

### RLHF (Reinforcement Learning from Human Feedback)

```
步骤：
1. SFT模型作为初始策略
2. 训练Reward Model（从人类偏好排序数据）
3. PPO优化：max E[r(x,y)] - β KL(π||π_ref)

优点：理论上最优，灵活
缺点：需要训练RM，PPO不稳定，工程复杂
```

### DPO (Direct Preference Optimization)

```
直接优化偏好数据，跳过显式RM：
L = -log σ(β(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))

优点：简单稳定，不需要RM和RL
缺点：需要成对偏好数据
```

### KTO (Kahneman-Tversky Optimization)

```
只需单个回答的好坏标签（不需要成对数据）：
- 基于前景理论（人类对损失的敏感度高于收益）
- 更简单的数据收集（只需评分是好/坏，不需要排序对比）
```

---

## 对齐的HHH原则

```
Helpful（有帮助的）：
- 准确回答问题
- 提供有用的信息
- 承认不知道

Honest（诚实的）：
- 不捏造信息
- 标明不确定性
- 区分事实和观点

Harmless（无害的）：
- 拒绝有害请求
- 不生成歧视/暴力内容
- 保护隐私和安全
```

---

## 面试高频问题

### Q1: 为什么需要SFT？Pretrain模型不够吗？

**答**：
Pretrain模型只会"续写文本"，不知道什么是"回答"。如果你问"1+1=?"，它可能继续随机文本而非回答"2"。SFT教会模型：
- 指令格式：理解"问题→答案"的结构
- 有帮助的输出风格
- 拒绝不当请求

### Q2: DPO为什么能替代RLHF中的PPO？

**答**：
DPO数学上推导了偏好概率与策略概率比的关系：
```
P(y_w > y_l) = σ(β log π(y_w)/π_ref(y_w) - β log π(y_l)/π_ref(y_l))
```
这意味着可以直接从偏好数据优化策略，而不需要中间的奖励模型和RL过程。效果接近RLHF但更简单。

### Q3: DPO和RLHF各适合什么场景？

| 场景 | 推荐方法 |
|------|---------|
| 有大量成对排序数据 | DPO |
| 有大量单点评分数据 | KTO |
| 需要在线反馈循环 | RLHF+PPO |
| 追求最优效果 | RLHF（理论最优） |
| 资源有限/快速迭代 | DPO |

---

## 相关知识点

- → [RLHF与DPO详解](../../3-深度学习核心/训练技术/知识点-RLHF与DPO.md)
- → [Pretrain训练](./知识点-Pretrain训练.md) - 预训练基础
- → [RLHF论文](../LLM必读论文清单.md)
