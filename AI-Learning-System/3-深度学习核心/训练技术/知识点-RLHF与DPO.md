# 知识点卡片：RLHF与DPO

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | RLHF / DPO / 对齐技术 |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P0 |
| 预估时间 | 8小时 |
| 面试频率 | ★★★★★ |

---

## RLHF三步流程

```
Step 1: SFT (Supervised Fine-Tuning)
  收集高质量的指令-回答对
  在基座模型上微调
  → 让模型学会"跟随指令"

Step 2: RM (Reward Model) 训练
  收集人类偏好数据（对同一prompt的多个回答排序）
  训练奖励模型预测人类偏好
  → 让模型学会"什么是好的回答"

Step 3: PPO (Proximal Policy Optimization)
  用RM的奖励信号，通过强化学习优化策略
  同时加KL惩罚防止偏离SFT太远
  → 让模型生成更符合人类偏好的回答
```

### RLHF损失函数

```
PPO目标：
max E[ r(x,y) - β KL(π_θ(y|x) || π_ref(y|x)) ]

其中：
- r(x,y)：奖励模型的打分
- KL项：防止新策略偏离参考策略太远
- β：KL惩罚系数
```

---

## DPO (Direct Preference Optimization)

DPO直接使用偏好数据优化模型，跳过显式的RM和RL：

```
DPO损失：
L_DPO = -E [ log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))) ]

其中：
- y_w：被人类偏好的回答（winning）
- y_l：不被偏好的回答（losing）
- σ：sigmoid函数
- β：控制偏离ref的程度

核心思想：
如果模型认为好的回答(y_w)比差的回答(y_l)更可能，
那么sigmoid内的差值为正，loss降低。
```

---

## RLHF vs DPO

| 方面 | RLHF | DPO |
|------|------|-----|
| 步骤 | 3步(SFT+RM+PPO) | 2步(SFT+DPO) |
| 奖励模型 | 需要单独训练RM | 隐式表示 |
| 训练复杂度 | 高(4个模型同时) | 低(2个模型) |
| 稳定性 | PPO训练不稳定 | 稳定(直接优化) |
| 效果 | 理论最优 | 接近RLHF |
| 工程难度 | 极高 | 中等 |

---

## 代码示例

```python
# DPO损失实现
import torch
import torch.nn.functional as F

def dpo_loss(model, ref_model, batch, beta=0.1):
    """
    batch包含: prompt, chosen(偏好), rejected(不偏好)
    """
    # 获取chosen和rejected的对数概率
    logp_chosen_model = model(batch['prompt'], batch['chosen'])
    logp_rejected_model = model(batch['prompt'], batch['rejected'])
    logp_chosen_ref = ref_model(batch['prompt'], batch['chosen'])
    logp_rejected_ref = ref_model(batch['prompt'], batch['rejected'])

    # 相对于ref的log概率比
    chosen_ratio = logp_chosen_model - logp_chosen_ref
    rejected_ratio = logp_rejected_model - logp_rejected_ref

    # DPO loss
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()

    # 准确率（chosen比rejected更被偏好的比例）
    accuracy = (logits > 0).float().mean()

    return loss, accuracy
```

---

## 面试高频问题

### Q1: 为什么需要RLHF？

**答**：
基座LLM只学习预测下一个token，不知道"什么是好的回答"。RLHF通过人类反馈信号教会模型什么是：
- 有帮助的（helpful）
- 诚实的（honest）
- 无害的（harmless）

这是从"语言建模"到"对齐人类意图"的关键步骤。

### Q2: RLHF中的KL惩罚为什么重要？

**答**：
如果只优化奖励信号，模型可能学会"欺骗"奖励模型（reward hacking）：
- 生成能得高分但无意义的文本
- 偏离自然语言分布

KL惩罚确保模型既获得高奖励又保持语言能力，防止过拟合到奖励模型。

### Q3: DPO相比RLHF的优势？

**答**：
1. 不需要单独训练奖励模型
2. 不需要强化学习（PPO训练不稳定）
3. 直接优化偏好，更简单
4. 理论和实践中接近RLHF效果
DD主要局限：需要成对的偏好数据（chosen vs rejected）

---

## 相关知识点

- → [RLHF论文](../../4-LLM与生成式AI/LLM必读论文清单.md)
- → [Pretrain训练](../../4-LLM与生成式AI/模型训练/知识点-Pretrain训练.md)
