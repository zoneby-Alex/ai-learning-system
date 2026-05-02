# 模型训练模块

本模块包含LLM训练的核心技术：预训练、SFT、RLHF、LoRA等。

## 训练流程

```
预训练(Pretrain) → 有监督微调(SFT) → 对齐微调(RLHF/DPO) → 部署
     ↓                    ↓                    ↓
  大量数据            少量高质量数据        人类偏好
```

---

## 预训练 (Pretrain)

### Scaling Law

```python
"""
Scaling Law: 模型性能随计算量、数据量、参数量幂律增长

公式：
L(N) ≈ (N_c / N)^α_N
L(D) ≈ (D_c / D)^α_D
L(C) ≈ (C_c / C)^α_C

其中：
- N: 参数量
- D: 数据量(tokens)
- C: 计算量(FLOPs)
- α ≈ 0.076 (N), 0.095 (D), 0.050 (C)
"""
```

### 分布式训练

```python
# 数据并行
# 每个GPU有完整模型，处理不同数据

# 张量并行
# 模型按维度切分到多个GPU

# 流水线并行
# 不同GPU处理不同层

# DeepSpeed ZeRO
# ZeRO-1: 分 optimizer states
# ZeRO-2: 分 gradients + optimizer states
# ZeRO-3: 分 parameters + gradients + optimizer states
```

---

## SFT (有监督微调)

```python
"""
使用标注数据进行微调

数据格式：
{
    "instruction": "将以下中文翻译成英文",
    "input": "你好",
    "output": "Hello"
}

训练：
- 使用因果语言建模损失
- 简短训练（1-3 epochs）
- 学习率比预训练小
"""
```

---

## RLHF (人类反馈强化学习)

```python
"""
三步训练流程：

1. SFT：有监督微调
2. RM（奖励模型）：
   - 训练奖励模型预测人类偏好
   - 损失：pairwise ranking loss
3. PPO强化学习：
   - 使用奖励模型优化策略
   - KL散度约束防止偏离

代码示例（简化）：
reward = reward_model(state, action)
loss = -reward  # 最大化奖励
"""
```

---

## DPO (Direct Preference Optimization)

```python
"""
简化RLHF，直接优化偏好

损失函数：
L = -log σ( r(x,y_w) - r(x,y_l) - β * log(π(y_w|x)/π_ref(y_w|x)) + β * log(π(y_l|x)/π_ref(y_l|x)) )

其中：
- y_w: chosen response (人类偏好)
- y_l: rejected response (人类不偏好)
- r: 奖励模型
- π: 策略模型
- π_ref: 参考模型

优势：
- 不需要单独的奖励模型
- 不需要PPO训练
- 更稳定，效果相当
"""
```

---

## LoRA / QLoRA

### LoRA原理

```python
"""
低秩适应：在预训练权重旁边添加低秩矩阵

核心：
- 冻结预训练权重W_0
- 添加可训练的低秩分解 ΔW = BA
- 其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)

前向：h = W_0 x + ΔW x = W_0 x + BAx

优势：
- 参数量大幅减少（r=8时，约0.1%参数）
- 可组合多个LoRA adapter
- 训练快，显存省
"""
```

### QLoRA

```python
"""
量化LoRA：4-bit NormalFloat量化 + LoRA

步骤：
1. 4-bit量化预训练权重（NF4）
2. LoRA adapter在16-bit训练
3. 合并权重用于推理

效果：
- 65B模型可以在48GB显存微调
- 训练速度：约25 tokens/s/A100
"""
```

### PEFT使用

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=8,                           # 秩
    lora_alpha=16,                 # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用到的模块
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.0623
```

---

## 面试高频问题

### Q1: RLHF和DPO的区别？

**答**：
- RLHF需要训练奖励模型 + PPO优化，三阶段
- DPO直接用偏好数据优化，两阶段
- DPO更简单稳定，效果相当

---

### Q2: LoRA为什么有效？

**答**：
- 预训练模型权重已接近最优，低秩扰动足够
- 任务相关参数在低秩子空间
- 减少过拟合风险

---

## 相关知识点

- → [RAG系统](../RAG系统/README.md) - 应用
- → [推理优化](../推理优化/README.md) - 部署
