# 知识点卡片：DeepSpeed分布式训练

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | DeepSpeed / FSDP 分布式训练 |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P1 |
| 预估时间 | 6小时 |

---

## ZeRO优化级别

```
ZeRO-1: 优化器状态分片
  - 每个GPU存完整参数+梯度，只分片优化器状态
  - 显存节省：4x (4 GPU)

ZeRO-2: 优化器状态+梯度分片
  - 每个GPU存完整参数，分片梯度和优化器状态
  - 显存节省：8x (8 GPU)

ZeRO-3: 参数+优化器+梯度全分片
  - 所有组件分片到所有GPU
  - 需要时通过allgather收集参数
  - 显存节省：线性扩展（N GPU → 可训练N倍大模型）
```

---

## 配置与使用

```python
# deepspeed_config.json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 2e-5, "betas": [0.9, 0.999], "weight_decay": 0.01}
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {"warmup_min_lr": 0, "warmup_max_lr": 2e-5, "warmup_num_steps": 1000}
    }
}
```

```python
# 训练脚本
import deepspeed
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        deepspeed="deepspeed_config.json",  # 只需指定配置文件！
        fp16=True,
    ),
    train_dataset=dataset,
)
trainer.train()
```

---

## DeepSpeed vs FSDP vs DDP

| 方面 | DDP | FSDP | DeepSpeed ZeRO-3 |
|------|-----|------|-----------------|
| 显存效率 | 低（每卡完整模型） | 高 | 最高 |
| 通信开销 | 低 | 中 | 较高 |
| 易用性 | 最简单 | 中等 | 中等 |
| Offload | 不支持 | 支持CPU offload | 支持CPU+NVMe offload |
| 适用模型 | <1B | 1-10B | 1-100B+ |

---

## 关键技巧

```python
# 1. 梯度累积（用小batch模拟大batch）
gradient_accumulation_steps = 8
effective_batch_size = per_device_batch_size * gpus * gradient_accumulation_steps

# 2. CPU Offload（把优化器状态放到CPU）
# "offload_optimizer": {"device": "cpu"}

# 3. 混合精度
# fp16 或 bf16（A100/H100用bf16更稳定）
```

---

## 相关知识点

- → [Pretrain训练](../../4-LLM与生成式AI/模型训练/知识点-Pretrain训练.md)
- → [PyTorch核心](../PyTorch/知识点-PyTorch核心.md)
