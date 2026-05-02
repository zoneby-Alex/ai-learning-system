# 知识点卡片：LoRA微调

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | LoRA / QLoRA 高效微调 |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 6小时 |
| 面试频率 | ★★★★★ |

---

## 核心原理

LoRA (Low-Rank Adaptation) 使用低秩分解来高效微调大模型：

```
核心假设：模型微调时的权重更新ΔW是低秩的

ΔW = B A
其中 A ∈ R^{r×k}, B ∈ R^{d×r}, r << min(d,k)

前向传播：h = W₀x + ΔWx = W₀x + BAx

训练时：
- 冻结原模型权重 W₀
- 只训练低秩矩阵 A 和 B
- r 通常取 4-64，参数量远小于原模型
```

---

## 代码实现

### PEFT使用

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA配置
lora_config = LoraConfig(
    r=16,                          # 秩（关键参数）
    lora_alpha=32,                 # 缩放因子
    target_modules=[               # 应用LoRA的模块
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 6,738,415,616 || trainable%: 0.62%

# 训练
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# 保存和加载
model.save_pretrained("./my_lora")
model = PeftModel.from_pretrained(base_model, "./my_lora")

# 合并权重（推理时）
merged_model = model.merge_and_unload()
```

### QLoRA (量化LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 其余与LoRA相同
model = get_peft_model(model, lora_config)
```

---

## 关键参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| r | 低秩分解的秩 | 8-64（大=强但参数量大） |
| lora_alpha | 缩放因子 | 通常2r |
| target_modules | 应用的模块 | QKV+FFN（线性层） |
| lora_dropout | Dropout比例 | 0.05-0.1 |

**α/r = 缩放因子**：实际更新 = α/r * BAx。较大的α/r使LoRA更新更具影响力。

---

## 面试高频问题

### Q1: LoRA为什么有效？

**答**：
1. **低秩假设**：微调时的权重更新确实在低秩子空间中（经验验证）
2. **过参数化**：大模型有大量冗余参数，低秩矩阵足以捕获任务相关变化
3. **高效性**：参数量减少99%+，显存需求大幅降低
4. **可组合**：不同任务的LoRA权重可以独立保存和切换

### Q2: LoRA vs 全量微调？

| 方面 | LoRA | 全量微调 |
|------|------|---------|
| 可训练参数 | 0.1-1% | 100% |
| 显存占用 | ~24GB (7B) | ~60GB+ |
| 训练速度 | 快 | 慢 |
| 灾难性遗忘 | 少（冻结原参数） | 多 |
| 效果 | 略低于全量 | 最优 |
| 部署 | 一个模型+多个LoRA | 每个任务一个模型 |

### Q3: r的选择有何影响？

**答**：
- r=4：最激进压缩，效果可能不够
- r=8：通常的起点，大多数任务够用
- r=16：更好的效果，略多参数
- r=64：接近全量微调效果

实践中r=8或16已足够大多数任务。

---

## 相关知识点

- → [QLoRA](../../4-LLM与生成式AI/模型训练/知识点-SFT与对齐.md) - 量化微调
- → [PEFT](../../7-技术栈/HuggingFace/知识点-Transformers使用.md) - 微调工具
