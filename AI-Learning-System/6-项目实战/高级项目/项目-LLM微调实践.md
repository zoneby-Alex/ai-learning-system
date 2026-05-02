# 项目：LLM微调实践

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐⭐⭐☆ |
| 预估时间 | 3-4周 |
| 目标就业 | LLM算法/微调工程师 |
| 技术栈 | PyTorch + PEFT + DeepSpeed |

---

## 项目目标

1. 理解LLM微调全流程
2. 实现LoRA/QLoRA微调
3. 评估微调效果

---

## 技术栈

```python
# 核心依赖
torch
transformers
peft
deepseed
accelerate
bitsandbytes  # 用于QLoRA
```

---

## 代码实现

### 1. 数据准备

```python
from datasets import load_dataset

# 加载数据集（alpaca格式）
dataset = load_dataset("yahma/alpaca-cleaned")

# 格式化
def format_instruction(example):
    return {
        "text": f"""<|user|>
{example['instruction']}

{example['input']}

<|assistant|>
{example['output']}"""
    }

dataset = dataset.map(format_instruction)
```

### 2. LoRA配置

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                          # 秩
    lora_alpha=32,                # 缩放因子
    target_modules=[               # 应用到的模块
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### 3. QLoRA微调

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载模型（QLoRA）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备训练
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()
# trainable params: 41,602,048 || all params: 6,738,415,616 || trainable%: 0.617%
```

### 4. 训练配置

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.03,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
```

---

## 项目结构

```
llm_finetune/
├── data/
│   └── prepare_data.py
├── scripts/
│   └── train.sh
├── src/
│   ├── __init__.py
│   ├── load_model.py
│   ├── lora_utils.py
│   └── train.py
├── output/
├── requirements.txt
└── README.md
```

---

## 面试口述要点

```
这个项目实现了LLM微调的完整流程：

1. 数据处理：
   - 使用alpaca格式数据
   - 实现数据清洗和格式化
   - 构建prompt模板

2. 量化技术：
   - 使用QLoRA进行4-bit量化
   - NF4数据类型减少显存
   - Double Quant进一步压缩

3. LoRA配置：
   - 秩r=16的选择（权衡效果和效率）
   - 目标模块的选择（qkv + FFN）
   - 缩放因子alpha=2r

4. 训练技巧：
   - Paged AdamW优化器
   - 梯度累积解决小batch
   - Warmup提高稳定性
```

---

## 验收标准

- [ ] 能加载和量化模型
- [ ] 能配置和训练LoRA
- [ ] 能合并权重进行推理
- [ ] 能评估微调效果
