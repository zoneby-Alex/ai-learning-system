# HuggingFace模块

本模块介绍HuggingFace Transformers库。

## 核心功能

```
HuggingFace = 模型 + 数据集 + 预训练 + 微调

主要库：
├── transformers: 模型和预训练
├── datasets: 数据集加载
├── peft: 高效微调
├── accelerate: 分布式训练
└── tokenizers: 分词器
```

---

## 模型加载

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 加载模型
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 配置
config = AutoConfig.from_pretrained("bert-base-uncased")
config.hidden_size = 256  # 修改配置
```

---

## 文本处理

```python
# 编码
inputs = tokenizer("Hello world!", return_tensors="pt")

# 解码
outputs = model.generate(**inputs)
text = tokenizer.decode(outputs[0])

# 批量处理
texts = ["Hello", "World"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

---

## 模型训练

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

## PEFT微调

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

---

## 相关知识点

- → [PyTorch](../PyTorch/README.md)
- → [LangChain](../LangChain/README.md)
