# 知识点卡片：HuggingFace Transformers

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | HuggingFace Transformers库 |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 8小时 |

---

## 核心API

### AutoModel / AutoTokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Tokenization
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
# {'input_ids': tensor([...]), 'attention_mask': tensor([...])}

# 批量处理（自动padding）
tokenizer.pad_token = tokenizer.eos_token
texts = ["Hello", "Longer text here"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

### Pipeline

```python
from transformers import pipeline

# 文本生成
generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")
result = generator("Once upon a time", max_new_tokens=50)

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# 命名实体识别
ner = pipeline("ner", grouped_entities=True)
entities = ner("My name is John and I live in Paris")

# 翻译
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
result = translator("你好世界")
```

### 生成策略

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7,        # 温度（越高越随机）
    top_p=0.9,             # nucleus sampling
    top_k=50,              # top-k sampling
    do_sample=True,        # 采样（False则greedy）
    num_beams=4,           # beam search
    repetition_penalty=1.1,# 惩罚重复
    early_stopping=True,
    pad_token_id=tokenizer.eos_token_id
)
```

### Dataset

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# 自定义处理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

---

## 模型加载方式

```python
# FP16加载
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)

# 8-bit量化
model = AutoModelForCausalLM.from_pretrained(name, load_in_8bit=True)

# 4-bit量化
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config)

# 分片加载（超大规模模型）
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", max_memory={0: "20GB", 1: "20GB"})
```

---

## 相关知识点

- → [PEFT/LoRA](../../3-深度学习核心/LoRA/知识点-LoRA微调.md)
- → [Llama模型](../../4-LLM与生成式AI/Llama系列/知识点-Llama演化.md)
