# 推理优化模块

本模块包含LLM推理优化的核心技术。

## 优化技术全景

```
推理优化
├── 模型压缩
│   ├── 量化 (INT8/FP16/GPTQ/AWQ)
│   ├── 剪枝
│   └── 知识蒸馏
│
├── 注意力优化
│   ├── Flash Attention
│   ├── PagedAttention
│   └── KV Cache
│
└── 推理框架
    ├── TensorRT-LLM
    ├── vLLM
    ├── SGLang
    └── lmdeploy
```

---

## 量化

### INT8量化

```python
# 动态量化
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 静态量化
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
```

### GPTQ量化

```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    "model_name",
    quantization_config={"bits": 4, "desc_act": False}
)
```

---

## Flash Attention

```python
"""
核心：分块计算 + online softmax

复杂度：
- 标准attention: O(N²) 显存
- Flash attention: O(N) 显存

原理：
- 不materialize整个N×N矩阵
- 分块计算，利用数值稳定的online softmax
"""
```

---

## vLLM (PagedAttention)

```python
"""
核心：类操作系统分页管理KV Cache

优势：
- 显存利用率提升2-4倍
- 吞吐量提升数倍
- 支持continuous batching

使用：
from vllm import LLM
model = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = model.generate(["Hello world"])
"""
```

---

## TensorRT-LLM

```python
"""
NVIDIA推理优化库

优化：
- Kernel Fusion
- FP8量化
- In-flight批处理
- 张量并行

性能：
- 比HuggingFace快5-10倍
"""
```

---

## 相关知识点

- → [FlashAttention](../3-深度学习核心/Transformer/README.md)
- → [KV Cache](../3-深度学习核心/Transformer/README.md)
