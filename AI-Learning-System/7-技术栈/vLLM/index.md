# vLLM模块

本模块介绍vLLM高效推理框架。

## 核心特性

```
vLLM = PagedAttention + Continuous Batching + 张量并行

优势：
├── 显存利用率提升2-4倍
├── 吞吐量提升数倍
├── 支持连续批处理
└── 分布式推理
```

---

## 快速上手

```python
from vllm import LLM, SamplingParam

# 创建LLM实例
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,  # 张量并行
    gpu_memory_utilization=0.9,
)

# 采样参数
sampling_params = SamplingParam(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

# 生成
outputs = llm.generate(["Hello world!"], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## PagedAttention原理

```python
"""
传统KV Cache：
- 连续显存分配
- 碎片化严重
- 利用率低

PagedAttention：
- 类似操作系统的分页管理
- 按需分配显存
- 共享前缀cache
"""
```

---

## API服务

```python
from vllm import LLM
from fastapi import FastAPI

app = FastAPI()
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

@app.post("/generate")
async def generate(request: Request):
    return llm.generate(request.prompts)
```

---

## 相关知识点

- → [推理优化](../4-LLM与生成式AI/推理优化/README.md)
- → [TensorRT](../4-LLM与生成式AI/推理优化/README.md)
