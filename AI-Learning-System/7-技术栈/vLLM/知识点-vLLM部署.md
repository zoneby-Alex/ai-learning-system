# 知识点卡片：vLLM推理部署

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | vLLM推理框架部署与调优 |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P0 |
| 预估时间 | 6小时 |

---

## 安装与快速开始

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="float16",
    tensor_parallel_size=1,  # GPU数量
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=256,
)

# 批量推理
prompts = ["Hello, how are you?", "What is the capital of France?"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

---

## API服务部署

```bash
# 启动OpenAI兼容API
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

```python
# 客户端调用
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
    temperature=0.7
)
```

---

## 性能调优参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| gpu_memory_utilization | GPU显存利用率 | 0.85-0.95 |
| max_model_len | 最大序列长度 | 根据模型定 |
| max_num_seqs | 最大并发序列数 | 256 |
| max_num_batched_tokens | 最大批处理token数 | 8192 |
| tensor_parallel_size | 张量并行GPU数 | GPU数量 |
| dtype | 推理精度 | "auto"/float16/bfloat16 |

---

## 性能对比

```python
# HuggingFace: ~40 tokens/s (batch=1)
# vLLM: ~150 tokens/s (batch=1), ~800 tokens/s (batch=8)
# 加速比：3-5x (单batch), 10-20x (高并发)
```

---

## 相关知识点

- → [推理优化技术](../../4-LLM与生成式AI/推理优化/知识点-推理优化技术.md)
- → [推理优化项目](../../6-项目实战/高级项目/项目-推理优化实践.md)
