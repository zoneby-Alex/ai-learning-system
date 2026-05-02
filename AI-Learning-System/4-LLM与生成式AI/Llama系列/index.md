# Llama系列模块

本模块包含Llama系列开源模型的发展和技术特点。

## 演化路线

```
Llama1(2023.02) → Llama2(2023.07) → Llama3(2024.04) → Llama4(2025)
```

## 模型对比

| 模型 | 参数量 | 上下文 | 特点 |
|------|--------|--------|------|
| Llama 1 | 7B/13B/33B/65B | 2048 | 首个开源LLaMA |
| Llama 2 | 7B/13B/70B | 4096 | 开源可商用 |
| Llama 3 | 8B/70B | 8192 | 指令微调/SFT |
| Llama 3.1 | 8B/70B/405B | 128K | 开源最强 |
| Llama 4 | - | 200K | 多模态/MoE |

---

## 核心技术

### Llama 1: 开创开源LLaMA

```python
"""
核心贡献：开源高效的基础语言模型

训练数据：
- CommonCrawl (67%)
- C4 (15%)
- Wikipedia (4.5%)
- Books (4.5%)
- arXiv (2.5%)
- GitHub (0.5%)

优化：
- RMSNorm（更稳定）
- SwiGLU激活函数
- RoPE位置编码
- 注意力优化（xformers）
"""
```

### Llama 2: 可商用开源

```python
"""
核心贡献：开源可商用，发布Llama 2-chat

安全性提升：
- 红队测试
- RLHF（SFT + PPO）
- Ghost Attention（多轮对话）

预训练改进：
- 40% more training tokens
- 2M context length
- Grouped-Query Attention (GQA)
"""
```

### Llama 3: 指令微调开源

```python
"""
核心贡献：开源指令微调模型，效果接近GPT-4

训练数据：
- 15T tokens（大幅增加）
- 高质量英语数据
- 5%非英语数据

架构变化：
- 8B: 8K context，FFN→SwiGLU
- 70B: GQA，8K context
- 训练稳定性提升
"""
```

---

## 本地部署

```python
# Ollama部署
# 1. 安装Ollama
# 2. 拉取模型
ollama pull llama3.2
ollama pull llama3.2:70b

# 3. 使用
ollama run llama3.2 "Hello, how are you?"

# Python API
from llama_index import LlamaIndex
# 或使用ollama Python SDK
import ollama
response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Hello!'}
])
```

---

## 面试问题

### Q1: Llama和其他开源模型的区别？

**答**：
- **Meta开源**：模型权重开放，可商用（Llama 2/3）
- **高效训练**：使用RMSNorm、SwiGLU、RoPE等优化
- **社区生态**：最多社区支持，生态最完善

---

## 相关知识点

- → [Qwen](../国内模型/README.md) - 国内开源对标
- → [MoE](../模型训练/README.md) - Llama 4可能使用MoE
