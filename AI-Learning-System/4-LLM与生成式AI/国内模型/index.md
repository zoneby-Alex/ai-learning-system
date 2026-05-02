# 国内模型模块

本模块包含国内主流大语言模型的介绍。

## 模型对比

| 模型 | 开发公司 | 参数量 | 特点 |
|------|---------|--------|------|
| ChatGLM | 清华/智谱 | 6B/130B | 中文优化/清华系 |
| Qwen | 阿里 | 7B/72B/110B | 电商/阿里生态 |
| DeepSeek | 深度求索 | 7B/33B/236B | 高性价比/MoE |
| Baichuan | 百川 | 7B/13B | 对话优化 |
| Yi | 零一万物 | 6B/34B | 认知能力 |
| InternLM | 上海AI Lab | 7B/104B | 开源最强 |

---

## ChatGLM

```python
"""
清华/智谱AI开发
基于GLM架构

ChatGLM-6B:
- 62亿参数
- FP16约12GB显存
- INT4量化后6GB

ChatGLM3-6B:
- 更强对话能力
- 代码/数学能力提升
- 多模态版本ChatGLM-V

部署：
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
"""
```

---

## Qwen (通义千问)

```python
"""
阿里开发，中文优化

Qwen系列：
- Qwen-1.8B/7B/14B/72B
- Qwen1.5-72B (更强)
- Qwen2.5-72B (最新)
- CodeQwen (代码模型)

特点：
- 中文能力最强
- 阿里云原生集成
- 开源可商用(Qwen2.5)
- 支持长上下文

使用：
from modelscope import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
"""
```

---

## DeepSeek

```python
"""
深度求索开发，高性价比

DeepSeek系列：
- DeepSeek-7B/67B
- DeepSeek-Math (数学专用)
- DeepSeek-Coder (代码专用)
- DeepSeek-V2 (MoE架构，236B)
- DeepSeek-V3 (最新，最强开源)

DeepSeek-V2创新：
- MoE架构 (256 experts)
- 稀疏激活
- 极低训练成本

DeepSeek-V3：
- 671B参数，MoE
- FP8训练
- 极强代码/数学能力
"""
```

---

## 模型选择建议

```python
# 中文对话/写作
Qwen2.5-72B-Instruct > ChatGLM4 > Yi

# 代码生成
DeepSeek-Coder > CodeQwen > ChatGLM

# 数学推理
DeepSeek-Math > GPT-4 > Qwen

# 性价比
DeepSeek-V2/V3 > Qwen2.5 > ChatGLM

# 本地部署
Qwen2.5-7B (最低门槛)
DeepSeek-7B (最强7B)
```

---

## 面试问题

### Q1: 国产模型和GPT的区别？

**答**：
- **中文能力**：国产模型中文优化更好
- **生态集成**：阿里/字节等和自家云服务深度集成
- **开源程度**：普遍更开源
- **模型能力**：GPT-4/Claude仍最强

---

### Q2: 如何选择模型？

**答**：
1. **任务类型**：代码选CodeQwen，中文选Qwen
2. **硬件限制**：7B可本地，72B需高配
3. **成本**：DeepSeek性价比最高
4. **商用**：确认开源协议

---

## 相关知识点

- → [Llama系列](../Llama系列/README.md) - 对比参考
- → [模型训练](../模型训练/README.md) - MoE等技术
