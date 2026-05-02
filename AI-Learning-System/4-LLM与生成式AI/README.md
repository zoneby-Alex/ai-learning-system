# LLM与生成式AI模块

本模块涵盖大语言模型（LLM）和生成式AI的核心知识。

## 目录结构

```
4-LLM与生成式AI/
├── README.md                # 本文件
├── GPT系列/                 # GPT1-4/o1
├── Llama系列/               # Llama1-4
├── 国内模型/                # Qwen/DeepSeek/ChatGLM
├── 模型训练/                # Pretrain/SFT/RLHF/DPO
├── RAG系统/                 # 检索增强生成
├── Agent/                   # Agent开发
└── 推理优化/                # vLLM/TensorRT
```

---

## LLM技术全景图

```
Pretrain阶段
├── Tokenizer (BPE/WordPiece/SentencePiece)
├── Transformer Decoder
├── Scaling Law
├── MoE架构
└── 分布式训练

Post-train阶段
├── SFT (有监督微调)
├── RLHF (人类反馈强化学习)
├── DPO (直接偏好优化)
└── KTO

推理阶段
├── KV Cache
├── Flash Attention
├── INT量化 (GPTQ/AWQ)
├── TensorRT-LLM
└── vLLM/SGLang
```

---

## 主流模型对比

| 模型 | 开发者 | 参数量 | 开源 | 特点 |
|------|--------|--------|------|------|
| GPT-4 | OpenAI | ~1.8T | 否 | 闭源最强者 |
| Claude 3.5 | Anthropic | - | 否 | 长上下文强 |
| Gemini | Google | - | 否 | 多模态 |
| Llama 3 | Meta | 8B/70B | 是 | 开源标杆 |
| Qwen 2.5 | 阿里 | 7B/72B | 是 | 中文强 |
| DeepSeek-V3 | 深度求索 | 236B MoE | 是 | 高性价比 |

---

## 学习路线

```
入门:
  1. HuggingFace Transformers使用
  2. 预训练模型加载和推理
  3. 生成策略（greedy/beam/sample）

进阶:
  4. PEFT微调（LoRA/QLoRA）
  5. RLHF/DPO原理
  6. RAG系统开发

深入:
  7. 推理优化原理
  8. vLLM/SGLang源码
  9. 模型架构演进
```

---

## 核心技能要求

### 必须掌握
- [ ] Transformers库使用
- [ ] 模型加载和推理
- [ ] LoRA/QLoRA微调
- [ ] RAG系统开发
- [ ] LangChain/LlamaIndex

### 理解原理
- [ ] Attention优化
- [ ] RLHF/DPO
- [ ] 量化原理
- [ ] Scaling Law

---

## 就业方向

| 方向 | 技能要求 | 薪资范围 | 难度 |
|------|---------|----------|------|
| LLM应用工程师 | RAG/Agent/部署 | 25-45K | 中 |
| LLM微调工程师 | PEFT/分布式 | 30-50K | 中高 |
| AI Infra | CUDA/推理优化 | 40-80K | 高 |

---

## 与其他模块的关系

```
LLM与生成式AI
    ├── → 前沿方向（Agent/多模态的基础）
    ├── → 推理优化（Infra方向）
    ├── → AI应用（LangChain/Prompt）
    └── → 数学基础（交叉熵/MLE/KL散度）
```

---

*LLM是2024-2026年最热方向，就业机会最多*
