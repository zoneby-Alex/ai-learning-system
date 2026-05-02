# 技术栈模块

本模块介绍工业界常用的AI技术栈。

## 目录结构

```
7-技术栈/
├── README.md                    # 本文件
├── PyTorch/                     # 深度学习框架
├── HuggingFace/                 # 模型库生态
├── LangChain/                   # LLM应用框架
├── vLLM/                        # 推理框架
├── Docker/                      # 容器化
└── FastAPI/                     # API开发
```

---

## 学习优先级

```
★★★★★ 必须精通
  1. PyTorch - 深度学习框架基础
  2. HuggingFace Transformers - 模型调用
  3. Git - 版本控制
  4. Docker - 容器化
  5. Linux - 服务器操作
  6. FastAPI - API开发
  7. Redis - 缓存

★★★★ 重要
  8. LangChain/LlamaIndex - LLM应用
  9. PEFT - 模型微调
  10. vLLM - 推理框架
  11. Milvus - 向量数据库
  12. DeepSpeed - 分布式训练

★★★ 加分项
  13. CUDA/Triton - 推理优化
  14. Kubernetes - 云原生
  15. Kafka - 消息队列
```

---

## 企业真实使用场景

### 场景1: AI应用公司（智能客服/文档处理）

```
技术栈: FastAPI + LangChain + Redis + Milvus + Docker

典型工作:
- RAG系统开发
- Prompt优化
- Agent编排
- API开发
```

### 场景2: AI Infra公司（推理平台）

```
技术栈: CUDA + TensorRT + vLLM + Kubernetes

典型工作:
- 推理性能优化
- 模型量化
- GPU调度
- 模型服务化
```

### 场景3: 算法公司（内容审核/推荐）

```
技术栈: PyTorch + timm + DeepSpeed + W&B

典型工作:
- 模型训练
- 调参优化
- 实验管理
- 模型部署
```

---

## 快速上手路线

```
第一周: Python工程化
  → FastAPI + Redis + Docker

第二周: 深度学习框架
  → PyTorch + timm

第三周: LLM应用
  → HuggingFace + LangChain + RAG

第四周: 模型微调与部署
  → PEFT + DeepSpeed + vLLM
```

---

## 工具速查

```python
# 深度学习
import torch
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl

# LLM应用
from langchain.chains import LLMChain
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings

# 实验管理
import wandb
from tensorboard import SummaryWriter

# API
from fastapi import FastAPI
import redis

# 部署
# docker build -t myapp:latest .
# docker run -p 8000:8000 myapp:latest
```
