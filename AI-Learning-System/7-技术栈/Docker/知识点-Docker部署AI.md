# 知识点卡片：Docker部署AI模型

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | Docker容器化AI模型部署 |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P1 |
| 预估时间 | 6小时 |

---

## Dockerfile模板

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制模型代码
COPY . .

# 启动FastAPI服务
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## GPU支持

```bash
# 需要nvidia-docker
docker run --gpus all -p 8000:8000 my-model:latest

# 指定GPU
docker run --gpus '"device=0,1"' -p 8000:8000 my-model:latest
```

---

## Docker Compose

```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports: ["8000:8000"]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: --model meta-llama/Llama-2-7b-hf --max-model-len 4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

---

## 多阶段构建（优化镜像大小）

```dockerfile
# 构建阶段
FROM python:3.10 as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行阶段
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
COPY . /app
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "/app/main.py"]
```

---

## 最佳实践

- 使用`.dockerignore`排除模型权重和缓存
- 模型权重通过volume挂载，不打入镜像
- 设置健康检查：`HEALTHCHECK`端点
- 使用`--restart unless-stopped`确保服务自恢复

---

## 相关知识点

- → [FastAPI部署](../../7-技术栈/README.md)
- → [vLLM部署](../vLLM/知识点-vLLM部署.md)
