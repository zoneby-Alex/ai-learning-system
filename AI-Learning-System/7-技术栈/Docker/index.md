# Docker模块

本模块介绍Docker容器化技术。

## 核心概念

```
Docker = 镜像 + 容器 + 仓库

核心：
├── Image: 镜像，只读模板
├── Container: 容器，镜像的运行实例
├── Dockerfile: 镜像构建脚本
└── Docker Compose: 多容器编排
```

---

## 基础命令

```bash
# 镜像操作
docker build -t myapp:latest .           # 构建镜像
docker images                               # 列出镜像
docker rmi myapp:latest                    # 删除镜像

# 容器操作
docker run -d --name myapp -p 8000:8000 myapp  # 运行容器
docker ps -a                               # 列出容器
docker stop myapp                          # 停止容器
docker rm myapp                            # 删除容器
docker logs -f myapp                       # 查看日志

# 进入容器
docker exec -it myapp bash
```

---

## Dockerfile示例

```dockerfile
# Python应用
FROM python:3.10-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 环境变量
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]
```

---

## Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: db
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## AI应用Dockerfile

```dockerfile
# RAG应用
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 启动
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 相关知识点

- → [FastAPI](../7-技术栈/FastAPI/README.md) - API开发
- → [MLOps](../9-学习资料/GitHub/README.md) - 部署相关
