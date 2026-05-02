# 知识点卡片：RAG核心原理

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | RAG (Retrieval-Augmented Generation) |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P0 |
| 预估时间 | 6小时 |
| 面试频率 | ★★★★☆ |

---

## 核心原理

RAG = Retrieval（检索） + Generation（生成）

**核心思想**：让LLM在生成答案前，先从外部知识库检索相关信息，作为上下文输入给LLM。

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query ──────────────────────────────────────────────────┐  │
│      │                                                      │  │
│      ▼                                                      │  │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐              │  │
│  │  检索   │────▶│  重排   │────▶│  生成   │              │  │
│  └─────────┘     └─────────┘     └─────────┘              │  │
│      │                              │                      │  │
│      ▼                              ▼                      │  │
│  ┌─────────┐                  ┌─────────┐                  │  │
│  │ 知识库  │                  │   LLM   │                  │  │
│  └─────────┘                  └─────────┘                  │  │
│                                                             │  │
└─────────────────────────────────────────────────────────────┘
```

---

## RAG vs Fine-tuning vs Prompting

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| RAG | 知识可更新/可解释/成本低 | 依赖检索质量 | 需要最新知识 |
| Fine-tuning | 效果稳定/个性化 | 成本高/更新慢 | 风格适配 |
| Prompting | 灵活/快速 | 知识有限 | 通用任务 |

---

## 核心组件

### 1. 文档处理

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 文本切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？"]
)

chunks = splitter.split_documents(documents)
```

### 2. 向量化

```python
from langchain_huggingface import HuggingFaceEmbeddings

# 使用sentence-transformers
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 将文本转为向量
vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
```

### 3. 向量检索

```python
from langchain_community.vectorstores import Milvus

# 存储到Milvus
vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="knowledge_base"
)

# 检索
results = vectorstore.similarity_search(query="...", k=5)
```

### 4. 混合检索

```python
def hybrid_search(query, vectorstore, bm25_retriever, k=5, alpha=0.5):
    """
    混合检索：结合向量检索和关键词检索
    alpha: 向量权重
    """
    # 向量检索
    vector_results = vectorstore.similarity_search(query, k=k)

    # BM25关键词检索
    bm25_results = bm25_retriever.get_relevant_documents(query, k=k)

    # 融合（简单RRF）
    rrf_scores = {}
    for rank, doc in enumerate(vector_results):
        rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (k + rank)

    for rank, doc in enumerate(bm25_results):
        rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (k + rank)

    # 按分数排序
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]
```

### 5. 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank_documents(query, documents, top_k=3):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)

    # 按分数排序
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]
```

### 6. 生成

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    template="""基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答：""",
    input_variables=["context", "question"]
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_answer(question, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return chain.run({"context": context, "question": question})
```

---

## 面试高频问题

### Q1: RAG的流程是什么？

**答**：
```
1. 文档处理：加载 → 清洗 → 切分
2. 向量化：将文本转为稠密向量
3. 存储：存入向量数据库
4. 检索：根据query检索相关文档
5. 重排：对检索结果重排序
6. 生成：将相关文档作为上下文，让LLM生成回答
```

---

### Q2: 文档切分的策略有哪些？

**答**：
- **固定大小切分**：简单但可能切断语义
- **语义切分**：按段落/句子切分，保持语义完整
- **重叠切分**：相邻chunk有重叠，保持连续性
- **结构感知切分**：根据标题/段落等结构切分

---

### Q3: 如何评估RAG效果？

**答**：
- **检索指标**：Recall@K, MRR, NDCG
- **生成指标**：RAGAS, BLEU, ROUGE
- **端到端**：人工评估/用户反馈

---

### Q4: RAG的局限性有哪些？

**答**：
1. **检索质量依赖**：检索不好，生成也不会好
2. **上下文长度限制**：无法引入太多文档
3. **多跳推理困难**：涉及多个文档的问题
4. **实时性**：知识库更新需要重新索引

---

### Q5: 如何优化RAG效果？

**答**：
1. **检索优化**：
   - 混合检索（向量+关键词）
   - 重排序
   - 查询扩展/改写

2. **生成优化**：
   - Prompt工程
   - 控制生成长度
   - 后处理过滤

3. **知识库优化**：
   - 更好的切分策略
   - 元数据增强
   - 知识图谱增强
```

---

## 实践建议

```python
# 1. 快速原型
from langchain import LangChain

# 2. 生产环境
# - 使用Milvus/Pinecone等向量数据库
# - 实现混合检索
# - 添加监控和评估

# 3. 评估框架
# - RAGAS (https://github.com/explodinggradients/ragas)
# - Trulens
```

---

## 相关知识点

- → [向量数据库](../技术栈/向量数据库.md) - Milvus/FAISS
- → [Embedding模型](./Embedding模型.md) - 文本向量化
- → [Agent](../前沿方向/Agent/README.md) - RAG+Agent
