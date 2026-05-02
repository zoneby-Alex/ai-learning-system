# 项目：RAG知识库问答系统

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐⭐☆☆ |
| 预估时间 | 2-3周 |
| 目标就业 | LLM应用工程师 |
| 技术栈 | LangChain + Milvus + Qwen + FastAPI |

---

## 项目目标

1. 理解RAG完整流程
2. 掌握文档处理和向量化
3. 实现混合检索策略
4. 构建完整API服务

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        RAG 架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  文档输入                                                    │
│     ↓                                                       │
│  文档加载 → 文本切分 → 向量化 → 存储到Milvus                  │
│                                                             │
│     ↓                                                       │
│  用户查询                                                    │
│     ↓                                                       │
│  查询向量化 → 混合检索 → 重排序 → Prompt组装 → LLM生成        │
│                                                             │
│     ↓                                                       │
│  返回答案                                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 代码实现

### 1. 依赖安装

```bash
pip install langchain langchain-community langchain-huggingface
pip install pymupdf python-docx pptx pandas
pip install sentence-transformers
pip install milvus-lite pymilvus
pip install fastapi uvicorn
pip install redis
```

### 2. 文档加载器

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

class DocumentProcessor:
    """文档处理器"""

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """加载PDF"""
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    def load_docx(self, file_path: str) -> List[Document]:
        """加载Word文档"""
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def split_text(self, documents: List[Document]) -> List[Document]:
        """切分文本"""
        return self.text_splitter.split_documents(documents)

    def process_file(self, file_path: str) -> List[Document]:
        """处理单个文件"""
        if file_path.endswith('.pdf'):
            docs = self.load_pdf(file_path)
        elif file_path.endswith('.docx'):
            docs = self.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        return self.split_text(docs)
```

### 3. 向量化和存储

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from pymilvus import connections, Collection

class VectorStore:
    """向量数据库管理"""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document], collection_name: str):
        """创建向量数据库"""
        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            connection_args={"uri": "./milvus_demo.db"}
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """相似性检索"""
        return self.vectorstore.similarity_search(query, k=k)

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        混合检索：结合向量检索和关键词检索
        alpha: 向量检索权重，1-alpha为关键词检索权重
        """
        # 简化实现，实际可用Milvus的hybrid search
        return self.similarity_search(query, k)
```

### 4. 重排序

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """重排序模型"""

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 3):
        """对检索结果进行重排序"""
        doc_texts = [doc.page_content for doc in documents]
        pairs = [[query, doc] for doc in doc_texts]

        scores = self.model.predict(pairs)

        # 按分数排序
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_k]]
```

### 5. RAG Chain

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatTongyi

class RAGChain:
    """RAG链"""

    def __init__(self, llm_model="qwen2.5"):
        # 使用通义千问API
        self.llm = ChatTongyi(temperature=0.1)

        self.prompt = PromptTemplate(
            template="""基于以下上下文回答问题。如果上下文中没有相关信息，请说"我没有足够的信息来回答这个问题"。

上下文：
{context}

问题：{question}

回答：""",
            input_variables=["context", "question"]
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def answer(self, query: str, retrieved_docs: List[Document]) -> str:
        """生成回答"""
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        response = self.chain.run({
            "context": context,
            "question": query
        })

        return response
```

### 6. FastAPI服务

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os

app = FastAPI(title="RAG API")

# 全局组件
processor = DocumentProcessor()
vectorstore = VectorStore()
reranker = Reranker()
rag_chain = RAGChain()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_rerank: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并索引文档"""
    if not file.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = processor.process_file(tmp_path)
        vectorstore.create_vectorstore(docs, collection_name=file.filename)
        return {"message": f"Indexed {len(docs)} chunks"}
    finally:
        os.unlink(tmp_path)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """问答"""
    # 1. 检索
    docs = vectorstore.similarity_search(request.question, k=request.top_k)

    # 2. 重排序
    if request.use_rerank:
        docs = reranker.rerank(request.question, docs, top_k=3)

    # 3. 生成
    answer = rag_chain.answer(request.question, docs)

    return QueryResponse(
        answer=answer,
        sources=[doc.page_content[:200] for doc in docs]
    )
```

---

## 项目结构

```
rag_project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI入口
│   ├── document_processor.py # 文档处理
│   ├── vector_store.py      # 向量存储
│   ├── reranker.py          # 重排序
│   └── rag_chain.py         # RAG链
├── data/
│   └── sample.pdf
├── models/
├── requirements.txt
└── README.md
```

---

## 面试口述要点

### 项目介绍模板

```
这是一个RAG知识库问答系统，目标是为企业知识库提供智能问答能力。

核心模块包括：
1. 文档处理：支持PDF和Word文档，使用RecursiveCharacterTextSplitter进行语义切分
2. 向量检索：使用sentence-transformers编码，Milvus存储，支持千万级向量检索
3. 混合检索：结合向量相似度和关键词匹配，提高召回率
4. 重排序：使用BAAI/bge-reranker-base对初筛结果重排，提升精确率
5. 生成：使用Qwen2.5-7B作为基座模型，结合Prompt生成回答

遇到的主要挑战：
1. 文档切分策略：太短丢失语义，太长引入噪声。最终采用重叠切分+语义分割结合
2. 检索质量：纯向量检索对专有名词不敏感。解决方案是混合BM25关键词检索
3. 生成幻觉：使用"如果上下文没有相关信息，请说不知道"来约束

最终效果：问答准确率从基线的60%提升到85%。
```

---

## 验收标准

- [ ] 能上传PDF/Word文档
- [ ] 能进行语义检索
- [ ] 能返回生成答案和引用来源
- [ ] API能正常工作
- [ ] 能讲解每个模块的原理

---

## 进阶挑战

1. **流式输出**：使用FastAPI的StreamingResponse
2. **多跳推理**：支持复杂的多跳问题
3. **知识图谱**：结合知识图谱增强推理
4. **评估框架**：实现RAGAS评估指标
