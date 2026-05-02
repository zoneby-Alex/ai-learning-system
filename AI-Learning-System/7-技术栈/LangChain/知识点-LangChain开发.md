# 知识点卡片：LangChain开发

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | LangChain / LangGraph / LlamaIndex |
| 掌握程度 | ★★★★★ |
| 学习优先级 | P0 |
| 预估时间 | 8小时 |

---

## 核心概念

```
LangChain组件：
├── Models（模型）：LLM/ChatModel/Embeddings
├── Prompts（提示词）：PromptTemplate/ChatPromptTemplate
├── Chains（链）：串联多个组件
├── Memory（记忆）：对话历史管理
├── Indexes（索引）：文档加载/向量存储/检索
├── Agents（代理）：基于LLM的工具调用
└── Callbacks（回调）：日志/监控
```

---

## 代码示例

### Chain

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    template="将以下文本翻译成{target_language}:\n{text}",
    input_variables=["text", "target_language"]
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run({"text": "Hello World", "target_language": "中文"})
```

### RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
answer = qa_chain.invoke("What is the capital?")
```

### Agent

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
    return f"搜索结果: 关于'{query}'的信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

agent = initialize_agent(
    tools=[search, calculate],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
agent.run("123 * 456 是多少？搜索验证一下")
```

### LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    next_step: str

def router(state):
    """根据状态决定下一步"""
    last_msg = state["messages"][-1]
    if "搜索" in last_msg:
        return "search"
    return "respond"

graph = StateGraph(State)
graph.add_node("think", think_node)
graph.add_node("search", search_node)
graph.add_node("respond", respond_node)
graph.add_conditional_edges("think", router, {"search": "search", "respond": "respond"})
graph.add_edge("search", "think")
graph.add_edge("respond", END)
graph.set_entry_point("think")

app = graph.compile()
result = app.invoke({"messages": ["帮我查一下今天的天气"]})
```

---

## 相关知识点

- → [RAG系统](../../4-LLM与生成式AI/RAG系统/知识点-RAG核心原理.md)
- → [Agent开发](../../5-前沿方向/AI-Agent/README.md)
