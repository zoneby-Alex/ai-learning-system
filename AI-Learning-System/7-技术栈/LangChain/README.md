# LangChain模块

本模块介绍LangChain LLM应用开发框架。

## 核心组件

```
LangChain = Models + Prompts + Chains + Agents + Memory

组件：
├── Models: LLM接口
├── Prompts: Prompt模板
├── Chains: 链式调用
├── Agents: Agent执行
└── Memory: 记忆存储
```

---

## 快速上手

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 创建LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 创建Prompt
prompt = PromptTemplate(
    template="请将以下中文翻译成英文：{text}",
    input_variables=["text"]
)

# 创建Chain
chain = prompt | llm

# 执行
result = chain.invoke({"text": "你好世界"})
```

---

## RAG开发

```python
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 向量数据库
vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings()
)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

result = qa_chain.invoke({"query": "问题"})
```

---

## Agent开发

```python
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import ChatPromptTemplate

# 定义工具
tools = [
    Tool(name="Search", func=search, description="搜索信息"),
]

# 创建Agent
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
    ("agent_scratchpad", "{agent_scratchpad}"),
])

agent = create_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "问题"})
```

---

## LangGraph

```python
from langgraph.graph import StateGraph, END

# 定义状态
class AgentState(TypedDict):
    messages: list

# 创建图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
graph.add_edge("agent", END)

app = graph.compile()
```

---

## 相关知识点

- → [RAG](../4-LLM与生成式AI/RAG系统/README.md)
- → [Agent](../4-LLM与生成式AI/Agent/README.md)
