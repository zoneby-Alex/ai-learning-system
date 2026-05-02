# Agent模块

本模块包含AI Agent的核心知识。

## Agent = LLM + Planning + Memory + Tools

---

## ReAct Agent

```python
"""
ReAct = Reasoning + Acting

思考 → 行动 → 观察 → 思考 → ...

示例：
Thought: 我需要搜索相关信息
Action: search[query]
Observation: 找到了结果
Thought: 根据结果，我需要计算
Action: calculator[expr]
...
"""
```

### 代码实现

```python
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate

# 定义工具
tools = [
    Tool(name="Search", func=search_wikipedia),
    Tool(name="Calculate", func=calculate),
]

# ReAct prompt
react_prompt = PromptTemplate.from_template("""你是AI Agent，使用ReAct模式。

工具：{tools}

格式：
Thought: 你需要思考
Action: 工具名[参数]
Observation: 观察结果
...（重复）
Final Answer: 最终答案

问题：{input}
""")

# 创建Agent
agent = (
    {"input": lambda x: x["input"], "tools": lambda x: tools}
    | react_prompt
    | llm
    | StrOutputParser()
)
```

---

## LangGraph Agent

```python
"""
使用状态机编排Agent

状态：messages, current_step, actions
节点：model, action, end
边：should_continue
"""
```

---

## MCP协议

```python
"""
Anthropic提出的标准化协议

MCP Server定义：
{
    "name": "filesystem",
    "tools": [
        {"name": "read_file", "description": "读取文件", ...},
        {"name": "write_file", ...}
    ]
}

优势：
- 标准化工具定义
- 跨平台兼容
- 安全隔离
"""
```

---

## 面试问题

### Q: Agent和普通LLM的区别？

**答**：
- Agent能主动规划、使用工具
- Agent有记忆能力
- Agent能执行多步骤任务

---

## 相关知识点

- → [LangChain](../7-技术栈/LangChain/README.md)
- → [RAG](../RAG系统/README.md)
