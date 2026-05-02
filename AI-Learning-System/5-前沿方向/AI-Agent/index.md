# 前沿方向：AI Agent

## 基本信息

| 属性 | 内容 |
|------|------|
| 方向 | AI Agent / Autonomous Agent |
| 热度 | ★★★★★ |
| 就业前景 | 极佳 |
| 学习难度 | ★★★☆☆ |

---

## 核心技术

```
AI Agent = LLM + Planning + Memory + Tools

├── Planning（规划能力）
│   ├── ReAct（推理+行动）
│   ├── CoT（思维链）
│   ├── ToT（思维树）
│   └── LangGraph（状态机编排）
│
├── Memory（记忆能力）
│   ├── 短期记忆（上下文）
│   ├── 长期记忆（向量存储）
│   └── 混合记忆
│
└── Tools（工具能力）
    ├── MCP（Model Context Protocol）
    ├── Function Calling
    └── Tool Learning
```

---

## 技术对比

### Agent框架对比

| 框架 | 开发公司 | 特点 | 适用场景 |
|------|---------|------|---------|
| LangChain | LangChain Inc | 生态最全 | RAG+Agent |
| LangGraph | LangChain Inc | 状态机设计 | 复杂工作流 |
| AutoGen | Microsoft | 多Agent协作 | 对话系统 |
| CrewAI | CrewAI | 多Agent角色 | 团队协作 |
| Claude MCP | Anthropic | 原生MCP支持 | Claude集成 |

### Agent架构对比

```
简单Agent（单轮）：
  Input → LLM → Output

RAG Agent（检索增强）：
  Input → 检索 → LLM → Output

ReAct Agent（推理+行动）：
  Input → Think → Act → Observe → ... → Output

Multi-Agent（多Agent协作）：
  Agent1 ↔ Agent2 ↔ Agent3
       ↓
     共享Memory/Tools
```

---

## 代码示例

### ReAct Agent实现

```python
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 定义工具
def search_wikipedia(query: str) -> str:
    """搜索维基百科"""
    # 实际调用Wikipedia API
    return f"ikipedia result for: {query}"

def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [
    Tool(name="Search", func=search_wikipedia, description="搜索维基百科"),
    Tool(name="Calculate", func=calculate, description="计算数学表达式")
]

# ReAct提示词
react_prompt = PromptTemplate.from_template("""你是一个AI Agent，使用ReAct模式。

工具：
{tools}

使用以下格式：
Thought: 你需要思考下一步
Action: 工具名称
Action Input: 工具输入
Observation: 工具输出
...（重复直到得到答案）
Final Answer: 最终答案

用户问题：{input}

开始！""")

# 创建Agent
llm = ChatOpenAI(model="gpt-4")
agent = (
    {"input": lambda x: x["input"], "tools": lambda x: tools}
    | react_prompt
    | llm
    | StrOutputParser()
)
```

### LangGraph Agent实现

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    current_step: str
    actions: List[str]

def should_continue(state: AgentState) -> str:
    """决定是否继续"""
    if len(state["actions"]) > 5:  # 限制步数
        return "end"
    return "continue"

def call_model(state: AgentState) -> AgentState:
    """调用模型"""
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def take_action(state: AgentState) -> AgentState:
    """执行动作"""
    last_msg = state["messages"][-1]
    actions = state["actions"] + [last_msg]
    return {"actions": actions}

# 构建图
graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("action", take_action)
graph.set_entry_point("model")
graph.add_conditional_edges("model", should_continue, {"continue": "action", "end": END})
graph.add_edge("action", "model")

app = graph.compile()
```

---

## MCP (Model Context Protocol)

### 什么是MCP？

MCP是Anthropic提出的标准化协议，用于连接AI模型和外部工具/数据源。

### MCP架构

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP 架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Host App（Claude Desktop / Cursor / etc）                  │
│       │                                                     │
│       │ MCP Protocol（JSON-RPC）                           │
│       ↓                                                     │
│   MCP Server                                                │
│       ├── File System Server                               │
│       ├── Git Server                                        │
│       ├── Database Server                                   │
│       └── Custom Server                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### MCP工具定义

```json
{
  "name": "filesystem",
  "description": "与本地文件系统交互",
  "tools": [
    {
      "name": "read_file",
      "description": "读取文件内容",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "文件路径"}
        }
      }
    },
    {
      "name": "write_file",
      "description": "写入文件",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "content": {"type": "string"}
        }
      }
    }
  ]
}
```

---

## 面试高频问题

### Q1: 什么是Agent？和LLM有什么区别？

**答**：
- **LLM**：被动响应输入，生成文本
- **Agent**：主动规划、使用工具、完成复杂任务

Agent = LLM + 规划 + 记忆 + 工具

---

### Q2: ReAct vs CoT vs ToT？

**答**：
- **CoT (Chain of Thought)**：只思考不行动，适合简单推理
- **ReAct (Reasoning + Acting)**：思考→行动→观察，适合需要交互的任务
- **ToT (Tree of Thought)**：探索多条推理路径，适合复杂问题

```python
# CoT: Think step by step
"I think the answer is..."

# ReAct: Think + Act + Observe
"Thought: I need to search
 Action: search[query]
 Observation: Found result
 ..."

# ToT: Explore multiple paths
"Branch 1: path1...
 Branch 2: path2...
 Branch 3: path3...
"
```

---

### Q3: 如何解决Agent的幻觉问题？

**答**：
1. **Tool的可靠输出**：使用确定性工具而非LLM
2. **验证机制**：让Agent验证自己的输出
3. **约束输出**：通过Prompt限制Agent行为
4. **人机协作**：关键决策需要人类确认
5. **检索增强**：基于真实文档而非记忆

---

## 学习资源

### 开源项目
- [LangChain/LangGraph](https://github.com/langchain-ai/langchain)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/crewAIai/crewAI)
- [AgentVerse](https://github.com/open-compass/AgentVerse)

### 必读论文
1. ReAct (2023) ⭐⭐⭐⭐ - 推理+行动范式
2. Tool Learning (2023) ⭐⭐⭐⭐ - 工具学习
3. AutoGen (2023) ⭐⭐⭐⭐ - 多Agent框架

---

## 就业方向

| 岗位 | 技能要求 | 薪资范围 |
|------|---------|----------|
| Agent开发工程师 | LangChain/MCP/Function Calling | 30-50K |
| AI应用工程师 | RAG/Prompt/Agent | 25-45K |
| AI架构师 | 系统设计/Agent/Infra | 50-80K |

---

*AI Agent是2024-2026年最热方向，就业机会极多*
