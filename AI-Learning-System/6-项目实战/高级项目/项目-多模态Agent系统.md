# 项目：多模态Agent系统

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐⭐⭐⭐ |
| 预估时间 | 4-6周 |
| 目标就业 | LLM应用/Agent工程师 |
| 技术栈 | LLaVA/InternVL + LangChain + MCP |

---

## 项目目标

1. 实现多模态理解（图像/文档/视频）
2. 构建Agent工具调用能力
3. 实现多步骤复杂任务

---

## 技术架构

```
用户输入（文本/图像/文档）
        ↓
多模态理解层（LLaVA/InternVL）
        ↓
Agent规划层（ReAct/CoT）
        ↓
工具调用层（MCP/Git/代码执行）
        ↓
执行反馈 → 继续规划
```

---

## 代码实现

### 1. 多模态Agent

```python
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from transformers import AutoModelForVision2Seq, AutoProcessor

class MultimodalAgent:
    def __init__(self):
        # 多模态模型
        self.vlm = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        # 工具
        self.tools = [
            Tool(name="WebSearch", func=web_search),
            Tool(name="CodeExec", func=execute_code),
            Tool(name="FileRead", func=read_file),
        ]

        # Agent
        self.agent = self.create_agent()

    def create_agent(self):
        prompt = """你是多模态Agent。

可用工具：{tools}

用户可能输入：
- 文本问题
- 图片（请描述图片内容）
- 文档（PDF/Word）

请规划使用哪些工具完成任务。
"""
        return AgentExecutor.from_agent_and_tools(
            agent=..., tools=self.tools, prompt=prompt
        )

    def process(self, input_data, images=None):
        # 多模态理解
        if images:
            # 提取图像信息
            image_features = self.vlm.encode(images)

        # Agent推理
        result = self.agent.run(input_data)
        return result
```

### 2. 文档理解Pipeline

```python
from PIL import Image
import torch

def process_document(image_path, query):
    """处理文档图像并回答问题"""
    image = Image.open(image_path)

    # 使用InternVL处理
    model = AutoModelForVision2Seq.from_pretrained("OpenGVLab/InternVL2-4B")

    inputs = {
        "pixel_values": processor(image),
        "input_ids": tokenizer(query),
    }

    output = model.generate(**inputs)
    return tokenizer.decode(output[0])
```

---

## 项目结构

```
multimodal_agent/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── multimodal_agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── web_search.py
│   │   ├── code_exec.py
│   │   └── file_ops.py
│   └── mcp/
│       └── ...
├── models/
├── data/
├── requirements.txt
└── README.md
```

---

## 验收标准

- [ ] 能处理文本+图像输入
- [ ] 能调用多种工具
- [ ] 能完成多步骤任务
- [ ] 能回答文档相关问题

---

## 面试口述要点

```
这个项目实现了一个多模态Agent系统：

1. 多模态理解层：
   - 使用InternVL处理图像和文本
   - 支持文档、表格、图表理解

2. Agent规划层：
   - 使用ReAct模式进行推理
   - 自动规划工具使用顺序

3. 工具层：
   - MCP协议定义标准化工具
   - 支持搜索、代码执行、文件操作

4. 实际应用场景：
   - 智能客服（图文问答）
   - 文档处理（合同审查/报表分析）
```

---

## 进阶挑战

1. 视频理解
2. 语音交互
3. 实时流式处理
