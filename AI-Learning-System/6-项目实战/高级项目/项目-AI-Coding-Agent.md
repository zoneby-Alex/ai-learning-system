# 项目：AI Coding Agent

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐⭐⭐⭐ |
| 预估时间 | 3-4周 |
| 目标就业 | Agent开发工程师 |
| 技术栈 | Claude/GPT-4 + REPL + Git + MCP |

---

## 项目目标

1. 实现代码生成Agent
2. 实现代码审查功能
3. 实现Bug自动修复
4. 完整工具链集成

---

## 核心实现

```python
from langchain.agents import AgentExecutor, Tool
from langchain.tools import BaseTool
import subprocess
import tempfile
import os

class CodeExecutionTool(BaseTool):
    name = "python_repl"
    description = "执行Python代码并返回结果"

    def _run(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(['python', f.name], capture_output=True, text=True, timeout=30)
            os.unlink(f.name)
            return result.stdout + result.stderr

class GitTool(BaseTool):
    name = "git_operations"
    description = "执行Git操作"

    def _run(self, command: str) -> str:
        result = subprocess.run(['git'] + command.split(), capture_output=True, text=True, cwd='./workspace')
        return result.stdout + result.stderr

class CodeReviewAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            CodeExecutionTool(),
            GitTool(),
            Tool(name="read_file", func=self.read_file, description="读取文件"),
            Tool(name="write_file", func=self.write_file, description="写入文件"),
        ]

    def generate_code(self, task_description):
        """根据描述生成代码"""
        prompt = f"""你是一个AI编程助手。根据以下需求生成代码：
需求：{task_description}

要求：
1. 完整的可运行代码
2. 包含错误处理
3. 有适当的注释

请直接输出代码："""
        return self.llm.invoke(prompt)

    def review_code(self, code):
        """代码审查"""
        prompt = f"""审查以下代码，检查：
1. 潜在的bug
2. 安全问题
3. 性能优化建议
4. 代码风格

代码：
{code}

请列出发现的问题："""
        return self.llm.invoke(prompt)

    def fix_bug(self, code, error_message):
        """修复bug"""
        prompt = f"""以下代码有bug，请修复：

代码：
{code}

错误信息：
{error_message}

请输出修复后的完整代码："""
        return self.llm.invoke(prompt)

    def run_agent_loop(self, task):
        """完整Agent循环：生成→执行→审查→修复"""
        # 1. 生成代码
        code = self.generate_code(task)

        # 2. 执行验证
        result = self.tools[0]._run(code)
        if 'Error' in result:
            # 3. 修复
            code = self.fix_bug(code, result)
            result = self.tools[0]._run(code)

        # 4. 审查
        review = self.review_code(code)

        return {"code": code, "execution": result, "review": review}
```

---

## 验收标准

- [ ] 能根据自然语言描述生成可运行代码
- [ ] 能自动检测并修复简单bug
- [ ] 有代码审查功能
- [ ] 工具链完整（执行/文件/Git）

## 面试Q&A

### Q: AI Coding Agent的核心挑战？
1. **代码正确性**：生成的代码可能语法正确但逻辑错误
2. **安全性**：执行未知代码有风险（沙箱隔离）
3. **上下文管理**：大型项目的代码理解
4. **迭代修复**：单次修复可能引入新bug

### Q: 如何保证代码执行安全？
- Docker沙箱隔离
- 资源限制（CPU/内存/时间）
- 网络隔离
- 文件系统限制
