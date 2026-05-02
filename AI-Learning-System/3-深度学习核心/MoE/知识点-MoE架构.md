# 知识点卡片：MoE架构

## 基本信息

| 属性 | 内容 |
|------|------|
| 知识点 | Mixture of Experts (MoE) |
| 掌握程度 | ★★★★☆ |
| 学习优先级 | P1 |
| 预估时间 | 5小时 |
| 面试频率 | ★★★★☆ |

---

## 核心原理

MoE用多个"专家"子网络替代单一大网络，通过路由器选择激活哪些专家：

```
MoE层 = Router + Experts + Sparse Activation

输出：y = Σ g_i(x) · E_i(x)
其中：
- g(x) = softmax(TopK(W_r · x))：路由器输出的专家权重
- E_i(x)：第i个专家的输出
- TopK：只激活权重最大的K个专家（稀疏性）
```

### 关键组件

```
1. Router（路由器/门控）
   - 决定哪些专家激活
   - 输出每个专家的分数
   - TopK选择激活的专家

2. Expert（专家）
   - 每个专家是一个FFN
   - 激活的专家并行计算
   - 不激活的专家不计算（稀疏性→省算力）

3. Load Balancing
   - 防止某些专家过度使用而其他闲置
   - 添加辅助损失鼓励均匀使用
```

---

## PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Experts (每个专家是一个FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # (B*T, C)

        # Router计算
        router_logits = self.router(x_flat)  # (B*T, num_experts)

        # TopK选择：只激活top_k个专家
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # 专家权重归一化

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 每个专家处理分配到的token
        for expert_idx in range(self.num_experts):
            # 找到被分配到该专家的token
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # 该专家处理的token
            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            # 获取对应的gate值
            gate_idx = (top_k_indices[expert_mask] == expert_idx).float().argmax(dim=-1)
            gate_values = top_k_gates[expert_mask, gate_idx].unsqueeze(-1)

            # 加权累加
            output[expert_mask] += gate_values * expert_output

        output = output.reshape(B, T, C)

        # Load balancing loss (辅助损失)
        # 鼓励token均匀分布到各专家
        load_balancing_loss = self._load_balancing_loss(router_logits)

        return output, load_balancing_loss

    def _load_balancing_loss(self, router_logits):
        """负载均衡损失：惩罚专家使用不均"""
        # 每个token选每个专家的概率
        probs = F.softmax(router_logits, dim=-1)
        # 平均使用频率
        avg_probs = probs.mean(dim=0)
        # 负载均衡 = 鼓励平均分布（即高熵）
        return (avg_probs * torch.log(avg_probs + 1e-10)).sum() * (-1)
```

---

## 代表模型

| 模型 | 总参数 | 激活参数 | 专家数 | TopK |
|------|--------|---------|--------|------|
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 |
| GPT-4 (推测) | ~1.8T | ~280B | 8-16 | 2 |
| Switch Transformer | 1.6T | - | 2048 | 1 |

---

## 面试高频问题

### Q1: MoE相比Dense模型的优势？

**答**：
- **更大参数容量**：同样的计算量下可以有更大的模型
- **稀疏激活**：每次只激活部分专家，计算效率高
- **专业化**：不同专家可以专注于不同类型的输入
- **成本效益**：Mixtral 8x7B用7B的计算量达到13B的效果

### Q2: Load Balancing为什么重要？

**答**：
如果没有负载均衡：
- 某些专家被频繁使用（过载），其他闲置（浪费）
- 路由可能崩塌到只使用少数专家
- 训练变得不稳定

解决方法：添加负载均衡损失（辅助损失），惩罚不均匀的专家使用，鼓励高熵分布。

---

## 相关知识点

- → [Transformer](../Transformer/知识点-Self-Attention.md) - MoE替代FFN
- → [Scaling Law](../../4-LLM与生成式AI/模型训练/知识点-Pretrain训练.md) - 扩展法则
