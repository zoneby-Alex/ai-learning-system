# 长上下文模块

本模块介绍长上下文处理方向。

## 技术挑战

```
长上下文问题：
├── 计算复杂度：O(N²) Attention
├── 显存占用：N×N矩阵
├── 位置编码外推：训练长度≠推理长度
└── 注意力分散：无关token干扰
```

---

## 解决方案

### 位置编码外推

```python
"""
位置编码进化：

1. Sinusoidal (原生)
   - 可泛化到训练长度
   - 外推效果差

2. RoPE (Llama/Qwen)
   - 旋转式，相对位置
   - 外推较好

3. ALiBi
   - 线性偏置
   - 外推好

4. 位置插值
   - 将大位置缩放到训练范围
   - 配合fine-tuning
"""
```

---

## 高效注意力

```python
"""
Sparse Attention：
- 局部窗口
- 随机稀疏
- 全局token

Approximate Attention：
- Flash Attention (精确)
- Linformer (近似)
- Performer (近似)
"""
```

---

## 代表模型

| 模型 | 上下文长度 | 技术 |
|------|-----------|------|
| Claude | 200K | - |
| Gemini | 1M | - |
| Kimi | 200K | - |
| Qwen2.5 | 128K | RoPE |
| Llama 3 | 128K | RoPE |

---

## 面试问题

### Q: 如何处理长上下文？

**答**：
1. 位置编码选择RoPE/ALiBi
2. 稀疏注意力/局部窗口
3. 显存优化（PagedAttention）
4. 检索+生成（RAG）

---

## 相关知识点

- → [RoPE](../3-深度学习核心/Transformer/README.md)
- → [Flash Attention](../3-深度学习核心/Transformer/README.md)
