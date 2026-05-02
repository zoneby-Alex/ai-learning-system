# 深度学习核心模块

本模块包含深度学习的核心知识点，是整个AI知识体系的基础。

## 目录结构

```
3-深度学习核心/
├── README.md                    # 本文件
├── BP反向传播/                  # 反向传播算法
├── 激活函数/                    # ReLU/Sigmoid/GELU等
├── CNN家族/                     # CNN基础到ResNet
├── RNN家族/                     # RNN/LSTM/GRU
├── Transformer/                 # Self-Attention/MHA/位置编码
├── ViT多模态/                  # Vision Transformer
└── Diffusion/                  # Diffusion Model
```

---

## 学习路径

```
Level 1: 基础原理
    BP反向传播 → 自动微分 → 激活函数 → 初始化 → 正则化

Level 2: CNN家族
    LeNet → AlexNet → VGG → Inception → ResNet → DenseNet

Level 3: RNN家族
    Vanilla RNN → LSTM → GRU → seq2seq → Attention

Level 4: Transformer
    Self-Attention → MHA → 位置编码 → Encoder-Decoder → Flash Attention

Level 5: 进阶主题
    ViT → Diffusion → MoE → LoRA → RLHF
```

---

## 核心知识点速查

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| BP反向传播 | ★★★★★ | P0 | ★★★★★ |
| 自动微分 | ★★★★☆ | P0 | ★★★☆☆ |
| 激活函数 | ★★★★★ | P0 | ★★★★☆ |
| 初始化策略 | ★★★★☆ | P1 | ★★★☆☆ |
| Dropout | ★★★★☆ | P1 | ★★★★☆ |
| BatchNorm | ★★★★★ | P0 | ★★★★★ |
| LayerNorm | ★★★★★ | P0 | ★★★★★ |
| CNN卷积 | ★★★★★ | P0 | ★★★★☆ |
| ResNet | ★★★★★ | P0 | ★★★★★ |
| Self-Attention | ★★★★★ | P0 | ★★★★★ |
| MHA | ★★★★★ | P0 | ★★★★★ |
| 位置编码 | ★★★★☆ | P1 | ★★★★☆ |
| Flash Attention | ★★★☆☆ | P1 | ★★★☆☆ |

---

## 面试高频考点

### 1. BP反向传播 (必须手推)
```
- 链式法则
- 计算图
- 梯度消失/爆炸原因
- 解决方案
```

### 2. BatchNorm (必须理解)
```
- 训练vs推理
- gamma/beta作用
- 为什么需要moving mean
- 与LayerNorm的区别
```

### 3. Attention/Transformer (必须理解)
```
- Self-Attention计算过程
- MHA vs Single Head
- Pre-LN vs Post-LN
- 位置编码选择
```

### 4. 优化器 (必须理解)
```
- SGD/Momentum/Adam对比
- Adam的bias correction
- 学习率调度策略
```

---

## 代码能力要求

```python
# Level 1: 会调用
model = nn.Sequential(...)
output = model(input)

# Level 2: 会修改
model = MyModel()
output = model(input)

# Level 3: 会从零实现
class MyAttention(nn.Module):
    def forward(self, x):
        # 手写Attention
        pass

# Level 4: 理解原理
# 能解释每一行代码为什么这样写
```

---

## 与其他模块的关系

```
深度学习核心
    ├── → LLM（Transformer是LLM的基础）
    ├── → CV（CNN是视觉模型基础）
    ├── → 推理优化（Attention优化）
    └── → 前沿方向（所有前沿方向的基础）
```

---

*深度学习核心是整个AI体系的基础，必须扎实掌握*
