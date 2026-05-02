# ViT与多模态模块

本模块包含Vision Transformer（ViT）和多模态学习的核心知识点。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| ViT | ★★★★☆ | P0 | ★★★★☆ |
| DeiT | ★★★☆☆ | P1 | ★★☆☆☆ |
| Swin Transformer | ★★★☆☆ | P1 | ★★☆☆☆ |
| CLIP | ★★★★☆ | P0 | ★★★★☆ |
| LLaVA | ★★★★☆ | P1 | ★★★☆☆ |

---

## ViT (Vision Transformer)

### 原理

```python
# ViT = 分块 + 线性投影 + 位置编码 + Transformer Encoder
"""
图像 -> 16x16 patches -> Linear Projection -> Transformer Encoder -> MLP Head -> 分类

步骤：
1. 将图像分成16x16个patches
2. 每个patch通过线性层得到embedding
3. 添加位置编码
4. 添加[CLS] token
5. 通过Transformer Encoder
6. 用[CLS] token的输出做分类
"""
```

### PyTorch实现

```python
import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 线性投影
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = rearrange(x, 'B E H W -> B (H W) E')  # (B, N, E)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, E)

        # 添加cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 使用cls token的输出
        return self.head(x[:, 0])
```

---

## CLIP

### 原理

```python
"""
CLIP = 图像编码器 + 文本编码器 + 对比学习

训练目标：让配对的图像-文本embedding更接近，让不配对的更远

损失：对称交叉熵
L = 0.5 * (loss_i2t + loss_t2i)

其中：
- loss_i2t = -log exp(sim(img_i, text_i)) / Σ_j exp(sim(img_i, text_j))
- sim = cosine(img, text) / temperature
"""
```

### 使用示例

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图像
image = Image.open("example.jpg")
inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 图像-文本相似度
```

---

## LLaVA

### 原理

```python
"""
LLaVA = Vision Encoder + Projector + LLM

架构：
1. Vision Encoder: CLIP ViT处理图像
2. Projector: 将视觉特征映射到LLM的embedding空间
3. LLM: 接收图像+文本，输出文本
"""
```

### 使用示例

```python
from transformers import AutoModelForVision2Seq
from PIL import Image

model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 输入图像和文本
image = Image.open("example.jpg")
prompt = "请描述这张图片"
inputs = processor(text=prompt, images=image, return_tensors="pt")

# 生成
outputs = model.generate(**inputs, max_new_tokens=100)
```

---

## 面试高频问题

### Q1: ViT和CNN的区别？

| 方面 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 局部性/平移不变性 | 无 |
| 数据需求 | 少 | 多（需大数据） |
| 序列长度 | - | N = (H/16)×(W/16) |
| 全局建模 | 需大感受野 | 自然建模 |
| 计算量 | O(HW) | O((HW)²) |

---

### Q2: ViT为什么需要更多数据？

**答**：
- CNN有先验的局部性和平移不变性
- ViT没有归纳偏置，需要从数据中学习
- 需要更多数据让ViT学习这些模式

---

### Q3: CLIP的训练方式？

**答**：
对比学习：
1. 编码图像和文本
2. 计算余弦相似度
3. 最大化配对的相似度，最小化不配对的相似度

---

## 相关知识点

- → [Transformer](../Transformer/README.md) - Encoder架构
- → [Self-Attention](../Transformer/知识点-Self-Attention.md) - 注意力机制
- → [多模态前沿](../5-前沿方向/多模态/README.md) - 前沿方向
