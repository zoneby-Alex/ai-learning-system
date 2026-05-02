# 项目：从零实现 Vision Transformer (ViT)

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐⭐☆☆ |
| 预估时间 | 1-2周 |
| 目标就业 | 算法工程师 |
| 技术栈 | PyTorch + timm |

---

## 项目目标

1. 从零实现ViT完整架构
2. 理解Patch Embedding和位置编码
3. 在CIFAR-10上训练验证
4. 可视化Attention Map

---

## 核心实现

```python
import torch
import torch.nn as nn
import math
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = rearrange(x, 'B E H W -> B (H W) E')
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # 取[CLS] token
        return self.head(x)

# 训练
model = ViT(img_size=32, patch_size=4, num_classes=10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

---

## Attention Map可视化

```python
def visualize_attention(model, image, head_idx=0):
    """可视化某个注意力头的Attention Map"""
    # 注册hook获取attention weights
    attentions = []
    def hook_fn(module, input, output):
        attentions.append(output)
    # 对TransformerEncoderLayer的self_attn注册hook
    # ... 具体实现略
    return attention_maps  # shape: (n_layers, n_patches+1, n_patches+1)
```

---

## 验收标准

- [ ] 准确率 > 80%（CIFAR-10，从零训练）
- [ ] 能解释Patch Embedding原理
- [ ] 实现了Attention Map可视化
- [ ] 能对比CNN vs ViT

## 面试Q&A

### Q: ViT相比CNN为什么需要更多数据？
**答**：CNN有内置的局部性和平移不变性归纳偏置，而ViT没有这些先验，需要从大量数据中自己学习。因此ViT在ImageNet这样的大数据集上才优于CNN。

### Q: [CLS] token的作用？
**答**：[CLS]是BERT风格的分类token，经过所有Transformer层后聚合了全局信息，用于最终分类。
