# Diffusion模型模块

本模块包含扩散模型（Diffusion Model）的核心知识点。

## 知识点列表

| 知识点 | 掌握程度 | 优先级 | 面试频率 |
|--------|---------|--------|---------|
| DDPM | ★★★★☆ | P0 | ★★★★☆ |
| DDIM | ★★★☆☆ | P1 | ★★☆☆☆ |
| Stable Diffusion | ★★★★☆ | P1 | ★★★☆☆ |
| Score Matching | ★★★☆☆ | P2 | ★★☆☆☆ |

---

## 原理概述

```
Diffusion = 前向过程 + 反向过程

前向过程（加噪）：
x_0 -> x_1 -> x_2 -> ... -> x_T
（逐步添加高斯噪声，最终趋近纯噪声）

反向过程（去噪）：
x_T -> x_{T-1} -> ... -> x_1 -> x_0
（学习从噪声恢复图像）
```

---

## DDPM (Denoising Diffusion Probabilistic Models)

### 前向过程

```python
# 给定x_0，逐步添加噪声
# q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

# 可以直接采样x_t（不需要逐步）
# x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε,  ε ~ N(0, I)
# 其中ᾱ_t = Π β_i

def forward_process(x_0, t, alphas_cumprod):
    """前向过程：直接计算x_t"""
    noise = torch.randn_like(x_0)
    sqrt_alphas_t = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alphas_t = (1 - alphas_cumprod[t]) ** 0.5
    return sqrt_alphas_t * x_0 + sqrt_one_minus_alphas_t * noise, noise
```

### 反向过程

```python
# 反向过程：学习 p_θ(x_{t-1} | x_t)
# p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), Σ_θ(x_t, t))

class DDPM(nn.Module):
    def __init__(self, network, betas, T=1000):
        super().__init__()
        self.network = network  # UNet
        self.betas = betas
        self.T = T
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def loss(self, x_0):
        """去噪损失"""
        batch_size = x_0.size(0)
        t = torch.randint(0, self.T, (batch_size,))

        # 前向过程
        x_t, noise = self.forward_process(x_0, t)

        # 预测噪声
        noise_pred = self.network(x_t, t)

        # MSE损失
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, shape):
        """DDPM采样"""
        x_t = torch.randn(shape)
        for t in reversed(range(self.T)):
            z = torch.randn_like(x_t) if t > 0 else 0
            noise_pred = self.network(x_t, t)
            x_{t-1} = self.ddim_step(x_t, noise_pred, t)
            x_t = x_{t-1}
        return x_0
```

---

## UNet（噪声预测网络）

```python
class UNet(nn.Module):
    """用于噪声预测的UNet架构"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=128):
        super().__init__()
        self.encoder = nn.ModuleList([
            ResidualBlock(in_channels, base_channels),
            ResidualBlock(base_channels, base_channels*2),
            ResidualBlock(base_channels*2, base_channels*4),
        ])
        self.decoder = nn.ModuleList([
            ResidualBlock(base_channels*4, base_channels*2),
            ResidualBlock(base_channels*2, base_channels),
        ])
        self.time_mlp = TimeEmbedding(base_channels)

    def forward(self, x, t):
        # 编码
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)

        # 解码 + skip connections
        skips = skips[::-1]
        for i, block in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2)
            x = torch.cat([x, skips[i]], dim=1)
            x = block(x)

        return x
```

---

## DDIM (Denoising Diffusion Implicit Models)

### 加速采样

```python
# DDPM需要1000步采样，太慢
# DDIM通过确定性采样加速，可减少到50步

def ddim_step(self, x_t, noise_pred, t):
    """DDIM单步"""
    alpha_t = self.alphas[t]
    alpha_prev = self.alphas[t-1] if t > 0 else 1

    # 预测x_0
    pred_x_0 = (x_t - (1-alpha_t)**0.5 * noise_pred) / alpha_t**0.5

    # DDIM更新
    sigma_t = 0  # DDIM是确定性的
    direction = (1 - alpha_prev - sigma_t**2)**0.5 * noise_pred

    x_{t-1} = alpha_prev**0.5 * pred_x_0 + direction + sigma_t * torch.randn_like(x_t)

    return x_{t-1}
```

---

## Stable Diffusion

### 架构

```python
"""
Stable Diffusion = VAE + CLIP Text Encoder + UNet + Classifier-Free Guidance

工作流程：
1. Text -> CLIP embedding
2. 随机噪声 -> UNet(latent space)
3. UNet在CLIP embedding条件下去噪
4. VAE decode -> 图像
"""
```

### 使用示例

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda")

image = pipe("a cat sitting on a chair").images[0]
```

---

## 面试高频问题

### Q1: Diffusion和GAN的区别？

| 方面 | GAN | Diffusion |
|------|-----|-----------|
| 训练 | 对抗训练，难 | 简单MSE |
| 生成质量 | 好 | 非常好 |
| 采样速度 | 快（一步） | 慢（多步） |
| 多样性 | 可能模式崩塌 | 多样性好 |

---

### Q2: 为什么需要这么多次采样？

**答**：
- 前向过程是马尔可夫的，每步只加一点噪声
- 反向过程需要逐步去噪
- DDIM等方法可以在保持质量的同时减少步数

---

### Q3: Classifier-Free Guidance的作用？

**答**：
在无条件生成模型中引导条件生成：
```python
# 无引导：noise_pred = model(x_t, t)
# 有引导：noise_pred = model(x_t, t, c) + w * (model(x_t, t, c) - model(x_t, t, null))
# w > 1时增强条件效果
```

---

## 相关知识点

- → [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder) - 变分自编码器
- → [Score Matching](https://arxiv.org/abs/1906.03632) - 分数匹配理论
