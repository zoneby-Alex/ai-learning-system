# 项目：DDPM Diffusion模型复现

## 基本信息

| 属性 | 内容 |
|------|------|
| 难度 | ⭐⭐⭐⭐☆ |
| 预估时间 | 2-3周 |
| 目标就业 | 算法工程师 |
| 技术栈 | PyTorch + UNet |

---

## 项目目标

1. 理解Diffusion前向加噪和反向去噪原理
2. 实现DDPM训练和采样
3. 实现DDIM加速采样
4. 在CelebA/MNIST上训练并可视化生成过程

---

## 核心实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDPM:
    def __init__(self, model, n_steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.model = model  # UNet噪声预测网络
        self.n_steps = n_steps
        self.device = device

        # 前向过程的参数
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x_0, t, noise=None):
        """前向加噪：x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def train_step(self, x_0):
        """单步训练：随机t，预测噪声"""
        batch_size = x_0.size(0)
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        x_t, noise = self.forward_process(x_0, t)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, shape, save_intermediate=False):
        """DDPM采样：x_t → x_{t-1} → ... → x_0"""
        x_t = torch.randn(shape, device=self.device)
        intermediates = []

        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x_t, t_batch)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            x_t = (1 / alpha_t.sqrt()) * (x_t - (1-alpha_t)/(1-alpha_cumprod_t).sqrt() * noise_pred) + beta_t.sqrt() * noise

            if save_intermediate and t % 100 == 0:
                intermediates.append(x_t.cpu())

        return x_t, intermediates

    @torch.no_grad()
    def ddim_sample(self, shape, n_steps=50, eta=0.0):
        """DDIM加速采样"""
        x_t = torch.randn(shape, device=self.device)
        step_size = self.n_steps // n_steps
        timesteps = list(reversed(range(0, self.n_steps, step_size)))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x_t, t_batch)

            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)

            # 预测x_0
            pred_x0 = (x_t - (1-alpha_cumprod_t).sqrt() * noise_pred) / alpha_cumprod_t.sqrt()

            # DDIM更新
            sigma_t = eta * ((1-alpha_cumprod_t_prev)/(1-alpha_cumprod_t)*(1-alpha_cumprod_t/alpha_cumprod_t_prev)).sqrt()
            direction = (1-alpha_cumprod_t_prev - sigma_t**2).sqrt() * noise_pred
            x_t = alpha_cumprod_t_prev.sqrt() * pred_x0 + direction

            if eta > 0:
                x_t = x_t + sigma_t * torch.randn_like(x_t)

        return x_t

# UNet噪声预测网络（简化版）
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # 编码器
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        # 中间
        self.mid = self._block(base_channels * 2, base_channels * 2)
        # 解码器
        self.dec2 = self._block(base_channels * 4, base_channels)
        self.dec1 = nn.Conv2d(base_channels * 2, in_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        m = self.mid(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(m, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        return d1
```

---

## 验收标准

- [ ] DDPM能在MNIST/CelebA上生成合理图像
- [ ] DDIM采样能在50步内生成
- [ ] FID评估
- [ ] 可视化去噪过程（从纯噪声逐步生成图像）

## 面试Q&A

### Q: Diffusion和GAN的核心区别？
- GAN：对抗训练，不稳定但生成快（1步）
- Diffusion：逐步去噪，稳定但生成慢（多步），DDIM缓解了速度问题

### Q: 为什么去噪过程的每一步都要加噪声？
DDPM的反向过程是随机微分方程（SDE），随机性保证采样多样性。DDIM通过η=0可关闭随机性实现确定性生成。
