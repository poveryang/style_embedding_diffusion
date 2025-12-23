"""
扩散调度器：管理噪声调度和采样过程
"""
import torch
import numpy as np
from typing import Optional


class DiffusionScheduler:
    """
    扩散调度器
    
    支持线性或余弦噪声调度
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.device = device
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == "cosine":
            # 余弦调度
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # 用于采样的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于去噪的系数
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        向输入添加噪声
        
        Args:
            x0: (B, C, H, W) 原始数据
            t: (B,) 时间步
            noise: (B, C, H, W) 可选的自定义噪声
            
        Returns:
            xt: (B, C, H, W) 加噪后的数据
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # 确保scheduler的参数与输入数据在同一设备上（支持DataParallel）
        device = x0.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt
    
    def sample_timesteps(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        随机采样时间步
        
        Args:
            batch_size: 批次大小
            device: 目标设备（如果为None，使用self.device）
            
        Returns:
            t: (batch_size,) 随机时间步
        """
        target_device = device if device is not None else self.device
        return torch.randint(0, self.num_timesteps, (batch_size,), device=target_device)
    
    def predict_x0_from_noise(self, noise_pred: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        从预测的噪声和加噪的latent预测原始x0
        
        Args:
            noise_pred: (B, C, H, W) 模型预测的噪声
            t: (B,) 当前时间步
            x_t: (B, C, H, W) 加噪后的latent
            
        Returns:
            pred_x0: (B, C, H, W) 预测的原始latent
        """
        device = x_t.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        # 标准公式: z_0 = (z_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        # 等价形式: z_0 = (1/sqrt(alpha_t)) * z_t - (sqrt(1-alpha_t)/sqrt(alpha_t)) * eps
        # 其中: sqrt(1/alpha_t - 1) = sqrt((1-alpha_t)/alpha_t) = sqrt(1-alpha_t)/sqrt(alpha_t)
        sqrt_recip_alphas_cumprod = 1.0 / torch.sqrt(alphas_cumprod)  # = 1/sqrt(alpha_t)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)  # = sqrt(1-alpha_t)/sqrt(alpha_t)
        
        pred_x0 = (sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t -
                   sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * noise_pred)
        
        return pred_x0
    
    def step(self, model_output: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        DDPM采样步骤
        
        Args:
            model_output: (B, C, H, W) 模型预测的噪声
            t: (B,) 当前时间步
            x: (B, C, H, W) 当前加噪的latent
            
        Returns:
            prev_x: (B, C, H, W) 去噪后的latent
        """
        # 确保scheduler的参数与输入数据在同一设备上（支持DataParallel）
        device = x.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        posterior_variance = self.posterior_variance.to(device)
        
        # 预测x0
        pred_x0 = self.predict_x0_from_noise(model_output, t, x)
        
        # 计算后验均值
        posterior_mean = (
            posterior_mean_coef1[t].view(-1, 1, 1, 1) * pred_x0 +
            posterior_mean_coef2[t].view(-1, 1, 1, 1) * x
        )
        
        # 计算后验方差
        posterior_var = posterior_variance[t].view(-1, 1, 1, 1)
        
        # 采样
        if t[0] == 0:
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_var) * noise
    
    def ddim_step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        eta: float = 0.0,
        prev_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DDIM采样步骤（确定性采样，更快）
        
        Args:
            model_output: (B, C, H, W) 模型预测的噪声
            t: (B,) 当前时间步
            x: (B, C, H, W) 当前加噪的latent
            eta: DDIM参数，0为完全确定性
            prev_t: (B,) 前一个时间步
            
        Returns:
            prev_x: (B, C, H, W) 去噪后的latent
        """
        if prev_t is None:
            prev_t = t - 1
        
        # 确保scheduler的参数与输入数据在同一设备上（支持DataParallel）
        device = x.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        posterior_variance = self.posterior_variance.to(device)
        
        # 预测x0
        sqrt_recip_alphas_cumprod = 1.0 / torch.sqrt(alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        
        pred_x0 = (sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x -
                   sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * model_output)
        
        # 计算方向指向xt
        dir_xt = torch.sqrt(1.0 - alphas_cumprod[prev_t].view(-1, 1, 1, 1) - 
                           eta ** 2 * posterior_variance[t].view(-1, 1, 1, 1)) * model_output
        
        # 随机噪声
        noise = eta * torch.sqrt(posterior_variance[t].view(-1, 1, 1, 1)) * torch.randn_like(x)
        
        prev_x = torch.sqrt(alphas_cumprod[prev_t].view(-1, 1, 1, 1)) * pred_x0 + dir_xt + noise
        
        return prev_x

