"""
成像风格编码器（Enc_style）【核心模块】
学习连续、抽象、可插值的成像风格表征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StyleEncoder(nn.Module):
    """
    成像风格编码器：提取成像风格表征
    
    输入: I_orig (B, 1, H, W) - 真实成像二维码图像
    输出: z_style (B, style_dim) - 低维风格向量
    
    关键约束：
    - 向量维度受限，防止编码内容信息
    - 可加入KL正则或方差约束
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 64,
        num_layers: int = 4,
        style_dim: int = 64,
        use_vae: bool = True,  # 是否使用VAE结构（KL正则）
        kl_weight: float = 0.01
    ):
        super().__init__()
        self.style_dim = style_dim
        self.use_vae = use_vae
        self.kl_weight = kl_weight
        
        # 特征提取网络
        layers = []
        current_dim = base_dim
        
        # 初始卷积
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 7, padding=3),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        ))
        
        # 下采样层
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(current_dim, current_dim * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(current_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_dim * 2, current_dim * 2, 3, padding=1),
                nn.BatchNorm2d(current_dim * 2),
                nn.ReLU(inplace=True)
            ))
            current_dim = current_dim * 2
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 风格向量生成
        if use_vae:
            # VAE结构：输出均值和方差
            self.fc_mu = nn.Linear(current_dim, style_dim)
            self.fc_logvar = nn.Linear(current_dim, style_dim)
        else:
            # 直接映射
            self.fc_style = nn.Sequential(
                nn.Linear(current_dim, style_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(style_dim * 2, style_dim)
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: (B, 1, H, W) 真实成像图像
            
        Returns:
            z_style: (B, style_dim) 风格向量
            kl_loss: (scalar) KL散度损失（如果use_vae=True）
        """
        # 特征提取
        features = self.feature_extractor(x)  # (B, C, H', W')
        features = self.global_pool(features)  # (B, C, 1, 1)
        features = features.view(features.size(0), -1)  # (B, C)
        
        if self.use_vae:
            # VAE结构
            mu = self.fc_mu(features)
            logvar = self.fc_logvar(features)
            
            # 重参数化
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z_style = mu + eps * std
            else:
                z_style = mu
            
            # KL散度损失
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_loss.mean() * self.kl_weight
            
            # 存储统计信息（用于诊断）
            if self.training:
                self._last_mu_stats = {
                    'mu_mean': mu.mean().item(),
                    'mu_std': mu.std().item(),
                    'mu_min': mu.min().item(),
                    'mu_max': mu.max().item(),
                }
                self._last_logvar_stats = {
                    'logvar_mean': logvar.mean().item(),
                    'logvar_std': logvar.std().item(),
                    'logvar_min': logvar.min().item(),
                    'logvar_max': logvar.max().item(),
                }
                self._last_z_style_stats = {
                    'z_style_mean': z_style.mean().item(),
                    'z_style_std': z_style.std().item(),
                    'z_style_min': z_style.min().item(),
                    'z_style_max': z_style.max().item(),
                }
                # 计算原始KL散度（未加权）
                raw_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                self._last_raw_kl = raw_kl.item()
            
            return z_style, kl_loss
        else:
            # 直接映射
            z_style = self.fc_style(features)
            return z_style, None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码接口（推理时使用）
        
        Args:
            x: (B, 1, H, W) 风格参考图像
            
        Returns:
            z_style: (B, style_dim) 风格向量
        """
        self.eval()
        with torch.no_grad():
            z_style, _ = self.forward(x)
        return z_style

