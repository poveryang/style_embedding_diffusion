"""
VAE模块：图像与latent空间之间的编码解码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = F.silu(out + residual)
        return out


class Encoder(nn.Module):
    """VAE编码器"""
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 128,
        scale_factor: int = 8
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        num_down = int(scale_factor).bit_length() - 1  # 计算下采样层数
        
        for i in range(num_down):
            out_channels = current_channels * 2 if i < num_down - 1 else current_channels
            self.down_blocks.append(nn.Sequential(
                ResidualBlock(current_channels, out_channels),
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            ))
            current_channels = out_channels
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, current_channels)
        self.conv_out = nn.Conv2d(current_channels, latent_channels * 2, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码
        
        Args:
            x: (B, C, H, W) 输入图像
            
        Returns:
            mu: (B, latent_channels, H', W') 均值
            logvar: (B, latent_channels, H', W') 对数方差
        """
        x = self.conv_in(x)
        
        for block in self.down_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        # 分离均值和方差
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """VAE解码器"""
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 128,
        scale_factor: int = 8
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 初始卷积
        self.conv_in = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        current_channels = base_channels
        num_up = int(scale_factor).bit_length() - 1
        
        for i in range(num_up):
            out_channels_block = current_channels // 2 if i < num_up - 1 else current_channels
            self.up_blocks.append(nn.Sequential(
                ResidualBlock(current_channels, current_channels),
                nn.ConvTranspose2d(current_channels, out_channels_block, 4, stride=2, padding=1)
            ))
            current_channels = out_channels_block
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, current_channels)
        self.conv_out = nn.Conv2d(current_channels, out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码
        
        Args:
            z: (B, latent_channels, H', W') latent表示
            
        Returns:
            x: (B, C, H, W) 重建图像
        """
        x = self.conv_in(z)
        
        for block in self.up_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return torch.tanh(x)  # 归一化到[-1, 1]


class VAE(nn.Module):
    """
    VAE模块：用于图像与latent空间之间的转换
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 128,
        scale_factor: int = 8,
        kl_weight: float = 1e-6
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, base_channels, scale_factor)
        self.decoder = Decoder(latent_channels, out_channels, base_channels, scale_factor)
        self.kl_weight = kl_weight
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码图像到latent空间
        
        Args:
            x: (B, C, H, W) 输入图像
            
        Returns:
            z: (B, latent_channels, H', W') latent表示
        """
        mu, logvar = self.encoder(x)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码latent到图像空间
        
        Args:
            z: (B, latent_channels, H', W') latent表示
            
        Returns:
            x: (B, C, H, W) 重建图像
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (B, C, H, W) 输入图像
            
        Returns:
            x_recon: (B, C, H, W) 重建图像
            z: (B, latent_channels, H', W') latent表示
            kl_loss: (scalar) KL散度损失
        """
        mu, logvar = self.encoder(x)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        x_recon = self.decode(z)
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        kl_loss = kl_loss.mean() * self.kl_weight
        
        return x_recon, z, kl_loss

