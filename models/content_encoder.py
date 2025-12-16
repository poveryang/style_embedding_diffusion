"""
内容编码器（Enc_content）
编码二维码的空间结构与模块布局，为扩散模型提供内容引导
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ContentEncoder(nn.Module):
    """
    内容编码器：提取二维码的结构与布局信息
    
    输入: I_bin (B, 1, H, W) - 二值二维码图像
    输出: 多尺度特征图 F_content - 用于条件注入
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 64,
        num_layers: int = 4,
        output_dims: List[int] = None
    ):
        super().__init__()
        if output_dims is None:
            output_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 4]
        
        self.layers = nn.ModuleList()
        self.output_dims = output_dims
        
        # 初始卷积
        self.initial_conv = nn.Conv2d(in_channels, base_dim, 7, padding=3)
        
        # 多尺度特征提取
        current_dim = base_dim
        for i in range(num_layers):
            # 下采样
            self.layers.append(nn.Sequential(
                ResBlock(current_dim, output_dims[i]),
                nn.MaxPool2d(2) if i < num_layers - 1 else nn.Identity()
            ))
            current_dim = output_dims[i]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (B, 1, H, W) 二值二维码图像
            
        Returns:
            features: List[torch.Tensor] 多尺度特征图列表
        """
        x = F.relu(self.initial_conv(x))
        
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        return features

