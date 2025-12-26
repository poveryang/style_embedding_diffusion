"""
条件潜空间扩散生成器
基于VAE + UNet的latent diffusion架构，支持内容和风格条件注入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """残差块（带时间步和风格条件）"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        style_dim: Optional[int] = None,
        style_condition_type: str = "adain"
    ):
        super().__init__()
        self.style_condition_type = style_condition_type
        self.style_dim = style_dim
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 风格条件
        # 注意：cross_attn 不需要在 ResBlock 中创建任何层，风格通过 CrossAttentionBlock 注入
        if style_dim is not None:
            if style_condition_type == "adain":
                self.style_scale = nn.Linear(style_dim, out_channels)
                self.style_shift = nn.Linear(style_dim, out_channels)
            elif style_condition_type == "film":
                self.style_gamma = nn.Linear(style_dim, out_channels)
                self.style_beta = nn.Linear(style_dim, out_channels)
            # cross_attn: 风格通过 CrossAttentionBlock 注入，ResBlock 中不需要处理
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        style_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = self.shortcut(x)
        
        # 第一个卷积
        x = self.conv1(x)
        x = self.norm1(x)
        
        # 时间步条件
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[:, :, None, None]
        x = x + time_emb
        
        # 风格条件注入
        if style_emb is not None and self.style_dim is not None:
            if self.style_condition_type == "adain":
                # Adaptive Instance Normalization
                scale = self.style_scale(style_emb)[:, :, None, None]
                shift = self.style_shift(style_emb)[:, :, None, None]
                x = x * (1 + scale) + shift
            elif self.style_condition_type == "film":
                # FiLM (Feature-wise Linear Modulation)
                gamma = self.style_gamma(style_emb)[:, :, None, None]
                beta = self.style_beta(style_emb)[:, :, None, None]
                x = x * (1 + gamma) + beta
            # cross_attn: 风格通过 CrossAttentionBlock 注入，这里不需要处理
        
        x = F.silu(x)
        
        # 第二个卷积
        x = self.conv2(x)
        x = self.norm2(x)
        
        return F.silu(x + residual)


class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑为多头注意力
        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        
        # 注意力计算
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        # 重塑回空间维度
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h1=H, w1=W)
        out = self.proj(out)
        
        return out + residual


class CrossAttentionBlock(nn.Module):
    """交叉注意力块：使用风格嵌入作为 key/value，特征图作为 query"""
    def __init__(self, channels: int, style_emb_dim: int, num_heads: int = 8):
        """
        Args:
            channels: 特征图通道数（用于 query 相关的层）
            style_emb_dim: 投影后的风格嵌入维度（用于 key/value 投影，通常是 model_channels）
            num_heads: 注意力头数
        """
        super().__init__()
        self.channels = channels
        self.style_emb_dim = style_emb_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Query 来自特征图 x
        self.norm = nn.GroupNorm(32, channels)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        
        # Key/Value 来自风格嵌入 style_emb（注意：style_emb 已经在 UNet 中被投影到 style_emb_dim 维度）
        self.kv_proj = nn.Linear(style_emb_dim, channels * 2)
        
        # 输出投影
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, channels, H, W) 特征图（作为 query）
            style_emb: (B, channels) 风格嵌入（作为 key/value，已在 UNet 中被投影到 channels 维度）
            
        Returns:
            out: (B, channels, H, W) 交叉注意力输出
        """
        B, C, H, W = x.shape
        residual = x
        
        # Query: 从特征图 x 生成
        x_norm = self.norm(x)
        q = self.q_proj(x_norm)  # (B, C, H, W)
        
        # Key/Value: 从风格嵌入 style_emb 生成
        kv = self.kv_proj(style_emb)  # (B, 2*C)
        k, v = kv.chunk(2, dim=1)  # 每个都是 (B, C)
        
        # 重塑为多头注意力
        # Query: (B, C, H, W) -> (B, num_heads, H*W, head_dim)
        q = rearrange(q, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.num_heads, d=self.head_dim)
        
        # Key/Value: (B, C) -> (B, num_heads, head_dim, 1)
        k = rearrange(k, 'b (h d) -> b h d 1', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b (h d) -> b h d 1', h=self.num_heads, d=self.head_dim)
        
        # 交叉注意力计算: Q @ K -> (B, num_heads, H*W, 1)
        # q: (B, num_heads, H*W, head_dim), k: (B, num_heads, head_dim, 1)
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-2)  # (B, num_heads, H*W, 1)
        
        # 加权求和: attn @ V -> (B, num_heads, H*W, head_dim)
        # attn: (B, num_heads, H*W, 1), v: (B, num_heads, head_dim, 1)
        # 需要将 v 扩展并转置: (B, num_heads, 1, head_dim)
        v_expanded = v.transpose(-2, -1)  # (B, num_heads, 1, head_dim)
        out = attn @ v_expanded  # (B, num_heads, H*W, head_dim)
        
        # 重塑回空间维度: (B, num_heads, H*W, head_dim) -> (B, C, H, W)
        out = rearrange(out, 'b h (h1 w1) d -> b (h d) h1 w1', h1=H, w1=W)
        out = self.proj(out)
        
        return out + residual


class ContentConditionBlock(nn.Module):
    """内容条件注入块（ControlNet风格）"""
    def __init__(self, content_channels: int, latent_channels: int):
        super().__init__()
        self.content_proj = nn.Conv2d(content_channels, latent_channels, 1)
        self.norm = nn.GroupNorm(32, latent_channels)
    
    def forward(self, x: torch.Tensor, content_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, latent_channels, H, W) 当前特征
            content_feat: (B, content_channels, H', W') 内容特征（可能需要上采样）
        """
        # 如果尺寸不匹配，上采样内容特征
        if content_feat.shape[2:] != x.shape[2:]:
            content_feat = F.interpolate(content_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        content_feat = self.content_proj(content_feat)
        content_feat = self.norm(content_feat)
        
        # 相加融合
        return x + content_feat


class ConditionalUNet(nn.Module):
    """
    条件UNet扩散模型
    
    支持两种条件注入：
    1. 内容条件（F_content）：通过ControlNet风格注入
    2. 风格条件（z_style）：通过AdaIN/FiLM全局调制
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (4, 2, 1),
        time_emb_dim: int = 512,
        style_dim: Optional[int] = 64,
        style_condition_type: str = "adain",
        content_condition_type: str = "controlnet",
        content_feature_dims: Tuple[int, ...] = (64, 128, 256, 256)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.style_dim = style_dim
        self.style_condition_type = style_condition_type
        self.content_condition_type = content_condition_type
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 风格嵌入（如果使用cross-attention）
        if style_condition_type == "cross_attn" and style_dim is not None:
            self.style_proj = nn.Linear(style_dim, model_channels)
        
        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 下采样层
        self.down_blocks = nn.ModuleList()
        self.down_content_blocks = nn.ModuleList()
        self.down_content_levels = []  # 记录每个content block对应的level
        input_block_channels = [model_channels]
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                # 内容条件注入
                if content_condition_type == "controlnet" and level < len(content_feature_dims):
                    self.down_content_blocks.append(
                        ContentConditionBlock(content_feature_dims[level], ch)
                    )
                    self.down_content_levels.append(level)
                else:
                    self.down_content_blocks.append(None)
                    self.down_content_levels.append(-1)
                
                layers = [
                    ResBlock(ch, out_ch, time_emb_dim, style_dim, style_condition_type),
                ]
                # 添加注意力层：如果使用 cross_attn，在 attention_resolutions 处添加交叉注意力
                if mult in attention_resolutions:
                    if style_condition_type == "cross_attn" and style_dim is not None:
                        # CrossAttentionBlock 的 kv_proj 需要使用 model_channels（投影后的 style_emb 维度）
                        layers.append(CrossAttentionBlock(out_ch, model_channels))
                    else:
                        layers.append(AttentionBlock(out_ch))
                else:
                    layers.append(nn.Identity())
                self.down_blocks.append(nn.ModuleList(layers))
                ch = out_ch
                input_block_channels.append(ch)
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                # 注意：不保存下采样层的通道数，因为上采样时不需要
        
        # 中间块
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, style_dim, style_condition_type)
        if style_condition_type == "cross_attn" and style_dim is not None:
            # CrossAttentionBlock 的 kv_proj 需要使用 model_channels（投影后的 style_emb 维度）
            self.mid_attn = CrossAttentionBlock(ch, model_channels)
        else:
            self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, style_dim, style_condition_type)
        
        # 上采样层
        self.up_blocks = nn.ModuleList()
        self.up_content_blocks = nn.ModuleList()
        self.up_content_levels = []  # 记录每个content block对应的level
        
        for level, mult in enumerate(reversed(channel_mult)):
            for i in range(num_res_blocks):
                ich = input_block_channels.pop()
                och = model_channels * mult
                
                # 内容条件注入
                content_level = len(content_feature_dims) - 1 - level
                if content_condition_type == "controlnet" and content_level >= 0 and content_level < len(content_feature_dims):
                    self.up_content_blocks.append(
                        ContentConditionBlock(content_feature_dims[content_level], ich + ch)
                    )
                    self.up_content_levels.append(content_level)
                else:
                    self.up_content_blocks.append(None)
                    self.up_content_levels.append(-1)
                
                layers = [
                    ResBlock(ich + ch, och, time_emb_dim, style_dim, style_condition_type),
                ]
                # 添加注意力层：如果使用 cross_attn，在 attention_resolutions 处添加交叉注意力
                if mult in attention_resolutions:
                    if style_condition_type == "cross_attn" and style_dim is not None:
                        # CrossAttentionBlock 的 kv_proj 需要使用 model_channels（投影后的 style_emb 维度）
                        layers.append(CrossAttentionBlock(och, model_channels))
                    else:
                        layers.append(AttentionBlock(och))
                else:
                    layers.append(nn.Identity())
                
                # 上采样：每个 level 的最后一个 block 后上采样，但第一个 level（最底层）不上采样
                # 注意：上采样在 forward 中执行，这里不添加
                
                self.up_blocks.append(nn.ModuleList(layers))
                ch = och
        
        # 输出层
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        content_features: Optional[List[torch.Tensor]] = None,
        style_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, in_channels, H, W) 加噪的latent
            timestep: (B,) 时间步
            content_features: List[torch.Tensor] 多尺度内容特征
            style_emb: (B, style_dim) 风格嵌入
            
        Returns:
            noise_pred: (B, out_channels, H, W) 预测的噪声
        """
        # 时间步嵌入
        time_emb = self.time_embed(timestep)
        
        # 风格嵌入（cross-attention方式）
        if self.style_condition_type == "cross_attn" and style_emb is not None:
            style_emb = self.style_proj(style_emb)
        
        # 输入投影
        x = self.input_conv(x)
        
        # 下采样
        h = []  # 只保存 res_block 的输出，不保存初始输入
        content_idx = 0
        
        for block_list in self.down_blocks:
            if isinstance(block_list[0], nn.Conv2d):
                # 下采样层：不保存输出，只执行下采样
                x = block_list[0](x)
                # 注意：不保存下采样层的输出，因为上采样时不需要
            else:
                # ResBlock + Attention
                # 内容条件注入（在 ResBlock 之前执行）
                if self.down_content_blocks[content_idx] is not None and content_features is not None:
                    level = self.down_content_levels[content_idx]
                    if level >= 0 and level < len(content_features):
                        x = self.down_content_blocks[content_idx](x, content_features[level])
                
                for layer in block_list:
                    if isinstance(layer, ResBlock):
                        x = layer(x, time_emb, style_emb)
                    elif isinstance(layer, CrossAttentionBlock):
                        x = layer(x, style_emb)
                    else:
                        x = layer(x)
                h.append(x)
                content_idx += 1
        
        # 中间块
        x = self.mid_block1(x, time_emb, style_emb)
        if isinstance(self.mid_attn, CrossAttentionBlock):
            x = self.mid_attn(x, style_emb)
        else:
            x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb, style_emb)
        
        # 反转 h 列表，使其与上采样顺序匹配
        h = list(reversed(h))
        
        # 上采样
        content_idx = 0
        for block_idx, block_list in enumerate(self.up_blocks):
            # 每个 level 的第一个 block 之前需要上采样（除了第一个 level）
            # block_idx % num_res_blocks == 0 表示是某个 level 的第一个 block
            if block_idx > 0 and block_idx % self.num_res_blocks == 0:
                # 上采样到下一个 level 的尺寸
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            
            # 从 h 中 pop 特征图（现在 h 已经反转，从前往后 pop）
            feat = h.pop(0)
            x = torch.cat([x, feat], dim=1)
            
            # 内容条件注入（在 ResBlock 之前执行）
            if self.up_content_blocks[content_idx] is not None and content_features is not None:
                level = self.up_content_levels[content_idx]
                if level >= 0 and level < len(content_features):
                    x = self.up_content_blocks[content_idx](x, content_features[level])
            
            for layer in block_list:
                if isinstance(layer, ResBlock):
                    x = layer(x, time_emb, style_emb)
                elif isinstance(layer, CrossAttentionBlock):
                    x = layer(x, style_emb)
                elif isinstance(layer, (AttentionBlock, nn.Identity)):
                    x = layer(x)
            
            content_idx += 1
        
        # 输出
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        
        return x

