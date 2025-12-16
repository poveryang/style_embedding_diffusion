"""
配置文件：定义模型超参数和训练设置
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # 内容编码器
    content_encoder_dim: int = 256
    content_encoder_layers: int = 4
    
    # 风格编码器
    style_embedding_dim: int = 64  # z_style维度
    style_encoder_base_dim: int = 64
    style_encoder_layers: int = 4
    style_kl_weight: float = 0.01  # KL正则权重
    
    # VAE
    vae_latent_channels: int = 4
    vae_in_channels: int = 1  # 灰度图
    vae_out_channels: int = 1
    vae_scale_factor: int = 8  # 下采样倍数
    
    # UNet扩散模型
    unet_in_channels: int = 4  # latent channels
    unet_out_channels: int = 4
    unet_model_channels: int = 320
    unet_attention_resolutions: tuple = (4, 2, 1)
    unet_channel_mult: tuple = (1, 2, 4, 4)
    unet_num_res_blocks: int = 2
    unet_dropout: float = 0.0
    
    # 条件注入
    content_condition_type: str = "controlnet"  # "controlnet" or "concat"
    style_condition_type: str = "adain"  # "adain", "film", or "cross_attn"
    
    # 扩散过程
    num_timesteps: int = 1000
    beta_schedule: str = "linear"  # "linear", "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32  # 总batch_size（所有GPU的总和），DDP时会自动分配到各GPU
                          # 例如：batch_size=32, 4张GPU → 每张GPU处理8个样本，总有效batch_size=32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    save_interval: int = 10
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 数据
    image_size: int = 256
    data_root: str = "data"
    
    # 优化器
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # 混合精度
    use_amp: bool = True
    
    # 风格一致性约束（可选）
    style_consistency_weight: float = 0.1
    
    # 图像保存
    save_samples: bool = True  # 是否保存样本图像
    sample_interval: int = 1  # 每N个epoch保存一次样本（1表示每个epoch都保存）
    num_sample_images: int = 4  # 每次保存的样本数量


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 42

