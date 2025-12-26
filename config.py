"""
配置文件：定义模型超参数和训练设置
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any


@dataclass
class ModelConfig:
    """模型配置"""
    # 内容编码器
    content_encoder_dim: int = 256
    content_encoder_layers: int = 4
    
    # 风格编码器
    style_embedding_dim: int = 512  # z_style维度（从256增加到512，提供更大的表征空间）
    style_encoder_base_dim: int = 128
    style_encoder_layers: int = 4
    style_kl_weight: float = 1e-4  # KL正则权重（从1e-2降到1e-4，降低约束强度，允许学习更丰富的风格表征）
    
    # VAE
    vae_latent_channels: int = 4
    vae_in_channels: int = 1  # 灰度图
    vae_out_channels: int = 1
    vae_scale_factor: int = 4  # 下采样倍数
    vae_recon_weight: float = 1.0  # VAE重建损失权重
    vae_kl_weight: float = 5e-6  # VAE KL散度损失权重
    normalize_vae_latent: bool = True  # 是否标准化 VAE latent
    log_latent_stats: bool = True  # 是否记录 latent 统计信息（用于诊断）
    
    # UNet扩散模型
    unet_in_channels: int = 4  # latent channels
    unet_out_channels: int = 4
    unet_model_channels: int = 320
    unet_attention_resolutions: tuple = (4, 2, 1)
    unet_channel_mult: tuple = (1, 2, 4, 4)
    unet_num_res_blocks: int = 2
    unet_dropout: float = 0.0
    diffusion_recon_weight: float = 1.0  # 扩散模型图像重建损失权重（0表示禁用）
    
    # 条件注入
    content_condition_type: str = "controlnet"  # "controlnet" or "concat"
    style_condition_type: str = "cross_attn"  # "adain", "film", or "cross_attn" (交叉注意力)
    
    # 扩散过程
    num_timesteps: int = 1000
    beta_schedule: str = "linear"  # "linear", "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    data_root: str = "data"  # 数据根目录
    batch_size: int = 24  # 总batch_size（所有GPU的总和），DDP时会自动分配到各GPU
    image_size: int = 256  # 图像大小

    # 预训练模型配置
    pretrained_model_path: Optional[str] = None  # 预训练模型路径
    pretrained_modules: Optional[List[str]] = field(default=None)  # 选择性加载模块：如果为None，则加载所有匹配的模块
    pretrained_strict: bool = False  # 是否严格匹配：如果为True，所有键必须匹配；如果为False，允许部分匹配

    # 训练轮次与策略
    num_epochs: int = 300
    lr_scheduler: str = "cosine"  # 学习率调度策略: "constant", "cosine", "linear"
    learning_rate: float = 5e-5
    min_lr: float = 1e-6  # 最小学习率（用于cosine/linear decay）
    training_stages: Optional[List[Tuple[int, int, Dict[str, Any]]]] = field(default_factory=lambda: [
        (0, 100, {
            'vae': True, 
            'content_encoder': True, 
            'style_encoder': True, 
            'unet': True,
            'learning_rate': 1e-4,      # 阶段1使用更高的学习率
            'lr_scheduler': 'cosine',
            'min_lr': 1e-6,
            'warmup_epochs': 20,
        }),
        (100, 150, {
            'vae': True, 
            'content_encoder': False, 
            'style_encoder': False, 
            'unet': False,
            'learning_rate': 8e-5,      # 阶段2使用中等学习率
            'lr_scheduler': 'cosine',
            'min_lr': 1e-6,
            'warmup_epochs': 5,
        }),
        (150, 250, {
            'vae': False, 
            'content_encoder': True, 
            'style_encoder': True, 
            'unet': True,
            'learning_rate': 6e-5,      # 阶段3使用较低学习率进行微调
            'lr_scheduler': 'cosine',
            'min_lr': 1e-6,
            'warmup_epochs': 5,
        }),
        (250, 300, {
            'vae': False, 
            'content_encoder': False, 
            'style_encoder': True, 
            'unet': False,
            'learning_rate': 4e-5,      # 阶段4使用最低学习率进行微调
            'lr_scheduler': 'cosine',
            'min_lr': 1e-6,
            'warmup_epochs': 5,
        }),
    ])
        
    # 优化器
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    warmup_steps: Optional[int] = None  # Warmup步数（如果为None，则使用warmup_epochs）
    warmup_epochs: int = 5  # Warmup epoch数（如果warmup_steps为None则使用此值）
    module_learning_rates: Optional[Dict[str, float]] = field(default=None)
    
    # 保存和日志
    save_interval: int = 10  # 每 N 个 epoch 保存一次检查点（模型权重、优化器状态等）
    log_interval: int = 100  # 每 N 个 batch 记录一次训练指标到 wandb（损失、学习率等）
    checkpoint_dir: str = "checkpoints"  # 检查点目录，如果未通过命令行指定，会自动添加时间戳避免覆盖
    log_dir: str = "logs"  # wandb本地文件目录
    
    # 混合精度
    use_amp: bool = True
    
    # 图像保存
    save_samples: bool = True  # 是否保存样本图像
    sample_interval: int = 1  # 每N个epoch保存一次样本（1表示每个epoch都保存）
    num_sample_images: int = 4  # 每次保存的样本数量
    
    # wandb模型保存
    save_model_to_wandb: bool = False  # 是否保存模型到 wandb
    save_best_model_to_wandb: bool = False  # 是否保存最佳模型到 wandb
    save_checkpoints_to_wandb: bool = False  # 是否保存定期检查点到 wandb（默认False，因为文件较大）


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 66

