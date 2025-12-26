"""
训练脚本：训练基于条件扩散模型的成像风格表征学习系统
支持DistributedDataParallel (DDP)多GPU训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import datetime
from typing import Dict, List, Optional, Any

from config import Config
from models import ContentEncoder, StyleEncoder, VAE, ConditionalUNet
from utils import DiffusionScheduler, QRCodeDataset
from utils.wandb_logger import WandbLogger


def save_sample_images(
    model,  # StyleEmbeddingDiffusionModel (定义在后面)
    val_loader: DataLoader,
    epoch: int,
    output_dir: str,
    num_samples: int = 4,
    num_inference_steps: int = 50,
    device: str = "cuda"
):
    """
    生成并保存样本图像
    
    保存的图像包含4部分（水平拼接）：
    1. I_bin: 输入的二值二维码图像
    2. I_orig: 原始真实图像
    3. x_recon: VAE重建图像
    4. I_gen: 扩散模型生成的图像
    
    Args:
        model: 训练中的模型
        val_loader: 验证数据加载器
        epoch: 当前epoch
        output_dir: 输出目录
        num_samples: 生成的样本数量
        num_inference_steps: 推理步数
        device: 设备
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 从验证集中获取样本
    val_iter = iter(val_loader)
    I_bin_samples = []
    I_orig_samples = []
    
    for _ in range(min(num_samples, len(val_loader))):
        try:
            I_bin, I_orig = next(val_iter)
            I_bin_samples.append(I_bin[0:1])  # 只取第一个样本
            I_orig_samples.append(I_orig[0:1])
        except StopIteration:
            break
    
    if len(I_bin_samples) == 0:
        return
    
    with torch.no_grad():
        for idx, (I_bin, I_orig) in enumerate(zip(I_bin_samples, I_orig_samples)):
            I_bin = I_bin.to(device)
            I_orig = I_orig.to(device)
            
            # 获取VAE重建图像
            # 处理DDP包装的模型
            if isinstance(model, DDP):
                actual_model = model.module
            else:
                actual_model = model
            
            x_recon, _, _ = actual_model.vae(I_orig)
            
            # 生成图像（扩散模型）
            I_gen = generate_image(model, I_bin, I_orig, num_inference_steps)
            
            # 转换为numpy并保存
            # 图像范围是[-1, 1]，需要转换到[0, 255]
            # 检查并处理 NaN 值（如果出现 NaN，使用默认值或跳过）
            def safe_convert_to_uint8(img_tensor, default_value=128):
                """安全地将图像 tensor 转换为 uint8，处理 NaN 值"""
                img_np = img_tensor.cpu().numpy()
                if np.isnan(img_np).any() or np.isinf(img_np).any():
                    # 如果包含 NaN 或 Inf，使用默认值（灰色）
                    img_np = np.full_like(img_np, default_value / 255.0 * 2 - 1, dtype=img_np.dtype)
                img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                return img_np
            
            I_bin_np = safe_convert_to_uint8(I_bin[0, 0])
            I_orig_np = safe_convert_to_uint8(I_orig[0, 0])
            x_recon_np = safe_convert_to_uint8(x_recon[0, 0])
            I_gen_np = safe_convert_to_uint8(I_gen[0, 0])
            
            # 创建对比图像（水平拼接）：输入 | 原始 | VAE重建 | 扩散生成
            comparison = np.hstack([I_bin_np, I_orig_np, x_recon_np, I_gen_np])
            
            # 保存图像，按sample组织：sample_{idx:02d}/epoch_{epoch:03d}.png
            # 这样便于按sample查看不同epoch的进度
            sample_dir = os.path.join(output_dir, f"sample_{idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            img_path = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
            Image.fromarray(comparison).save(img_path)
    
    model.train()


def generate_image(
    model,  # StyleEmbeddingDiffusionModel (定义在后面) 或 DDP包装的模型
    I_bin: torch.Tensor,
    I_style_ref: torch.Tensor,
    num_inference_steps: int = 50,
    use_ddim: bool = True
) -> torch.Tensor:
    """
    生成图像（简化版，用于训练过程中的可视化）
    
    Args:
        model: 模型（可能是DDP包装的）
        I_bin: (1, 1, H, W) 二值二维码图像
        I_style_ref: (1, 1, H, W) 风格参考图像
        num_inference_steps: 推理步数
        use_ddim: 是否使用DDIM
        
    Returns:
        I_gen: (1, 1, H, W) 生成的图像
    """
    # 处理DDP包装的模型：通过model.module访问原始模型
    if isinstance(model, DDP):
        actual_model = model.module
    else:
        actual_model = model
    
    # 1. 提取风格表征
    z_style = actual_model.style_encoder.encode(I_style_ref)
    
    # 2. 编码内容结构
    F_content = actual_model.content_encoder(I_bin)
    
    # 3. 从纯噪声开始条件扩散采样
    z_T = torch.randn(
        (1, actual_model.config.model.vae_latent_channels,
         I_bin.shape[2] // actual_model.config.model.vae_scale_factor,
         I_bin.shape[3] // actual_model.config.model.vae_scale_factor),
        device=I_bin.device
    )
    
    # 时间步序列
    timesteps = torch.linspace(
        actual_model.scheduler.num_timesteps - 1, 0, num_inference_steps,
        device=I_bin.device
    ).long()
    
    z_t = z_T
    for i, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0)
        
        # 预测噪声（推理时不需要DDP同步，使用原始模型）
        eps_pred = actual_model.unet(z_t, t_batch, content_features=F_content, style_emb=z_style)
        
        # 去噪步骤
        if use_ddim:
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=I_bin.device)
            z_t = actual_model.scheduler.ddim_step(eps_pred, t_batch, z_t, eta=0.0, prev_t=prev_t.unsqueeze(0))
        else:
            # DDPM 也需要 prev_t 参数以支持跳步采样
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=I_bin.device)
            z_t = actual_model.scheduler.step(eps_pred, t_batch, z_t, prev_t=prev_t.unsqueeze(0))
    
    # 4. 解码生成图像
    I_gen = actual_model.vae.decode(z_t)
    
    return I_gen


class StyleEmbeddingDiffusionModel(nn.Module):
    """
    完整的风格嵌入扩散模型系统
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        
        # 初始化各模块
        self.content_encoder = ContentEncoder(
            in_channels=1,
            base_dim=model_cfg.content_encoder_dim,
            num_layers=model_cfg.content_encoder_layers
        )
        
        self.style_encoder = StyleEncoder(
            in_channels=1,
            base_dim=model_cfg.style_encoder_base_dim,
            num_layers=model_cfg.style_encoder_layers,
            style_dim=model_cfg.style_embedding_dim,
            use_vae=True,
            kl_weight=model_cfg.style_kl_weight
        )
        
        self.vae = VAE(
            in_channels=model_cfg.vae_in_channels,
            out_channels=model_cfg.vae_out_channels,
            latent_channels=model_cfg.vae_latent_channels,
            scale_factor=model_cfg.vae_scale_factor,
            kl_weight=model_cfg.vae_kl_weight
        )
        
        # 获取内容编码器的输出维度
        content_feature_dims = [
            model_cfg.content_encoder_dim,
            model_cfg.content_encoder_dim * 2,
            model_cfg.content_encoder_dim * 4,
            model_cfg.content_encoder_dim * 4
        ]
        
        self.unet = ConditionalUNet(
            in_channels=model_cfg.unet_in_channels,
            out_channels=model_cfg.unet_out_channels,
            model_channels=model_cfg.unet_model_channels,
            channel_mult=model_cfg.unet_channel_mult,
            num_res_blocks=model_cfg.unet_num_res_blocks,
            attention_resolutions=model_cfg.unet_attention_resolutions,
            style_dim=model_cfg.style_embedding_dim,
            style_condition_type=model_cfg.style_condition_type,
            content_condition_type=model_cfg.content_condition_type,
            content_feature_dims=content_feature_dims
        )
        
        # 扩散调度器
        self.scheduler = DiffusionScheduler(
            num_timesteps=model_cfg.num_timesteps,
            beta_schedule=model_cfg.beta_schedule,
            beta_start=model_cfg.beta_start,
            beta_end=model_cfg.beta_end,
            device=config.device
        )
    
    def forward(self, I_bin: torch.Tensor, I_orig: torch.Tensor) -> dict:
        """
        前向传播（训练阶段）
        
        Args:
            I_bin: (B, 1, H, W) 二值二维码图像
            I_orig: (B, 1, H, W) 真实成像图像
            
        Returns:
            losses: 损失字典
        """
        # 1. VAE编码-解码：获取重建图像、latent和KL损失
        # 使用forward方法以获取重建损失和KL损失，确保解码器被训练
        x_recon, z0, vae_kl_loss = self.vae(I_orig)
        
        # 修复：VAE latent 尺度标准化（可选）
        # 扩散调度器假设输入 ~N(0,1)，但 VAE 输出的 latent 可能不满足这个假设
        # 如果 KL weight 很小（如 1e-6），VAE 可能学习到方差较大的分布
        if self.config.model.normalize_vae_latent:
            # 按 batch 标准化：z0 = (z0 - mean) / std
            z0_mean = z0.mean(dim=[1, 2, 3], keepdim=True)
            z0_std = z0.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
            z0 = (z0 - z0_mean) / z0_std
        
        # 计算VAE重建损失（L1损失，对图像重建更稳定）
        vae_recon_loss = nn.functional.l1_loss(x_recon, I_orig)
        
        # 2. 随机采样扩散步并加噪
        batch_size = I_bin.size(0)
        # 确保时间步采样在与数据相同的设备上（支持DataParallel）
        device = I_bin.device
        t = self.scheduler.sample_timesteps(batch_size).to(device)
        noise = torch.randn_like(z0)
        z_t = self.scheduler.add_noise(z0, t, noise)
        
        # 诊断：记录 z0 和 z_t 的统计信息（用于监控 latent 分布）
        # 这些信息可以帮助确认 latent 是否接近 N(0,1) 分布
        if self.config.model.log_latent_stats:
            z0_stats = {
                'z0_mean': z0.mean().item(),
                'z0_std': z0.std().item(),
                'z0_min': z0.min().item(),
                'z0_max': z0.max().item(),
            }
            z_t_stats = {
                'z_t_mean': z_t.mean().item(),
                'z_t_std': z_t.std().item(),
                'z_t_min': z_t.min().item(),
                'z_t_max': z_t.max().item(),
            }
            # 存储统计信息，在训练循环中记录到 wandb
            self._last_z0_stats = z0_stats
            self._last_z_t_stats = z_t_stats
        
        # 3. 提取条件信息
        F_content = self.content_encoder(I_bin)
        z_style, style_kl_loss = self.style_encoder(I_orig)
        
        # 诊断：记录风格嵌入的统计信息（用于监控风格分布）
        if self.config.model.log_latent_stats and self.training:
            if hasattr(self.style_encoder, '_last_mu_stats'):
                self._last_style_stats = {
                    **self.style_encoder._last_mu_stats,
                    **self.style_encoder._last_logvar_stats,
                    **self.style_encoder._last_z_style_stats,
                    'raw_kl': self.style_encoder._last_raw_kl,
                }
        
        # 4. 条件扩散预测噪声
        eps_pred = self.unet(z_t, t, content_features=F_content, style_emb=z_style)
        
        # 5. 计算扩散损失（噪声预测损失）
        diffusion_loss = nn.functional.mse_loss(eps_pred, noise)
        
        # 6. 可选的扩散模型图像重建损失
        diffusion_recon_loss = torch.tensor(0.0, device=device)
        if self.config.model.diffusion_recon_weight > 0:
            # 从预测的噪声推理出z0
            z0_pred = self.scheduler.predict_x0_from_noise(eps_pred, t, z_t)
            # 解码生成图像
            I_recon_diff = self.vae.decode(z0_pred)
            # 计算重建损失
            diffusion_recon_loss = nn.functional.l1_loss(I_recon_diff, I_orig)
        
        # 总损失：扩散损失 + VAE重建损失 + VAE KL损失 + 风格KL损失 + 扩散重建损失
        total_loss = diffusion_loss
        total_loss = total_loss + self.config.model.vae_recon_weight * vae_recon_loss
        total_loss = total_loss + vae_kl_loss
        if style_kl_loss is not None:
            total_loss = total_loss + style_kl_loss
        total_loss = total_loss + self.config.model.diffusion_recon_weight * diffusion_recon_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'vae_recon_loss': vae_recon_loss,
            'vae_kl_loss': vae_kl_loss,
            'style_kl_loss': style_kl_loss if style_kl_loss is not None else torch.tensor(0.0),
            'diffusion_recon_loss': diffusion_recon_loss,
        }


def load_pretrained_model(
    model: StyleEmbeddingDiffusionModel,
    pretrained_path: str,
    device: str,
    modules_to_load: Optional[List[str]] = None,
    strict: bool = False,
    is_main_process: bool = True
) -> Dict[str, bool]:
    """
    加载预训练模型权重
    
    Args:
        model: 模型实例（可能是DDP包装的）
        pretrained_path: 预训练模型路径
        device: 设备
        modules_to_load: 要加载的模块列表，例如 ['vae', 'content_encoder']
                        如果为None，则加载所有匹配的模块
        strict: 是否严格匹配所有键
        is_main_process: 是否为主进程
        
    Returns:
        loaded_modules: 字典，包含各模块的加载状态
    """
    if not os.path.exists(pretrained_path):
        if is_main_process:
            print(f"❌ 错误: 预训练模型文件不存在: {pretrained_path}")
        return {}
    
    # 获取实际模型（处理DDP包装）
    if isinstance(model, DDP):
        actual_model = model.module
    else:
        actual_model = model
    
    # 加载checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 提取模型状态字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
    else:
        # 如果checkpoint本身就是state_dict
        pretrained_state_dict = checkpoint
    
    # 处理DDP前缀：移除module前缀（如果存在）
    if any(k.startswith('module.') for k in pretrained_state_dict.keys()):
        pretrained_state_dict = {k.replace('module.', ''): v for k, v in pretrained_state_dict.items()}
    
    # 定义模块名称到模型组件的映射
    module_mapping = {
        'vae': actual_model.vae,
        'content_encoder': actual_model.content_encoder,
        'style_encoder': actual_model.style_encoder,
        'unet': actual_model.unet
    }
    
    # 如果没有指定要加载的模块，则加载所有匹配的模块
    if modules_to_load is None or len(modules_to_load) == 0:
        modules_to_load = list(module_mapping.keys())
    
    loaded_modules = {}
    missing_keys = []
    unexpected_keys = []
    
    # 为每个模块加载权重
    for module_name in modules_to_load:
        if module_name not in module_mapping:
            if is_main_process:
                print(f"⚠️  警告: 未知的模块名称 '{module_name}'，跳过")
            continue
        
        module = module_mapping[module_name]
        module_state_dict = module.state_dict()
        
        # 提取该模块的预训练权重
        module_pretrained_dict = {}
        module_prefix = f"{module_name}."
        
        for key, value in pretrained_state_dict.items():
            if key.startswith(module_prefix):
                # 移除模块前缀
                module_key = key[len(module_prefix):]
                if module_key in module_state_dict:
                    # 检查形状是否匹配
                    if module_state_dict[module_key].shape == value.shape:
                        module_pretrained_dict[module_key] = value
                    else:
                        if is_main_process:
                            print(f"⚠️  警告: {key} 的形状不匹配，跳过")
                            print(f"    期望: {module_state_dict[module_key].shape}, 实际: {value.shape}")
                else:
                    unexpected_keys.append(key)
        
        # 加载权重
        if len(module_pretrained_dict) > 0:
            try:
                module.load_state_dict(module_pretrained_dict, strict=False)
                loaded_modules[module_name] = True
                if is_main_process:
                    print(f"✓ 已加载 {module_name} 模块 ({len(module_pretrained_dict)} 个参数)")
            except Exception as e:
                loaded_modules[module_name] = False
                if is_main_process:
                    print(f"❌ 加载 {module_name} 模块失败: {e}")
        else:
            loaded_modules[module_name] = False
            if is_main_process:
                print(f"⚠️  未找到 {module_name} 模块的匹配权重")
    
    # 打印加载摘要
    if is_main_process:
        loaded_list = [k for k, v in loaded_modules.items() if v]
        failed_list = [k for k, v in loaded_modules.items() if not v]
        
        print(f"\n预训练模型加载摘要:")
        print(f"  成功加载: {', '.join(loaded_list) if loaded_list else '无'}")
        if failed_list:
            print(f"  加载失败: {', '.join(failed_list)}")
        if unexpected_keys and strict:
            print(f"  未使用的键数量: {len(unexpected_keys)}")
    
    return loaded_modules


def set_module_training_stage(
    model: StyleEmbeddingDiffusionModel,
    config: Config,
    epoch: int,
    use_ddp: bool = False,
    is_main_process: bool = True
) -> Dict[str, Any]:
    """
    根据当前epoch设置哪些模块需要训练，并返回阶段配置信息
    
    Args:
        model: 模型实例
        config: 配置对象
        epoch: 当前epoch
        use_ddp: 是否使用DDP
        is_main_process: 是否为主进程
        
    Returns:
        stage_config: 字典，包含各模块的训练状态和阶段配置（学习率、调度器等）
    """
    # 获取实际模型（处理DDP包装）
    if isinstance(model, DDP):
        actual_model = model.module
    else:
        actual_model = model
    
    # 默认：所有模块都训练
    stage_config = {
        'vae': True,
        'content_encoder': True,
        'style_encoder': True,
        'unet': True,
        'learning_rate': config.training.learning_rate,
        'lr_scheduler': config.training.lr_scheduler,
        'min_lr': config.training.min_lr,
        'warmup_epochs': config.training.warmup_epochs,
    }
    
    # 如果配置了训练阶段，则根据当前epoch设置
    if config.training.training_stages is not None and len(config.training.training_stages) > 0:
        # 查找当前epoch所属的训练阶段
        for start_epoch, end_epoch, module_flags in config.training.training_stages:
            if start_epoch <= epoch <= end_epoch:
                stage_config = module_flags.copy()
                # 如果阶段配置中没有指定学习率等参数，使用全局配置
                if 'learning_rate' not in stage_config:
                    stage_config['learning_rate'] = config.training.learning_rate
                if 'lr_scheduler' not in stage_config:
                    stage_config['lr_scheduler'] = config.training.lr_scheduler
                if 'min_lr' not in stage_config:
                    stage_config['min_lr'] = config.training.min_lr
                if 'warmup_epochs' not in stage_config:
                    stage_config['warmup_epochs'] = config.training.warmup_epochs
                break
        
        # 设置各模块的requires_grad
        for param in actual_model.vae.parameters():
            param.requires_grad = stage_config['vae']
        
        for param in actual_model.content_encoder.parameters():
            param.requires_grad = stage_config['content_encoder']
        
        for param in actual_model.style_encoder.parameters():
            param.requires_grad = stage_config['style_encoder']
        
        for param in actual_model.unet.parameters():
            param.requires_grad = stage_config['unet']
        
        # DDP模式下同步所有进程
        if use_ddp:
            dist.barrier()
        
        # 打印当前训练阶段信息（只在主进程）
        if is_main_process:
            active_modules = [k for k in ['vae', 'content_encoder', 'style_encoder', 'unet'] if stage_config.get(k, False)]
            inactive_modules = [k for k in ['vae', 'content_encoder', 'style_encoder', 'unet'] if not stage_config.get(k, False)]
            if inactive_modules:
                print(f"  Epoch {epoch}: 训练模块: {', '.join(active_modules)} | 冻结模块: {', '.join(inactive_modules)}")
            print(f"  阶段学习率: {stage_config['learning_rate']:.2e}, 调度器: {stage_config['lr_scheduler']}")
    
    return stage_config


class StageLRScheduler:
    """
    阶段学习率调度器包装类，支持阶段切换时重置步数
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        stage_config: Dict[str, Any],
        train_loader: DataLoader,
        start_epoch: int,
        end_epoch: int,
        is_main_process: bool = True
    ):
        self.optimizer = optimizer
        self.stage_config = stage_config
        self.train_loader = train_loader
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.is_main_process = is_main_process
        
        # 计算阶段内的训练步数
        self.stage_steps = len(train_loader) * (end_epoch - start_epoch + 1)
        
        # 确定warmup步数
        warmup_epochs = stage_config.get('warmup_epochs', 5)
        self.warmup_steps = len(train_loader) * warmup_epochs
        
        # 获取阶段配置
        self.learning_rate = stage_config.get('learning_rate', 5e-5)
        self.scheduler_type = stage_config.get('lr_scheduler', 'cosine').lower()
        self.min_lr = stage_config.get('min_lr', 1e-6)
        
        # 阶段内的步数计数器（从0开始）
        self.stage_step = 0
        
        if is_main_process:
            print(f"  创建阶段调度器: {self.scheduler_type}, LR={self.learning_rate:.2e}, min_lr={self.min_lr:.2e}, warmup={warmup_epochs} epochs")
    
    def step(self):
        """更新学习率（在每次优化器步骤后调用）"""
        # 计算当前步的学习率倍数
        lr_multiplier = self._get_lr_multiplier(self.stage_step)
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate * lr_multiplier
        
        # 增加阶段步数
        self.stage_step += 1
    
    def _get_lr_multiplier(self, step: int) -> float:
        """根据步数计算学习率倍数"""
        if self.scheduler_type == "constant":
            return min(step / self.warmup_steps, 1.0) if self.warmup_steps > 0 else 1.0
        elif self.scheduler_type == "cosine":
            if step < self.warmup_steps:
                return step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            else:
                progress = (step - self.warmup_steps) / (self.stage_steps - self.warmup_steps) if self.stage_steps > self.warmup_steps else 0
                min_lr_ratio = self.min_lr / self.learning_rate
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.scheduler_type == "linear":
            if step < self.warmup_steps:
                return step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            else:
                progress = (step - self.warmup_steps) / (self.stage_steps - self.warmup_steps) if self.stage_steps > self.warmup_steps else 0
                min_lr_ratio = self.min_lr / self.learning_rate
                return max(min_lr_ratio, 1.0 - progress)
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.scheduler_type}")
    
    def get_last_lr(self):
        """获取当前学习率"""
        lr_multiplier = self._get_lr_multiplier(self.stage_step)
        return [self.learning_rate * lr_multiplier]
    
    def reset(self):
        """重置阶段步数（阶段切换时调用）"""
        self.stage_step = 0


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    stage_config: Dict[str, Any],
    train_loader: DataLoader,
    start_epoch: int,
    end_epoch: int,
    is_main_process: bool = True
) -> StageLRScheduler:
    """
    根据阶段配置创建学习率调度器
    
    Args:
        optimizer: 优化器
        stage_config: 阶段配置字典
        train_loader: 训练数据加载器
        start_epoch: 阶段开始epoch
        end_epoch: 阶段结束epoch
        is_main_process: 是否为主进程
        
    Returns:
        scheduler: 阶段学习率调度器
    """
    return StageLRScheduler(
        optimizer=optimizer,
        stage_config=stage_config,
        train_loader=train_loader,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        is_main_process=is_main_process
    )


def train_epoch(
    model: StyleEmbeddingDiffusionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: Config,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler = None,
    is_main_process: bool = True,
    wandb_logger: Optional[WandbLogger] = None,
    global_last_step: Optional[int] = None
):
    """训练一个epoch
    
    Args:
        global_last_step: 全局最后一个记录的 step，用于确保 step 单调递增
    
    Returns:
        dict: 包含平均损失和最后一个记录的 step
    """
    model.train()
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_vae_recon_loss = 0.0
    total_vae_kl_loss = 0.0
    total_style_kl_loss = 0.0
    total_diffusion_recon_loss = 0.0
    last_logged_step = None  # 记录最后一个实际记录的 step
    
    # 只在主进程显示进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process)
    for batch_idx, (I_bin, I_orig) in enumerate(pbar):
        I_bin = I_bin.to(config.device)
        I_orig = I_orig.to(config.device)
        
        optimizer.zero_grad()
        
        # 混合精度训练（仅在CUDA可用时使用）
        if scaler is not None and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                losses = model(I_bin, I_orig)
            
            scaler.scale(losses['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = model(I_bin, I_orig)
            losses['total_loss'].backward()
            optimizer.step()
        
        scheduler.step()
        
        # 记录损失
        total_loss += losses['total_loss'].item()
        total_diffusion_loss += losses['diffusion_loss'].item()
        total_vae_recon_loss += losses['vae_recon_loss'].item()
        total_vae_kl_loss += losses['vae_kl_loss'].item()
        total_diffusion_recon_loss += losses['diffusion_recon_loss'].item()
        if losses['style_kl_loss'] is not None:
            total_style_kl_loss += losses['style_kl_loss'].item()
        
        # 更新进度条
        postfix_dict = {
            'loss': f"{losses['total_loss'].item():.4f}",
            'diff': f"{losses['diffusion_loss'].item():.4f}",
            'vae_recon': f"{losses['vae_recon_loss'].item():.4f}",
            'vae_kl': f"{losses['vae_kl_loss'].item():.6f}",
            'style_kl': f"{losses['style_kl_loss'].item() if losses['style_kl_loss'] is not None else 0:.4f}"
        }
        if losses['diffusion_recon_loss'].item() > 0:
            postfix_dict['diff_recon'] = f"{losses['diffusion_recon_loss'].item():.4f}"
        pbar.set_postfix(postfix_dict)
        
        # 记录到 wandb（只在主进程）
        if is_main_process:
            global_step = epoch * len(dataloader) + batch_idx
            # 确保 step 单调递增：如果计算的 step 小于全局最后一个 step，则跳过或调整
            if global_last_step is not None and global_step <= global_last_step:
                # 跳过这个记录，或者使用全局最后一个 step + 1
                # 这里选择跳过，因为下一个 batch 的 step 会更大
                pass
            elif wandb_logger is not None and batch_idx % config.training.log_interval == 0:
                wandb_metrics = {
                    'train/loss': losses['total_loss'].item(),
                    'train/diffusion_loss': losses['diffusion_loss'].item(),
                    'train/vae_recon_loss': losses['vae_recon_loss'].item(),
                    'train/vae_kl_loss': losses['vae_kl_loss'].item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                }
                if losses['diffusion_recon_loss'].item() > 0:
                    wandb_metrics['train/diffusion_recon_loss'] = losses['diffusion_recon_loss'].item()
                if losses['style_kl_loss'] is not None:
                    wandb_metrics['train/style_kl_loss'] = losses['style_kl_loss'].item()
                
                # 记录 latent 统计信息（用于诊断 VAE latent 尺度问题）
                if config.model.log_latent_stats:
                    actual_model = model.module if isinstance(model, DDP) else model
                    if hasattr(actual_model, '_last_z0_stats'):
                        wandb_metrics.update({
                            'latent/z0_mean': actual_model._last_z0_stats['z0_mean'],
                            'latent/z0_std': actual_model._last_z0_stats['z0_std'],
                            'latent/z0_min': actual_model._last_z0_stats['z0_min'],
                            'latent/z0_max': actual_model._last_z0_stats['z0_max'],
                            'latent/z_t_mean': actual_model._last_z_t_stats['z_t_mean'],
                            'latent/z_t_std': actual_model._last_z_t_stats['z_t_std'],
                            'latent/z_t_min': actual_model._last_z_t_stats['z_t_min'],
                            'latent/z_t_max': actual_model._last_z_t_stats['z_t_max'],
                        })
                    # 记录风格嵌入统计信息（用于诊断风格分布）
                    if hasattr(actual_model, '_last_style_stats'):
                        wandb_metrics.update({
                            'style/mu_mean': actual_model._last_style_stats['mu_mean'],
                            'style/mu_std': actual_model._last_style_stats['mu_std'],
                            'style/mu_min': actual_model._last_style_stats['mu_min'],
                            'style/mu_max': actual_model._last_style_stats['mu_max'],
                            'style/logvar_mean': actual_model._last_style_stats['logvar_mean'],
                            'style/logvar_std': actual_model._last_style_stats['logvar_std'],
                            'style/logvar_min': actual_model._last_style_stats['logvar_min'],
                            'style/logvar_max': actual_model._last_style_stats['logvar_max'],
                            'style/z_style_mean': actual_model._last_style_stats['z_style_mean'],
                            'style/z_style_std': actual_model._last_style_stats['z_style_std'],
                            'style/z_style_min': actual_model._last_style_stats['z_style_min'],
                            'style/z_style_max': actual_model._last_style_stats['z_style_max'],
                            'style/raw_kl': actual_model._last_style_stats['raw_kl'],  # 原始KL散度（未加权）
                        })
                
                wandb_logger.log_metrics(wandb_metrics, step=global_step)
                last_logged_step = global_step
    
    # 确保最后一个 batch 也被记录（即使不满足 log_interval），避免 step 冲突
    if is_main_process and wandb_logger is not None:
        final_step = epoch * len(dataloader) + len(dataloader) - 1
        # 确保 step 单调递增
        if global_last_step is not None and final_step <= global_last_step:
            final_step = global_last_step + 1
        if last_logged_step is None or final_step > last_logged_step:
            # 记录最后一个 batch 的指标，确保 step 单调递增
            wandb_metrics = {
                'train/loss': total_loss / len(dataloader),
                'train/diffusion_loss': total_diffusion_loss / len(dataloader),
                'train/vae_recon_loss': total_vae_recon_loss / len(dataloader),
                'train/vae_kl_loss': total_vae_kl_loss / len(dataloader),
                'train/learning_rate': scheduler.get_last_lr()[0],
            }
            if total_diffusion_recon_loss > 0:
                wandb_metrics['train/diffusion_recon_loss'] = total_diffusion_recon_loss / len(dataloader)
            if total_style_kl_loss > 0:
                wandb_metrics['train/style_kl_loss'] = total_style_kl_loss / len(dataloader)
            wandb_logger.log_metrics(wandb_metrics, step=final_step)
            last_logged_step = final_step
    
    avg_loss = total_loss / len(dataloader)
    avg_diffusion_loss = total_diffusion_loss / len(dataloader)
    avg_vae_recon_loss = total_vae_recon_loss / len(dataloader)
    avg_vae_kl_loss = total_vae_kl_loss / len(dataloader)
    avg_style_kl_loss = total_style_kl_loss / len(dataloader)
    avg_diffusion_recon_loss = total_diffusion_recon_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'diffusion_loss': avg_diffusion_loss,
        'vae_recon_loss': avg_vae_recon_loss,
        'vae_kl_loss': avg_vae_kl_loss,
        'style_kl_loss': avg_style_kl_loss,
        'diffusion_recon_loss': avg_diffusion_recon_loss,
        'last_logged_step': last_logged_step if is_main_process else None
    }


def validate(
    model: StyleEmbeddingDiffusionModel,
    dataloader: DataLoader,
    config: Config,
    epoch: int,
    train_loader: DataLoader,
    last_logged_step: Optional[int] = None,
    use_ddp: bool = False,
    is_main_process: bool = True,
    wandb_logger: Optional[WandbLogger] = None
):
    """验证
    
    Args:
        last_logged_step: 训练循环中最后一个记录的 step，用于确保验证 step 单调递增
    
    Returns:
        tuple: (avg_loss, validation_step) - 平均损失和验证时使用的 step
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        # 只在主进程显示进度条
        pbar = tqdm(dataloader, desc="Validating", disable=not is_main_process)
        for I_bin, I_orig in pbar:
            I_bin = I_bin.to(config.device)
            I_orig = I_orig.to(config.device)
            
            losses = model(I_bin, I_orig)
            total_loss += losses['total_loss'].item()
    
    avg_loss = total_loss / len(dataloader)
    
    # DDP时聚合所有进程的损失
    if use_ddp:
        # 创建tensor用于all_reduce
        loss_tensor = torch.tensor(avg_loss, device=config.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    
    # 记录到 wandb
    # 使用 epoch 结束时的 step，确保单调递增
    validation_step = None
    if is_main_process and wandb_logger is not None:
        # 计算 epoch 结束时的 step（与训练时的 global_step 保持一致）
        # 使用训练循环中最后一个记录的 step + 1，确保不冲突
        if last_logged_step is not None:
            validation_step = last_logged_step + 1
        else:
            # 如果没有记录过，使用下一个 epoch 的第一个 step
            validation_step = (epoch + 1) * len(train_loader)
        wandb_logger.log_metrics({
            'val/loss': avg_loss, 
            'epoch': epoch
        }, step=validation_step)
    
    return avg_loss, validation_step


def main_worker(rank, world_size, args):
    """
    单个训练进程的工作函数（用于DDP）
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        args: 命令行参数
    """
    # 设置当前进程的GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    
    # 设置DDP环境变量（用于后续代码检测）
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 调用实际的训练函数
    _train_main(args, device, rank, world_size, is_main_process=(rank == 0))


def _train_main(args, device, rank=0, world_size=1, is_main_process=True):
    """
    实际的训练主函数
    
    Args:
        args: 命令行参数
        device: 设备
        rank: 当前进程rank（单GPU时为0）
        world_size: 总进程数（单GPU时为1）
        is_main_process: 是否为主进程
    """
    # 初始化分布式训练
    use_ddp = False
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 检查环境变量（torchrun和torch.distributed.launch都会设置）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 使用torchrun或torch.distributed.launch启动，或由main_worker设置
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # LOCAL_RANK可能不存在（torch.distributed.launch使用--local-rank参数）
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
        elif args.local_rank == -1:
            # 如果没有LOCAL_RANK环境变量，尝试从RANK推断（单节点情况）
            args.local_rank = args.rank
        
        # 如果WORLD_SIZE > 1，使用DDP
        if args.world_size > 1:
            use_ddp = True
            print(f"[DEBUG] 检测到DDP环境变量: RANK={args.rank}, WORLD_SIZE={args.world_size}, LOCAL_RANK={args.local_rank}")
        else:
            use_ddp = False
            if is_main_process:
                print(f"[DEBUG] 检测到DDP环境变量但WORLD_SIZE=1，禁用DDP，使用单GPU训练")
    elif args.local_rank != -1:
        # 手动设置DDP
        args.rank = args.rank if args.rank != -1 else args.local_rank
        args.world_size = args.world_size if args.world_size != -1 else num_available_gpus
        use_ddp = True
        print(f"[DEBUG] 手动设置DDP: rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}")
    else:
        # 没有DDP环境变量，单GPU训练
        use_ddp = False
        if is_main_process:
            print(f"[DEBUG] 未检测到DDP环境变量，使用单GPU训练")
    
    if use_ddp:
        # 初始化进程组
        try:
            # 优先使用环境变量（torchrun会自动设置），如果没有则使用命令行参数
            master_addr = os.environ.get('MASTER_ADDR') or args.master_addr or 'localhost'
            master_port = os.environ.get('MASTER_PORT') or args.master_port or '12355'
            
            if is_main_process:
                print(f"[DEBUG] 正在初始化DDP进程组...")
                print(f"  backend=nccl, rank={args.rank}, world_size={args.world_size}")
                print(f"  master_addr={master_addr}, master_port={master_port}")
            
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{master_addr}:{master_port}',
                rank=args.rank,
                world_size=args.world_size,
                timeout=datetime.timedelta(seconds=1800)  # 30分钟超时
            )
            torch.cuda.set_device(args.local_rank)
            device = torch.device(f'cuda:{args.local_rank}')
            is_main_process = (args.rank == 0)
            if is_main_process:
                print(f"[DEBUG] ✓ DDP进程组初始化成功")
        except Exception as e:
            print(f"[ERROR] DDP初始化失败: {e}")
            print(f"回退到单GPU训练")
            use_ddp = False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            is_main_process = True
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    
    # 加载配置
    config = Config()
    if args.data_root:
        config.training.data_root = args.data_root
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    config.device = str(device)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        
        # 检查GPU内存使用情况
        if is_main_process:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                print(f"  GPU {i} 内存: {allocated:.2f}GB已分配 / {reserved:.2f}GB已保留 / {total:.2f}GB总计")
            
            # 如果GPU内存被占用，尝试清理
            if torch.cuda.memory_allocated(0) > 0:
                print(f"  ⚠️  检测到GPU内存被占用，尝试清理...")
                torch.cuda.empty_cache()
                allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                print(f"  清理后GPU 0内存: {allocated_after:.2f}GB")
    
    # 只在主进程打印信息
    if is_main_process:
        cuda_available = torch.cuda.is_available()
        num_gpus = args.world_size if use_ddp else (torch.cuda.device_count() if cuda_available else 0)
        
        if cuda_available:
            print(f"✓ CUDA可用: {torch.cuda.get_device_name(args.local_rank)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            if use_ddp:
                print(f"  使用DDP训练")
                print(f"  总进程数: {args.world_size}")
                print(f"  当前rank: {args.rank}")
                print(f"  每张GPU的batch_size: {config.training.batch_size // args.world_size}")
                print(f"  总有效batch_size: {config.training.batch_size}")
            else:
                print(f"  设备数量: {num_gpus}")
        else:
            print("⚠ CUDA不可用，将使用CPU训练（速度较慢）")
        
        print(f"使用设备: {config.device}")
    
    # 创建目录（只在主进程）
    if is_main_process:
        # 自动生成唯一目录名，避免多次训练覆盖文件
        base_checkpoint_dir = config.training.checkpoint_dir
        user_specified_dir = getattr(args, 'checkpoint_dir', None)
        
        # 如果用户通过命令行明确指定了目录，使用指定的（允许覆盖）
        # 否则自动生成带时间戳的唯一目录名
        if user_specified_dir and user_specified_dir != 'checkpoints':
            # 用户明确指定了目录，使用指定的
            config.training.checkpoint_dir = user_specified_dir
        else:
            # 自动生成唯一目录名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 如果配置中有时间戳占位符，替换它
            if '{timestamp}' in base_checkpoint_dir:
                config.training.checkpoint_dir = base_checkpoint_dir.replace('{timestamp}', timestamp)
            elif base_checkpoint_dir == "checkpoints":
                # 默认格式：checkpoints-YYYYMMDD_HHMMSS
                config.training.checkpoint_dir = f"checkpoints-{timestamp}"
            elif base_checkpoint_dir.startswith("checkpoints-") and len(base_checkpoint_dir.split('-')) == 2:
                # 如果已经是 checkpoints-xxx 格式但没有时间戳，添加时间戳
                config.training.checkpoint_dir = f"checkpoints-{timestamp}"
            else:
                # 其他情况，追加时间戳
                config.training.checkpoint_dir = f"{base_checkpoint_dir}-{timestamp}"
        
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)
        
        print(f"✓ 检查点保存目录: {config.training.checkpoint_dir}")
    
    # 同步所有进程
    if use_ddp:
        dist.barrier()
    
    # 创建数据集
    train_dataset_full = QRCodeDataset(
        data_root=config.training.data_root,
        image_size=config.training.image_size,
        split='train'
    )
    val_dataset_full = QRCodeDataset(
        data_root=config.training.data_root,
        image_size=config.training.image_size,
        split='val'
    )
    
    # 如果指定了数据比例，只使用部分数据（用于快速测试）
    if args.data_ratio < 1.0:
        train_size = int(len(train_dataset_full) * args.data_ratio)
        val_size = int(len(val_dataset_full) * args.data_ratio)
        
        # 随机选择索引
        train_indices = torch.randperm(len(train_dataset_full))[:train_size].tolist()
        val_indices = torch.randperm(len(val_dataset_full))[:val_size].tolist()
        
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)
        
        if is_main_process:
            print(f"⚠️  使用部分数据训练:")
            print(f"   训练集: {train_size}/{len(train_dataset_full)} ({args.data_ratio*100:.1f}%%)")
            print(f"   验证集: {val_size}/{len(val_dataset_full)} ({args.data_ratio*100:.1f}%%)")
    else:
        train_dataset = train_dataset_full
        val_dataset = val_dataset_full
        if is_main_process:
            print(f"使用全部数据:")
            print(f"   训练集: {len(train_dataset)} 样本")
            print(f"   验证集: {len(val_dataset)} 样本")
    
    # DDP时调整batch_size（每张GPU的batch_size）
    if use_ddp:
        per_gpu_batch_size = config.training.batch_size // args.world_size
        if config.training.batch_size % args.world_size != 0:
            if is_main_process:
                old_batch_size = config.training.batch_size
                per_gpu_batch_size = config.training.batch_size // args.world_size
                print(f"\n⚠️  警告: batch_size ({old_batch_size}) 不能被GPU数量 ({args.world_size}) 整除")
                print(f"  每张GPU的batch_size: {per_gpu_batch_size}")
                print(f"  总有效batch_size: {per_gpu_batch_size * args.world_size}")
        per_gpu_batch_size = config.training.batch_size // args.world_size
        
        # 检查每张GPU的batch_size是否太小
        if per_gpu_batch_size < 1:
            if is_main_process:
                print(f"\n❌ 错误: 每张GPU的batch_size ({per_gpu_batch_size}) 太小！")
                print(f"  总batch_size ({config.training.batch_size}) 必须 >= GPU数量 ({args.world_size})")
                print(f"  建议: 将batch_size设置为至少 {args.world_size} 或更大")
            raise ValueError(f"batch_size ({config.training.batch_size}) 太小，无法分配给 {args.world_size} 张GPU")
        
        if per_gpu_batch_size == 1 and is_main_process:
            print(f"\n⚠️  警告: 每张GPU的batch_size=1，这可能导致训练不稳定或效率低下")
            print(f"  建议: 将batch_size设置为至少 {args.world_size * 2} 或更大")
    else:
        per_gpu_batch_size = config.training.batch_size
    
    # 创建DistributedSampler（DDP时使用）
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=config.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
        shuffle = False  # DistributedSampler已经处理shuffle
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=4,  # DDP时减少worker数量
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # 创建模型
    model = StyleEmbeddingDiffusionModel(config).to(device)
    
    # 加载预训练模型（在DDP包装之前）
    pretrained_path = args.pretrained or config.training.pretrained_model_path
    if pretrained_path:
        if is_main_process:
            print(f"\n加载预训练模型: {pretrained_path}")
        loaded_modules = load_pretrained_model(
            model=model,
            pretrained_path=pretrained_path,
            device=device,
            modules_to_load=config.training.pretrained_modules,
            strict=config.training.pretrained_strict,
            is_main_process=is_main_process
        )
        # DDP模式下同步所有进程
        if use_ddp:
            dist.barrier()
    
    # DDP包装模型
    if use_ddp:
        # 如果使用分阶段训练，某些模块可能被冻结，需要设置find_unused_parameters=True
        # 否则设置为False可以提升性能（避免每次迭代额外遍历autograd图）
        use_find_unused_params = (
            config.training.training_stages is not None and 
            len(config.training.training_stages) > 0
        )
        model = DDP(
            model, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank, 
            find_unused_parameters=use_find_unused_params
        )
        if is_main_process:
            print(f"\n✓ 使用DDP在 {args.world_size} 张GPU上训练")
            print(f"  每张GPU的batch_size: {per_gpu_batch_size}")
            print(f"  总有效batch_size: {per_gpu_batch_size * args.world_size}")
            if use_find_unused_params:
                print(f"  注意: 检测到分阶段训练，已启用find_unused_parameters=True（可能略微降低性能）")
    
    # 创建优化器和调度器
    # 如果配置了不同模块的学习率，则分别为各模块创建参数组
    if config.training.module_learning_rates is not None:
        # 获取实际模型（处理DDP包装）
        if isinstance(model, DDP):
            actual_model = model.module
        else:
            actual_model = model
        
        # 为不同模块创建参数组
        param_groups = []
        
        # VAE参数组
        if len(list(actual_model.vae.parameters())) > 0:
            vae_lr = config.training.module_learning_rates.get('vae', config.training.learning_rate)
            param_groups.append({
                'params': actual_model.vae.parameters(),
                'lr': vae_lr,
                'name': 'vae'
            })
        
        # 内容编码器参数组
        if len(list(actual_model.content_encoder.parameters())) > 0:
            content_lr = config.training.module_learning_rates.get('content_encoder', config.training.learning_rate)
            param_groups.append({
                'params': actual_model.content_encoder.parameters(),
                'lr': content_lr,
                'name': 'content_encoder'
            })
        
        # 风格编码器参数组
        if len(list(actual_model.style_encoder.parameters())) > 0:
            style_lr = config.training.module_learning_rates.get('style_encoder', config.training.learning_rate)
            param_groups.append({
                'params': actual_model.style_encoder.parameters(),
                'lr': style_lr,
                'name': 'style_encoder'
            })
        
        # UNet参数组
        if len(list(actual_model.unet.parameters())) > 0:
            unet_lr = config.training.module_learning_rates.get('unet', config.training.learning_rate)
            param_groups.append({
                'params': actual_model.unet.parameters(),
                'lr': unet_lr,
                'name': 'unet'
            })
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config.training.weight_decay
        )
        
        if is_main_process:
            print(f"✓ 使用模块特定学习率:")
            for group in param_groups:
                print(f"  {group['name']}: {group['lr']}")
    else:
        # 统一学习率
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    
    # 创建初始学习率调度器
    # 如果配置了分阶段训练，使用第一个阶段的配置；否则使用全局配置
    if config.training.training_stages is not None and len(config.training.training_stages) > 0:
        # 使用第一个阶段的配置
        first_stage_start, first_stage_end, first_stage_config = config.training.training_stages[0]
        initial_stage_config = {
            'learning_rate': first_stage_config.get('learning_rate', config.training.learning_rate),
            'lr_scheduler': first_stage_config.get('lr_scheduler', config.training.lr_scheduler),
            'min_lr': first_stage_config.get('min_lr', config.training.min_lr),
            'warmup_epochs': first_stage_config.get('warmup_epochs', config.training.warmup_epochs),
        }
        scheduler = create_lr_scheduler(
            optimizer=optimizer,
            stage_config=initial_stage_config,
            train_loader=train_loader,
            start_epoch=first_stage_start,
            end_epoch=first_stage_end,
            is_main_process=is_main_process
        )
        if is_main_process:
            print(f"✓ 使用分阶段训练，初始阶段: Epoch {first_stage_start}-{first_stage_end}")
    else:
        # 使用全局配置创建传统调度器
        total_steps = len(train_loader) * config.training.num_epochs
        
        # 确定warmup步数：优先使用warmup_steps，如果为None则使用warmup_epochs计算
        if config.training.warmup_steps is not None:
            warmup_steps = config.training.warmup_steps
            warmup_type = "步数"
        else:
            warmup_steps = len(train_loader) * config.training.warmup_epochs
            warmup_type = f"{config.training.warmup_epochs}个epoch"
        
        scheduler_type = config.training.lr_scheduler.lower()
        if scheduler_type == "constant":
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(step / warmup_steps, 1.0)
            )
        elif scheduler_type == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    min_lr_ratio = config.training.min_lr / config.training.learning_rate
                    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif scheduler_type == "linear":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    min_lr_ratio = config.training.min_lr / config.training.learning_rate
                    return max(min_lr_ratio, 1.0 - progress)
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(f"Unknown lr_scheduler: {scheduler_type}. Choose from 'constant', 'cosine', 'linear'")
        
        if is_main_process:
            print(f"✓ 学习率调度策略: {scheduler_type}")
            print(f"  初始学习率: {config.training.learning_rate}")
            print(f"  Warmup: {warmup_type} ({warmup_steps}步)")
            print(f"  总训练步数: {total_steps}")
            if scheduler_type != "constant":
                print(f"  最小学习率: {config.training.min_lr}")
    
    # 混合精度（仅在CUDA可用时启用）
    # 检查device是否为CUDA设备
    is_cuda_device = torch.cuda.is_available() and (
        str(device).startswith('cuda') or 
        (hasattr(device, 'type') and device.type == 'cuda')
    )
    use_amp = config.training.use_amp and is_cuda_device
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if config.training.use_amp and not use_amp and is_main_process:
        print(f"警告: 混合精度训练已请求，但CUDA不可用。已禁用AMP。")
    elif use_amp and is_main_process:
        print(f"✓ 混合精度训练已启用")
    
    # wandb（只在主进程，必选）
    wandb_logger = None
    if is_main_process:
        try:
            # 构建tags列表
            tags = []
            if hasattr(args, 'wandb_tags') and args.wandb_tags:
                if isinstance(args.wandb_tags, list):
                    tags = list(args.wandb_tags)
                elif isinstance(args.wandb_tags, str):
                    tags = [args.wandb_tags]
            
            # 自动添加一些标签
            if args.resume:
                tags.append("resume")
            
            # 根据训练阶段自动添加标签
            if config.training.training_stages is not None and len(config.training.training_stages) > 0:
                tags.append("staged_training")
                # 添加阶段数量标签
                num_stages = len(config.training.training_stages)
                tags.append(f"stages_{num_stages}")
            
            # 去重并排序
            tags = sorted(list(set(tags)))
            
            wandb_logger = WandbLogger(
                config=config,
                project_name=getattr(args, 'wandb_project', 'style_embedding_diffusion'),
                run_name=getattr(args, 'wandb_name', None),
                tags=tags,
                enabled=True,
                is_main_process=is_main_process
            )
            if wandb_logger.enabled:
                print("✓ wandb 已启用")
        except Exception as e:
            print(f"错误: wandb 初始化失败: {e}")
            print("提示: 请确保已安装 wandb (pip install wandb) 并已登录 (wandb login)")
            raise
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        # 加载checkpoint（所有进程都需要加载模型状态）
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['model_state_dict']
        # 处理DDP的情况：移除module前缀（如果checkpoint是DDP保存的）
        if use_ddp:
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # 加载模型状态
        if isinstance(model, DDP):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        # 只在主进程加载优化器和调度器状态
        if is_main_process:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 检查检查点中的调度器类型
            if 'scheduler_type' in checkpoint and checkpoint['scheduler_type'] == 'stage':
                # StageLRScheduler：重新创建当前阶段的调度器
                stage_config = set_module_training_stage(model, config, checkpoint['epoch'], use_ddp, is_main_process)
                if config.training.training_stages is not None and len(config.training.training_stages) > 0:
                    for stage_start, stage_end, _ in config.training.training_stages:
                        if stage_start <= checkpoint['epoch'] <= stage_end:
                            scheduler = create_lr_scheduler(
                                optimizer=optimizer,
                                stage_config=stage_config,
                                train_loader=train_loader,
                                start_epoch=stage_start,
                                end_epoch=stage_end,
                                is_main_process=is_main_process
                            )
                            # 恢复阶段步数
                            if 'scheduler_stage_step' in checkpoint:
                                scheduler.stage_step = checkpoint['scheduler_stage_step']
                            break
            elif isinstance(scheduler, StageLRScheduler):
                # 如果当前是 StageLRScheduler 但检查点不是，重新创建
                stage_config = set_module_training_stage(model, config, checkpoint['epoch'], use_ddp, is_main_process)
                if config.training.training_stages is not None and len(config.training.training_stages) > 0:
                    for stage_start, stage_end, _ in config.training.training_stages:
                        if stage_start <= checkpoint['epoch'] <= stage_end:
                            scheduler = create_lr_scheduler(
                                optimizer=optimizer,
                                stage_config=stage_config,
                                train_loader=train_loader,
                                start_epoch=stage_start,
                                end_epoch=stage_end,
                                is_main_process=is_main_process
                            )
                            break
            else:
                # 传统调度器：加载状态
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"从epoch {start_epoch}恢复训练")
        
        # 同步所有进程
        if use_ddp:
            dist.barrier()
            # 广播start_epoch到所有进程
            start_epoch_tensor = torch.tensor([start_epoch], device=device)
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()
    
    # 训练循环
    best_val_loss = float('inf')
    stage_start_epoch = None
    stage_end_epoch = None
    
    # 跟踪全局最后一个记录的 step，确保 step 单调递增
    global_last_step = None
    
    # 初始化阶段跟踪（如果使用分阶段训练）
    if config.training.training_stages is not None and len(config.training.training_stages) > 0:
        for start_epoch_stage, end_epoch_stage, _ in config.training.training_stages:
            if start_epoch_stage <= start_epoch <= end_epoch_stage:
                stage_start_epoch = start_epoch_stage
                stage_end_epoch = end_epoch_stage
                break
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # DDP时设置sampler的epoch（确保每个epoch的数据顺序不同）
        if use_ddp:
            train_sampler.set_epoch(epoch)
        
        # 设置当前epoch的训练模块（分阶段训练控制）
        stage_config = set_module_training_stage(model, config, epoch, use_ddp, is_main_process)
        
        # 检测阶段切换：如果配置了训练阶段，检查是否需要切换调度器
        if config.training.training_stages is not None and len(config.training.training_stages) > 0:
            # 查找当前epoch所属的训练阶段
            new_stage_start = None
            new_stage_end = None
            for start_epoch_stage, end_epoch_stage, _ in config.training.training_stages:
                if start_epoch_stage <= epoch <= end_epoch_stage:
                    new_stage_start = start_epoch_stage
                    new_stage_end = end_epoch_stage
                    break
            
            # 如果阶段发生变化，重新创建调度器并更新学习率
            if (new_stage_start != stage_start_epoch or new_stage_end != stage_end_epoch):
                stage_start_epoch = new_stage_start
                stage_end_epoch = new_stage_end
                
                if is_main_process:
                    print(f"\n🔄 切换到新训练阶段: Epoch {stage_start_epoch}-{stage_end_epoch}")
                
                # 更新优化器的学习率
                stage_lr = stage_config.get('learning_rate', config.training.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = stage_lr
                
                # 重新创建调度器（会自动重置步数）
                scheduler = create_lr_scheduler(
                    optimizer=optimizer,
                    stage_config=stage_config,
                    train_loader=train_loader,
                    start_epoch=stage_start_epoch,
                    end_epoch=stage_end_epoch,
                    is_main_process=is_main_process
                )
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch, scaler, is_main_process, wandb_logger,
            global_last_step=global_last_step
        )
        
        # 获取训练循环中最后一个记录的 step
        last_logged_step = train_metrics.get('last_logged_step') if is_main_process else None
        # 更新全局最后一个 step
        if last_logged_step is not None:
            global_last_step = last_logged_step
        
        # 验证
        val_loss, validation_step = validate(
            model, val_loader, config, epoch, train_loader, 
            last_logged_step=last_logged_step,
            use_ddp=use_ddp, is_main_process=is_main_process, wandb_logger=wandb_logger
        )
        # 更新全局最后一个 step
        if validation_step is not None:
            global_last_step = validation_step
        
        # 只在主进程打印
        if is_main_process:
            print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, val_loss={val_loss:.4f}")
        
        # 生成并保存样本图像（方便监控训练进度，只在主进程）
        if is_main_process and config.training.save_samples and (epoch + 1) % config.training.sample_interval == 0:
            sample_output_dir = os.path.join(config.training.checkpoint_dir, 'samples')
            save_sample_images(
                model, val_loader, epoch, sample_output_dir,
                num_samples=config.training.num_sample_images,
                num_inference_steps=50,
                device=config.device
            )
            print(f"保存样本图像到: {sample_output_dir}")
            
            # 记录样本图像到 wandb
            if wandb_logger is not None:
                try:
                    from PIL import Image
                    sample_images = []
                    for idx in range(config.training.num_sample_images):
                        # 图像保存在 sample_{idx:02d}/epoch_{epoch:03d}.png
                        img_path = os.path.join(sample_output_dir, f"sample_{idx:02d}", f"epoch_{epoch:03d}.png")
                        if os.path.exists(img_path):
                            sample_images.append(Image.open(img_path))
                    if sample_images:
                        # 使用 log_images 函数记录图像，图像会在wandb的时间轴上按epoch显示
                        # 每个样本使用独立的key：samples/sample_{idx:02d}
                        # 这样在wandb的Media面板中，每个sample会形成一条时间序列，x轴按epoch显示
                        # 使用比验证步骤大 1 的 step，确保 step 单调递增且不冲突
                        if validation_step is not None:
                            # 使用验证步骤 + 1，确保不冲突
                            image_step = validation_step + 1
                        else:
                            # 如果没有验证步骤，使用训练循环最后一个 step + 2
                            if last_logged_step is not None:
                                image_step = last_logged_step + 2
                            else:
                                # 如果没有记录过，使用下一个 epoch 的第一个 step + 1
                                image_step = (epoch + 1) * len(train_loader) + 1
                        # 确保 step 单调递增：如果计算的 step 小于全局最后一个 step，则调整
                        if global_last_step is not None and image_step <= global_last_step:
                            image_step = global_last_step + 1
                        captions = [f"Epoch {epoch:03d}, Sample {i:02d}" for i in range(len(sample_images))]
                        wandb_logger.log_images(
                            images=sample_images,
                            key="samples",
                            step=image_step,  # 使用比验证大 1 的 step，确保不冲突
                            captions=captions,
                            use_time_series=True
                        )
                        # 更新全局最后一个 step
                        global_last_step = image_step
                except Exception as e:
                    print(f"警告: 记录样本图像到 wandb 失败: {e}")
        
        # 保存检查点（只在主进程）
        if is_main_process and (epoch + 1) % config.training.save_interval == 0:
            checkpoint_path = os.path.join(
                config.training.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            # 保存模型状态：DDP或DataParallel时，保存model.module.state_dict()以便单GPU加载
            if isinstance(model, (DDP, nn.DataParallel)):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            # StageLRScheduler 不支持 state_dict，保存阶段信息用于恢复
            if isinstance(scheduler, StageLRScheduler):
                checkpoint_data['scheduler_type'] = 'stage'
                checkpoint_data['scheduler_stage_step'] = scheduler.stage_step
                checkpoint_data['scheduler_stage_config'] = scheduler.stage_config
            else:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
            
            # 保存检查点到 wandb（可选）
            if wandb_logger is not None and config.training.save_checkpoints_to_wandb:
                wandb_logger.log_model_checkpoint(
                    checkpoint_path=checkpoint_path,
                    artifact_name=f"checkpoint_epoch_{epoch+1}",
                    artifact_type="checkpoint",
                    metadata={
                        'epoch': epoch,
                        'val_loss': val_loss,
                    }
                )
        
        # 保存最佳模型（只在主进程）
        if is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.training.checkpoint_dir, 'best_model.pth')
            # 保存模型状态：DDP或DataParallel时，保存model.module.state_dict()以便单GPU加载
            if isinstance(model, (DDP, nn.DataParallel)):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'val_loss': val_loss,
            }, best_path)
            print(f"保存最佳模型: {best_path}")
            
            # 更新 wandb 摘要
            if wandb_logger is not None:
                wandb_logger.update_summary({
                    'best_val_loss': best_val_loss,
                    'best_epoch': epoch
                })
                
                # 保存最佳模型到 wandb（可选）
                if config.training.save_best_model_to_wandb:
                    wandb_logger.log_model_checkpoint(
                        checkpoint_path=best_path,
                        artifact_name="best_model",
                        artifact_type="best_model",
                        metadata={
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'is_best': True,
                        }
                    )
    
    # 清理
    if is_main_process:
        if wandb_logger is not None:
            wandb_logger.finish()
        print("训练完成！")
    
    # 清理分布式进程组
    if use_ddp:
        dist.destroy_process_group()


def main():
    """
    主入口函数：解析参数，检测多GPU并自动启动DDP训练
    """
    parser = argparse.ArgumentParser(description='训练风格嵌入扩散模型')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--data_root', type=str, default='data', help='数据根目录')
    
    # job_name: 统一管理实验名称，会自动设置 checkpoint_dir 和 wandb_name
    parser.add_argument('--job_name', type=str, default=None, help='实验名称（会自动设置 checkpoint_dir=checkpoints/{job_name} 和 wandb_name={job_name}）')
    
    # 以下参数保持向后兼容，但如果指定了 job_name，会被 job_name 覆盖
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录（如果指定了job_name，会被覆盖为checkpoints/{job_name}）')
    parser.add_argument('--batch_size', type=int, default=None, help='总batch_size（所有GPU的总和），DDP时会自动分配到各GPU')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径（包含优化器和调度器状态）')
    parser.add_argument('--pretrained', type=str, default=None, help='预训练模型路径（只加载模型权重，不加载优化器状态）')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='使用数据的比例（0.0-1.0），例如0.01表示使用1%%的数据')
    
    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='style_embedding_diffusion', help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb运行名称（如果指定了job_name，会被覆盖为{job_name}；如果为None，会自动生成）')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='wandb标签列表，例如: --wandb_tags baseline stage1 high_lr')
    
    # DDP相关参数（torchrun会自动设置这些环境变量，通常不需要手动指定）
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1, dest='local_rank', 
                       help='本地GPU rank（torchrun自动设置，通常不需要手动指定）')
    parser.add_argument('--world_size', type=int, default=-1, 
                       help='总进程数（torchrun自动设置，通常不需要手动指定）')
    parser.add_argument('--rank', type=int, default=-1, 
                       help='全局rank（torchrun自动设置，通常不需要手动指定）')
    parser.add_argument('--master_addr', '--master-addr', type=str, 
                       default=None, dest='master_addr', 
                       help='主节点地址（torchrun自动设置，通常不需要手动指定）')
    parser.add_argument('--master_port', '--master-port', type=str, 
                       default=None, dest='master_port', 
                       help='主节点端口（torchrun自动设置，通常不需要手动指定）')
    
    args = parser.parse_args()
    
    # 如果指定了 job_name，自动设置 checkpoint_dir 和 wandb_name
    if args.job_name:
        args.checkpoint_dir = f"checkpoints/{args.job_name}"
        args.wandb_name = args.job_name
        print(f"✓ 使用 job_name: {args.job_name}")
        print(f"  - checkpoint_dir: {args.checkpoint_dir}")
        print(f"  - wandb_name: {args.wandb_name}")
    
    # 检测GPU数量
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 检查是否已有DDP环境变量（torchrun启动）
    has_ddp_env = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    # 如果使用torchrun但WORLD_SIZE=1，且有多GPU，提示用户
    if has_ddp_env:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size == 1 and num_available_gpus > 1:
            # torchrun启动了但只启动了1个进程，提示用户需要指定--nproc_per_node
            print(f"\n⚠️  检测到 {num_available_gpus} 张GPU，但torchrun只启动了1个进程")
            print(f"   要使用所有GPU，请重新运行并指定 --nproc_per_node：")
            print(f"\n   torchrun --nproc_per_node={num_available_gpus} train.py \\")
            cmd_parts = []
            if args.data_root and args.data_root != 'data':
                cmd_parts.append(f"       --data_root {args.data_root}")
            if args.job_name:
                cmd_parts.append(f"       --job_name {args.job_name}")
            elif args.checkpoint_dir and args.checkpoint_dir != 'checkpoints':
                cmd_parts.append(f"       --checkpoint_dir {args.checkpoint_dir}")
            if args.config:
                cmd_parts.append(f"       --config {args.config}")
            if args.resume:
                cmd_parts.append(f"       --resume {args.resume}")
            if cmd_parts:
                print(" \\\n".join(cmd_parts))
            print(f"\n   或者直接运行（会自动使用所有GPU）：")
            print(f"   python train.py" + (" " + " ".join(cmd_parts).replace(" \\\n       ", " ").replace("       ", "")) if cmd_parts else "")
            print(f"\n   继续使用单GPU训练...\n")
    
    # 如果没有DDP环境变量，且有多GPU，自动启动多进程训练
    if not has_ddp_env and num_available_gpus > 1:
        print(f"\n✓ 检测到 {num_available_gpus} 张GPU，自动启动多GPU训练")
        print(f"  使用 torch.multiprocessing.spawn 启动 {num_available_gpus} 个进程\n")
        
        # 使用spawn方法启动多进程
        torch.multiprocessing.spawn(
            main_worker,
            args=(num_available_gpus, args),
            nprocs=num_available_gpus,
            join=True
        )
    else:
        # 单GPU训练或已有DDP环境变量（WORLD_SIZE > 1）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _train_main(args, device, rank=0, world_size=1, is_main_process=True)


if __name__ == '__main__':
    main()

