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
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import datetime

from config import Config
from models import ContentEncoder, StyleEncoder, VAE, ConditionalUNet
from utils import DiffusionScheduler, QRCodeDataset


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
            
            # 生成图像
            I_gen = generate_image(model, I_bin, I_orig, num_inference_steps)
            
            # 转换为numpy并保存
            # 图像范围是[-1, 1]，需要转换到[0, 255]
            I_bin_np = ((I_bin[0, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            I_orig_np = ((I_orig[0, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            I_gen_np = ((I_gen[0, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            
            # 创建对比图像（水平拼接）
            comparison = np.hstack([I_bin_np, I_orig_np, I_gen_np])
            
            # 保存
            img_path = os.path.join(output_dir, f"epoch_{epoch:03d}_sample_{idx:02d}.png")
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
            z_t = actual_model.scheduler.step(eps_pred, t_batch, z_t)
    
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
        
        # 计算VAE重建损失（L1损失，对图像重建更稳定）
        vae_recon_loss = nn.functional.l1_loss(x_recon, I_orig)
        
        # 2. 随机采样扩散步并加噪
        batch_size = I_bin.size(0)
        # 确保时间步采样在与数据相同的设备上（支持DataParallel）
        device = I_bin.device
        t = self.scheduler.sample_timesteps(batch_size).to(device)
        noise = torch.randn_like(z0)
        z_t = self.scheduler.add_noise(z0, t, noise)
        
        # 3. 提取条件信息
        F_content = self.content_encoder(I_bin)
        z_style, style_kl_loss = self.style_encoder(I_orig)
        
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


def train_epoch(
    model: StyleEmbeddingDiffusionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: Config,
    epoch: int,
    writer: SummaryWriter,
    scaler: torch.cuda.amp.GradScaler = None,
    is_main_process: bool = True
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_vae_recon_loss = 0.0
    total_vae_kl_loss = 0.0
    total_style_kl_loss = 0.0
    total_diffusion_recon_loss = 0.0
    
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
        
        # 记录到tensorboard（只在主进程）
        if is_main_process:
            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % config.training.log_interval == 0 and writer is not None:
                writer.add_scalar('train/loss', losses['total_loss'].item(), global_step)
                writer.add_scalar('train/diffusion_loss', losses['diffusion_loss'].item(), global_step)
                writer.add_scalar('train/vae_recon_loss', losses['vae_recon_loss'].item(), global_step)
                writer.add_scalar('train/vae_kl_loss', losses['vae_kl_loss'].item(), global_step)
                if losses['diffusion_recon_loss'].item() > 0:
                    writer.add_scalar('train/diffusion_recon_loss', losses['diffusion_recon_loss'].item(), global_step)
                if losses['style_kl_loss'] is not None:
                    writer.add_scalar('train/style_kl_loss', losses['style_kl_loss'].item(), global_step)
            writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
    
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
        'diffusion_recon_loss': avg_diffusion_recon_loss
    }


def validate(
    model: StyleEmbeddingDiffusionModel,
    dataloader: DataLoader,
    config: Config,
    epoch: int,
    writer: SummaryWriter,
    use_ddp: bool = False,
    is_main_process: bool = True
):
    """验证"""
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
    
    # 只在主进程记录到tensorboard
    if is_main_process and writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
    
    return avg_loss


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
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)
    
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
    
    # DDP包装模型
    if use_ddp:
        # 修复VAE训练后，所有参数都参与损失计算，无需find_unused_parameters
        # 设置为False可以提升性能（避免每次迭代额外遍历autograd图）
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        if is_main_process:
            print(f"\n✓ 使用DDP在 {args.world_size} 张GPU上训练")
            print(f"  每张GPU的batch_size: {per_gpu_batch_size}")
            print(f"  总有效batch_size: {per_gpu_batch_size * args.world_size}")
    
    # 创建优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 计算总训练步数（用于学习率调度）
    total_steps = len(train_loader) * config.training.num_epochs
    
    # 创建学习率调度器
    scheduler_type = config.training.lr_scheduler.lower()
    if scheduler_type == "constant":
        # 简单的warmup后保持恒定学习率（原始策略）
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step / config.training.warmup_steps, 1.0)
        )
    elif scheduler_type == "cosine":
        # Warmup + Cosine Annealing（推荐）
        def lr_lambda(step):
            if step < config.training.warmup_steps:
                # Warmup阶段：线性增长
                return step / config.training.warmup_steps
            else:
                # Cosine decay阶段
                progress = (step - config.training.warmup_steps) / (total_steps - config.training.warmup_steps)
                min_lr_ratio = config.training.min_lr / config.training.learning_rate
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == "linear":
        # Warmup + Linear Decay
        def lr_lambda(step):
            if step < config.training.warmup_steps:
                # Warmup阶段：线性增长
                return step / config.training.warmup_steps
            else:
                # Linear decay阶段
                progress = (step - config.training.warmup_steps) / (total_steps - config.training.warmup_steps)
                min_lr_ratio = config.training.min_lr / config.training.learning_rate
                return max(min_lr_ratio, 1.0 - progress)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_type}. Choose from 'constant', 'cosine', 'linear'")
    
    if is_main_process:
        print(f"✓ 学习率调度策略: {scheduler_type}")
        print(f"  初始学习率: {config.training.learning_rate}")
        print(f"  Warmup步数: {config.training.warmup_steps}")
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
    
    # TensorBoard（只在主进程）
    writer = SummaryWriter(config.training.log_dir) if is_main_process else None
    
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
    for epoch in range(start_epoch, config.training.num_epochs):
        # DDP时设置sampler的epoch（确保每个epoch的数据顺序不同）
        if use_ddp:
            train_sampler.set_epoch(epoch)
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch, writer, scaler, is_main_process
        )
        
        # 验证
        val_loss = validate(model, val_loader, config, epoch, writer, use_ddp, is_main_process)
        
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
        
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
    
    # 清理
    if is_main_process:
        if writer is not None:
            writer.close()
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
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='使用数据的比例（0.0-1.0），例如0.01表示使用1%%的数据')
    
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
            if args.checkpoint_dir and args.checkpoint_dir != 'checkpoints':
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

