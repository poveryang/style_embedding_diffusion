"""
Weights & Biases (wandb) 日志记录工具
可以轻松集成到现有训练代码中，实现实验管理和对比
"""
import wandb
from typing import Optional, Dict, List, Any
from config import Config


class WandbLogger:
    """wandb 日志记录器封装类"""
    
    def __init__(self, config: Config, project_name: str = "style_embedding_diffusion",
                 run_name: Optional[str] = None, tags: Optional[List[str]] = None,
                 enabled: bool = True, is_main_process: bool = True):
        """
        初始化 wandb logger
        
        Args:
            config: 训练配置对象
            project_name: wandb 项目名称
            run_name: 实验运行名称（如果为None，会自动生成）
            tags: 实验标签列表
            enabled: 是否启用 wandb
            is_main_process: 是否为主进程（DDP训练时使用）
        """
        self.enabled = enabled and is_main_process
        self.is_main_process = is_main_process
        
        if not self.enabled:
            return
        
        # 构建配置字典
        config_dict = self._config_to_dict(config)
        
        # 生成运行名称
        if run_name is None:
            run_name = self._generate_run_name(config)
        
        # 初始化 wandb
        try:
            wandb.init(
                project=project_name,
                name=run_name,
                tags=tags or [],
                config=config_dict,
                dir=config.training.log_dir,
            )
        except Exception as e:
            print(f"警告: wandb 初始化失败: {e}")
            print("提示: 运行 'wandb login' 进行登录，或设置 enabled=False 禁用 wandb")
            self.enabled = False
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        return {
            "model": {
                "content_encoder_dim": config.model.content_encoder_dim,
                "content_encoder_layers": config.model.content_encoder_layers,
                "style_embedding_dim": config.model.style_embedding_dim,
                "style_encoder_base_dim": config.model.style_encoder_base_dim,
                "style_encoder_layers": config.model.style_encoder_layers,
                "style_kl_weight": config.model.style_kl_weight,
                "vae_latent_channels": config.model.vae_latent_channels,
                "vae_scale_factor": config.model.vae_scale_factor,
                "vae_recon_weight": config.model.vae_recon_weight,
                "vae_kl_weight": config.model.vae_kl_weight,
                "unet_model_channels": config.model.unet_model_channels,
                "unet_dropout": config.model.unet_dropout,
                "content_condition_type": config.model.content_condition_type,
                "style_condition_type": config.model.style_condition_type,
                "num_timesteps": config.model.num_timesteps,
                "beta_schedule": config.model.beta_schedule,
            },
            "training": {
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "optimizer": config.training.optimizer,
                "weight_decay": config.training.weight_decay,
                "lr_scheduler": config.training.lr_scheduler,
                "min_lr": config.training.min_lr,
                "warmup_epochs": config.training.warmup_epochs,
                "use_amp": config.training.use_amp,
            },
            "data": {
                "image_size": config.training.image_size,
                "data_root": config.training.data_root,
            }
        }
    
    def _generate_run_name(self, config: Config) -> str:
        """生成运行名称"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        return f"lr{config.training.learning_rate}_bs{config.training.batch_size}_{timestamp}"
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, commit: bool = True):
        """
        记录指标
        
        Args:
            metrics: 指标字典，例如 {'train/loss': 0.5, 'train/acc': 0.9}
            step: 步数（可选）
            commit: 是否立即提交（默认True）
        """
        if self.enabled and wandb.run is not None:
            wandb.log(metrics, step=step, commit=commit)
    
    def log_images(self, images: List, key: str = "samples", step: Optional[int] = None, 
                   captions: Optional[List[str]] = None, use_time_series: bool = True):
        """
        记录图像到 wandb（时间序列模式，便于在时间轴上按epoch显示）
        
        Args:
            images: 图像列表（PIL Image, numpy array, 或 torch.Tensor）
            key: 图像在 wandb 中的键名（基础键名）
            step: 步数（epoch），用于在时间轴上显示
            captions: 图像标题列表（可选）
            use_time_series: 是否使用时间序列模式（每个sample使用独立的key，便于在时间轴上按epoch显示）
        
        注意：当前 train.py 中直接使用 log_metrics 记录图像，此函数保留作为备用API
        """
        if not self.enabled or wandb.run is None:
            return
        
        if use_time_series and step is not None:
            # 时间序列模式：每个sample使用独立的key，使用epoch作为step
            # 这样在wandb的Media面板中，每个sample会形成一条时间序列，x轴按epoch显示
            for idx, img in enumerate(images):
                caption = captions[idx] if captions else f"Epoch {step}, Sample {idx}"
                # 使用统一的key格式：samples/sample_{idx:02d}
                # 使用epoch作为step，图像会在时间轴上按epoch顺序显示
                image_key = f"{key}/sample_{idx:02d}"
                wandb.log({image_key: wandb.Image(img, caption=caption)}, step=step)
        else:
            # 传统方式：所有图像在一个key下
            wandb_images = []
            for idx, img in enumerate(images):
                caption = captions[idx] if captions else None
                wandb_images.append(wandb.Image(img, caption=caption))
            wandb.log({key: wandb_images}, step=step)
    
    def update_summary(self, metrics: Dict[str, Any]):
        """
        更新运行摘要（通常用于记录最佳值）
        
        Args:
            metrics: 摘要指标字典
        """
        if self.enabled and wandb.run is not None:
            wandb.run.summary.update(metrics)
    
    def log_model_checkpoint(self, checkpoint_path: str, artifact_name: str = "model", 
                            artifact_type: str = "checkpoint", metadata: Optional[Dict[str, Any]] = None):
        """
        将模型检查点保存到 wandb artifacts
        
        Args:
            checkpoint_path: 检查点文件路径
            artifact_name: artifact 名称（默认: "model"）
            artifact_type: artifact 类型（"checkpoint" 或 "best_model"）
            metadata: 额外的元数据字典（例如 epoch, val_loss 等）
        """
        if not self.enabled or wandb.run is None:
            return
        
        try:
            import os
            if not os.path.exists(checkpoint_path):
                print(f"警告: 检查点文件不存在: {checkpoint_path}")
                return
            
            # 创建 artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {}
            )
            
            # 添加文件到 artifact
            artifact.add_file(checkpoint_path)
            
            # 记录 artifact
            wandb.log_artifact(artifact)
            
            if self.is_main_process:
                print(f"✓ 已保存模型到 wandb: {artifact_name} ({artifact_type})")
        except Exception as e:
            print(f"警告: 保存模型到 wandb 失败: {e}")
    
    def finish(self):
        """完成 wandb 运行"""
        if self.enabled and wandb.run is not None:
            wandb.finish()

