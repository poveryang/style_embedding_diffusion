"""
推理脚本：使用训练好的模型进行风格迁移生成
"""
import torch
import torch.nn as nn
import argparse
import os
from PIL import Image
import numpy as np
from torchvision import transforms

from config import Config
from models import ContentEncoder, StyleEncoder, VAE, ConditionalUNet
from utils import DiffusionScheduler


class StyleEmbeddingDiffusionModel(nn.Module):
    """完整的风格嵌入扩散模型系统（推理版本）"""
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
            scale_factor=model_cfg.vae_scale_factor
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
    
    @torch.no_grad()
    def generate(
        self,
        I_bin: torch.Tensor,
        I_style_ref: torch.Tensor,
        num_inference_steps: int = 50,
        use_ddim: bool = True,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        生成图像
        
        Args:
            I_bin: (1, 1, H, W) 二值二维码图像
            I_style_ref: (1, 1, H, W) 风格参考图像
            num_inference_steps: 推理步数
            use_ddim: 是否使用DDIM采样
            eta: DDIM参数
            
        Returns:
            I_gen: (1, 1, H, W) 生成的图像
        """
        self.eval()
        
        # 1. 提取风格表征
        z_style = self.style_encoder.encode(I_style_ref)
        
        # 2. 编码内容结构
        F_content = self.content_encoder(I_bin)
        
        # 3. 从纯噪声开始条件扩散采样
        z_T = torch.randn(
            (1, self.config.model.vae_latent_channels,
             I_bin.shape[2] // self.config.model.vae_scale_factor,
             I_bin.shape[3] // self.config.model.vae_scale_factor),
            device=I_bin.device
        )
        
        # 时间步序列
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps,
            device=I_bin.device
        ).long()
        
        z_t = z_T
        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)
            
            # 预测噪声
            eps_pred = self.unet(z_t, t_batch, content_features=F_content, style_emb=z_style)
            
            # 去噪步骤
            if use_ddim:
                prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=I_bin.device)
                z_t = self.scheduler.ddim_step(eps_pred, t_batch, z_t, eta, prev_t.unsqueeze(0))
            else:
                z_t = self.scheduler.step(eps_pred, t_batch, z_t)
        
        # 4. 解码生成图像
        I_gen = self.vae.decode(z_t)
        
        return I_gen


def load_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
    ])
    
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    return image


def save_image(tensor: torch.Tensor, save_path: str):
    """保存图像"""
    # 反归一化
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为numpy并保存
    image = tensor.squeeze(0).squeeze(0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(save_path)


def main():
    parser = argparse.ArgumentParser(description='推理：生成风格化的二维码图像')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--bin_image', type=str, required=True, help='二值二维码图像路径')
    parser.add_argument('--style_ref', type=str, required=True, help='风格参考图像路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像路径')
    parser.add_argument('--num_steps', type=int, default=50, help='推理步数')
    parser.add_argument('--use_ddim', action='store_true', help='使用DDIM采样')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM参数')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    config.training.image_size = args.image_size
    
    # 创建模型
    model = StyleEmbeddingDiffusionModel(config).to(config.device)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"已加载模型: {args.checkpoint}")
    
    # 加载图像
    I_bin = load_image(args.bin_image, args.image_size).to(config.device)
    I_style_ref = load_image(args.style_ref, args.image_size).to(config.device)
    
    print("开始生成...")
    # 生成图像
    I_gen = model.generate(
        I_bin,
        I_style_ref,
        num_inference_steps=args.num_steps,
        use_ddim=args.use_ddim,
        eta=args.eta
    )
    
    # 保存结果
    save_image(I_gen, args.output)
    print(f"生成完成，已保存到: {args.output}")


if __name__ == '__main__':
    main()

