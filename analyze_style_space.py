"""
风格空间分析脚本：分析风格表征空间的连续性和可解释性
"""
import torch
import argparse
from torch.utils.data import DataLoader

from config import Config
from models import ContentEncoder, StyleEncoder, VAE, ConditionalUNet
from utils import QRCodeDataset, analyze_style_space


class StyleEmbeddingDiffusionModel(torch.nn.Module):
    """完整的风格嵌入扩散模型系统"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        
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


def main():
    parser = argparse.ArgumentParser(description='分析风格表征空间')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_root', type=str, default='data', help='数据根目录')
    parser.add_argument('--max_samples', type=int, default=1000, help='最大分析样本数')
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    config.training.data_root = args.data_root
    
    # 创建模型
    model = StyleEmbeddingDiffusionModel(config).to(config.device)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"已加载模型: {args.checkpoint}")
    
    # 创建数据集
    dataset = QRCodeDataset(
        data_root=config.training.data_root,
        image_size=config.training.image_size,
        split='val'
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 分析风格空间
    print("开始分析风格表征空间...")
    style_embeddings = analyze_style_space(
        model,
        dataloader,
        device=config.device,
        max_samples=args.max_samples
    )
    
    print("分析完成！")


if __name__ == '__main__':
    main()

