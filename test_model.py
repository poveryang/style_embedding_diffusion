"""
快速测试脚本：测试模型架构是否正确
"""
import torch
from config import Config
from models import ContentEncoder, StyleEncoder, VAE, ConditionalUNet


def test_models():
    """测试各个模型模块"""
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    image_size = 256
    
    print("=" * 50)
    print("测试模型架构")
    print("=" * 50)
    
    # 创建测试数据
    I_bin = torch.randn(batch_size, 1, image_size, image_size).to(device)
    I_orig = torch.randn(batch_size, 1, image_size, image_size).to(device)
    
    print(f"\n输入形状: I_bin={I_bin.shape}, I_orig={I_orig.shape}")
    
    # 测试内容编码器
    print("\n1. 测试内容编码器...")
    content_encoder = ContentEncoder(
        in_channels=1,
        base_dim=config.model.content_encoder_dim,
        num_layers=config.model.content_encoder_layers
    ).to(device)
    F_content = content_encoder(I_bin)
    print(f"   输出: {len(F_content)} 个特征图")
    for i, feat in enumerate(F_content):
        print(f"   F_content[{i}]: {feat.shape}")
    
    # 测试风格编码器
    print("\n2. 测试风格编码器...")
    style_encoder = StyleEncoder(
        in_channels=1,
        base_dim=config.model.style_encoder_base_dim,
        num_layers=config.model.style_encoder_layers,
        style_dim=config.model.style_embedding_dim,
        use_vae=True
    ).to(device)
    z_style, kl_loss = style_encoder(I_orig)
    print(f"   z_style: {z_style.shape}")
    print(f"   kl_loss: {kl_loss.item() if kl_loss is not None else None}")
    
    # 测试VAE
    print("\n3. 测试VAE...")
    vae = VAE(
        in_channels=config.model.vae_in_channels,
        out_channels=config.model.vae_out_channels,
        latent_channels=config.model.vae_latent_channels,
        scale_factor=config.model.vae_scale_factor
    ).to(device)
    z0 = vae.encode(I_orig)
    print(f"   z0 (latent): {z0.shape}")
    I_recon, _, vae_kl = vae(I_orig)
    print(f"   I_recon: {I_recon.shape}")
    print(f"   vae_kl_loss: {vae_kl.item()}")
    
    # 测试UNet
    print("\n4. 测试条件UNet...")
    content_feature_dims = [
        config.model.content_encoder_dim,
        config.model.content_encoder_dim * 2,
        config.model.content_encoder_dim * 4,
        config.model.content_encoder_dim * 4
    ]
    unet = ConditionalUNet(
        in_channels=config.model.unet_in_channels,
        out_channels=config.model.unet_out_channels,
        model_channels=config.model.unet_model_channels,
        channel_mult=config.model.unet_channel_mult,
        num_res_blocks=config.model.unet_num_res_blocks,
        attention_resolutions=config.model.unet_attention_resolutions,
        style_dim=config.model.style_embedding_dim,
        style_condition_type=config.model.style_condition_type,
        content_condition_type=config.model.content_condition_type,
        content_feature_dims=content_feature_dims
    ).to(device)
    
    # 模拟扩散过程
    t = torch.randint(0, config.model.num_timesteps, (batch_size,)).to(device)
    noise = torch.randn_like(z0)
    eps_pred = unet(z0, t, content_features=F_content, style_emb=z_style)
    print(f"   eps_pred: {eps_pred.shape}")
    
    # 测试完整前向传播
    print("\n5. 测试完整前向传播...")
    from train import StyleEmbeddingDiffusionModel
    model = StyleEmbeddingDiffusionModel(config).to(device)
    losses = model(I_bin, I_orig)
    print(f"   总损失: {losses['total_loss'].item():.4f}")
    print(f"   扩散损失: {losses['diffusion_loss'].item():.4f}")
    print(f"   风格KL损失: {losses['style_kl_loss'].item():.4f}")
    
    print("\n" + "=" * 50)
    print("所有测试通过！模型架构正确。")
    print("=" * 50)


if __name__ == '__main__':
    test_models()

