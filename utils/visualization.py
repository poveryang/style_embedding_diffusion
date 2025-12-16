"""
可视化工具：用于分析和可视化风格表征空间
"""
import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("警告: matplotlib 或 sklearn 未安装，可视化功能不可用")
from typing import List, Optional
import os


def visualize_style_embeddings(
    style_embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    method: str = "tsne",
    title: str = "风格表征空间可视化"
):
    """
    可视化风格嵌入空间
    
    Args:
        style_embeddings: (N, style_dim) 风格嵌入向量
        labels: 可选的标签列表
        save_path: 保存路径
        method: "tsne" 或 "pca"
        title: 图表标题
    """
    if not HAS_VIS:
        print("可视化功能不可用，请安装 matplotlib 和 sklearn")
        return
    
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(style_embeddings) - 1))
        embeddings_2d = reducer.fit_transform(style_embeddings)
    elif method == "pca":
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(style_embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(embeddings_2d)), cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def interpolate_styles(
    z_style1: torch.Tensor,
    z_style2: torch.Tensor,
    num_steps: int = 10
) -> torch.Tensor:
    """
    在风格空间中插值
    
    Args:
        z_style1: (style_dim,) 风格向量1
        z_style2: (style_dim,) 风格向量2
        num_steps: 插值步数
        
    Returns:
        interpolated: (num_steps, style_dim) 插值后的风格向量
    """
    alphas = torch.linspace(0, 1, num_steps, device=z_style1.device)
    interpolated = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z_style1 + alpha * z_style2
        interpolated.append(z_interp)
    return torch.stack(interpolated)


def analyze_style_space(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    max_samples: int = 1000
):
    """
    分析风格表征空间
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最大样本数
    """
    model.eval()
    style_embeddings = []
    
    with torch.no_grad():
        count = 0
        for I_bin, I_orig in dataloader:
            if count >= max_samples:
                break
            
            I_orig = I_orig.to(device)
            z_style = model.style_encoder.encode(I_orig)
            style_embeddings.append(z_style.cpu().numpy())
            count += I_orig.size(0)
    
    style_embeddings = np.concatenate(style_embeddings, axis=0)
    
    # 统计分析
    print(f"风格嵌入统计:")
    print(f"  形状: {style_embeddings.shape}")
    print(f"  均值: {style_embeddings.mean(axis=0)}")
    print(f"  标准差: {style_embeddings.std(axis=0)}")
    print(f"  最小值: {style_embeddings.min(axis=0)}")
    print(f"  最大值: {style_embeddings.max(axis=0)}")
    
    # 可视化
    visualize_style_embeddings(
        style_embeddings,
        save_path="style_space_tsne.png",
        method="tsne",
        title="风格表征空间 (t-SNE)"
    )
    
    visualize_style_embeddings(
        style_embeddings,
        save_path="style_space_pca.png",
        method="pca",
        title="风格表征空间 (PCA)"
    )
    
    return style_embeddings

