"""
数据加载器：加载二维码图像对（I_bin, I_orig）
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import torchvision.transforms as transforms


class QRCodeDataset(Dataset):
    """
    二维码数据集
    
    每个样本包含一对图像：
    - I_bin: 理想的二值二维码图像
    - I_orig: 真实成像的二维码图像
    """
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            data_root: 数据根目录
            image_size: 图像尺寸
            split: "train" 或 "val"
            transform: 可选的变换
        """
        self.data_root = data_root
        self.image_size = image_size
        self.split = split
        
        # 数据路径
        self.bin_dir = os.path.join(data_root, split, "bin")
        self.orig_dir = os.path.join(data_root, split, "orig")
        
        # 获取文件列表
        if os.path.exists(self.bin_dir) and os.path.exists(self.orig_dir):
            self.bin_files = sorted([f for f in os.listdir(self.bin_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.orig_files = sorted([f for f in os.listdir(self.orig_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # 确保文件对应
            assert len(self.bin_files) == len(self.orig_files), \
                f"bin和orig文件数量不匹配: {len(self.bin_files)} vs {len(self.orig_files)}"
        else:
            # 如果目录不存在，创建空列表（用于测试）
            self.bin_files = []
            self.orig_files = []
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.bin_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据样本
        
        Returns:
            I_bin: (1, H, W) 二值二维码图像
            I_orig: (1, H, W) 真实成像图像
        """
        if len(self.bin_files) == 0:
            # 返回虚拟数据（用于测试）
            I_bin = torch.randn(1, self.image_size, self.image_size) * 0.5 + 0.5
            I_orig = torch.randn(1, self.image_size, self.image_size) * 0.3 + 0.5
            I_bin = torch.clamp(I_bin, 0, 1) * 2 - 1  # 归一化到[-1, 1]
            I_orig = torch.clamp(I_orig, 0, 1) * 2 - 1
            return I_bin, I_orig
        
        # 加载图像
        bin_path = os.path.join(self.bin_dir, self.bin_files[idx])
        orig_path = os.path.join(self.orig_dir, self.orig_files[idx])
        
        I_bin = Image.open(bin_path).convert('L')  # 转为灰度图
        I_orig = Image.open(orig_path).convert('L')
        
        # 应用变换
        I_bin = self.transform(I_bin)
        I_orig = self.transform(I_orig)
        
        return I_bin, I_orig


def create_dummy_dataset(data_root: str, num_samples: int = 100, image_size: int = 256):
    """
    创建虚拟数据集（用于测试）
    
    Args:
        data_root: 数据根目录
        num_samples: 样本数量
        image_size: 图像尺寸
    """
    import os
    from PIL import Image
    
    # 创建目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(data_root, split, "bin"), exist_ok=True)
        os.makedirs(os.path.join(data_root, split, "orig"), exist_ok=True)
    
    # 生成虚拟数据
    for split in ["train", "val"]:
        n = num_samples if split == "train" else num_samples // 10
        for i in range(n):
            # 生成简单的二值图像（模拟二维码）
            bin_img = np.random.randint(0, 2, (image_size, image_size), dtype=np.uint8) * 255
            # 添加噪声和模糊（模拟真实成像）
            orig_img = bin_img.astype(np.float32)
            orig_img = orig_img + np.random.normal(0, 20, orig_img.shape)
            orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)
            
            # 保存
            Image.fromarray(bin_img).save(
                os.path.join(data_root, split, "bin", f"{i:04d}.png")
            )
            Image.fromarray(orig_img).save(
                os.path.join(data_root, split, "orig", f"{i:04d}.png")
            )
    
    print(f"已创建虚拟数据集: {num_samples} 训练样本, {num_samples // 10} 验证样本")

