# 基于条件扩散模型的成像风格表征学习系统

## 项目概述

本项目构建了一个基于条件扩散模型的成像风格表征学习系统，用于学习一个连续、抽象、无预定义类别的成像风格表示空间（style embedding）。

系统以二维码图像为研究对象，利用二维码内容高度结构化、可精确表达的特性，实现**内容（content）与成像风格（imaging style）的解耦建模**。

## 系统架构

系统由四个主要模块组成：

1. **内容编码器（Enc_content）**：编码二维码的空间结构与模块布局
2. **成像风格编码器（Enc_style）**：学习连续、抽象的成像风格表征（核心模块）
3. **条件潜空间扩散生成器**：基于VAE + UNet的latent diffusion架构
4. **VAE模块**：图像与latent空间之间的编码解码

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

### 数据整理

如果原始数据文件在同一目录下，且使用 `_gt` 和 `_real` 后缀区分 bin 和 orig 文件，可以使用数据整理脚本：

```bash
# 使用符号链接（推荐，节省空间）
python organize_data.py \
    --source_dir /mnt/d/Archived/barcode-paired-binary/dm_neg/ \
    --target_dir data \
    --train_ratio 0.8 \
    --symlink

# 或复制文件（如果需要在不同位置使用）
python organize_data.py \
    --source_dir /mnt/d/Archived/barcode-paired-binary/dm_neg/ \
    --target_dir data \
    --train_ratio 0.8
```

### 数据目录结构

整理后的数据目录结构应为：
```
data/
├── train/
│   ├── bin/          # 二值二维码图像
│   └── orig/         # 真实成像图像
└── val/
    ├── bin/
    └── orig/
```

每个样本由一对图像组成：
- `bin/`: 理想的二值二维码图像（I_bin），文件名格式：`{id}_gt.png`
- `orig/`: 真实场景中拍摄的二维码图像（I_orig），文件名格式：`{id}_real.png`

如果数据目录不存在，训练脚本会自动创建虚拟数据用于测试。

## 使用方法

### 1. 测试模型架构

首先测试模型是否正确构建：

```bash
python test_model.py
```

### 2. 训练模型

```bash
python train.py --data_root data --checkpoint_dir checkpoints
```

训练参数可以通过修改 `config.py` 中的配置进行调整。

### 3. 推理生成

使用训练好的模型进行风格迁移生成：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --bin_image path/to/qr_bin.png \
    --style_ref path/to/style_reference.png \
    --output output.png \
    --num_steps 50 \
    --use_ddim
```

### 4. 分析风格空间

分析训练好的风格表征空间：

```bash
python analyze_style_space.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root data \
    --max_samples 1000
```

## 项目结构

```
style_embedding_diffusion/
├── models/
│   ├── __init__.py
│   ├── content_encoder.py      # 内容编码器
│   ├── style_encoder.py        # 风格编码器（核心）
│   ├── vae.py                  # VAE模块
│   └── diffusion_model.py      # 扩散模型
├── utils/
│   ├── __init__.py
│   ├── dataset.py              # 数据加载
│   ├── scheduler.py            # 扩散调度器
│   └── visualization.py        # 可视化工具
├── config.py                   # 配置文件
├── train.py                    # 训练脚本
├── inference.py                # 推理脚本
├── analyze_style_space.py      # 风格空间分析
├── test_model.py               # 模型测试脚本
├── organize_data.py            # 数据整理脚本
├── requirements.txt            # 依赖包
└── README.md
```

## 配置说明

主要配置在 `config.py` 中：

- **模型配置**：编码器维度、风格嵌入维度、VAE参数、UNet参数等
- **训练配置**：批次大小、学习率、训练轮数等
- **扩散配置**：时间步数、噪声调度方式等

## 训练流程

1. **VAE编码**：将真实图像编码到latent空间
2. **加噪**：随机采样时间步并添加噪声
3. **条件提取**：
   - 从I_bin提取内容特征（F_content）
   - 从I_orig提取风格向量（z_style）
4. **条件扩散**：UNet在内容和风格条件下预测噪声
5. **优化**：最小化预测噪声与真实噪声的MSE损失

## 推理流程

1. **风格提取**：从风格参考图像提取z_style
2. **内容编码**：从内容图像提取F_content
3. **条件采样**：从纯噪声开始，在内容和风格条件下进行扩散采样
4. **VAE解码**：将生成的latent解码为图像

## 研究关注点

- **连续性**：成像风格是否在z_style空间中形成连续、可解释的流形
- **聚类性**：相似成像条件的样本，其z_style是否聚集
- **可插值性**：不同成像条件是否在latent空间中呈现可插值、可组合关系
- **解耦性**：风格表征是否与内容变化解耦

## 技术特点

- **条件注入**：
  - 内容条件：ControlNet风格的多尺度特征注入
  - 风格条件：AdaIN/FiLM全局调制
- **风格编码**：VAE结构确保风格向量的正则化
- **高效采样**：支持DDPM和DDIM两种采样方式

