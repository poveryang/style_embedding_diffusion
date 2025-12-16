# 使用PyTorch官方镜像作为基础镜像
# 支持CUDA，适用于GPU训练
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt /workspace/requirements.txt

# 配置pip使用清华镜像源（加速下载）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . /workspace/

# 设置Python路径
ENV PYTHONPATH=/workspace

# 创建必要的目录
RUN mkdir -p /workspace/checkpoints \
    /workspace/logs \
    /workspace/data

# 默认命令（可以在运行时覆盖）
CMD ["python", "train.py"]

