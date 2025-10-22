#!/bin/bash

# ARM64 PyTorch 容器构建和运行脚本

set -e

echo "开始构建 ARM64 PyTorch 容器..."

# 检查必要的wheel文件是否存在
TORCH_WHEEL="torch-2.3.0-cp310-cp310-linux_aarch64.whl"
TORCHAUDIO_WHEEL="torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHEEL="torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl"

echo "检查PyTorch wheel文件..."
if [ ! -f "$TORCH_WHEEL" ]; then
    echo "错误: 找不到 $TORCH_WHEEL"
    echo "请确保该文件在当前目录中"
    exit 1
fi

if [ ! -f "$TORCHAUDIO_WHEEL" ]; then
    echo "错误: 找不到 $TORCHAUDIO_WHEEL"
    echo "请确保该文件在当前目录中"
    exit 1
fi

if [ ! -f "$TORCHVISION_WHEEL" ]; then
    echo "错误: 找不到 $TORCHVISION_WHEEL"
    echo "请确保该文件在当前目录中"
    exit 1
fi

echo "所有wheel文件检查完成 ✓"

# 构建镜像
IMAGE_NAME="chineseocr-arm64"
TAG="latest"

echo "构建Docker镜像: $IMAGE_NAME:$TAG"
docker build -f Dockerfile.arm64 -t $IMAGE_NAME:$TAG .

if [ $? -eq 0 ]; then
    echo "✓ 镜像构建成功!"
    echo ""
    echo "您可以使用以下命令运行容器:"
    echo "docker run -p 8080:8080 $IMAGE_NAME:$TAG"
    echo ""
    echo "或者使用GPU支持（如果可用）:"
    echo "docker run --gpus all -p 8080:8080 $IMAGE_NAME:$TAG"
    echo ""
    echo "交互式运行容器:"
    echo "docker run -it -p 8080:8080 $IMAGE_NAME:$TAG bash"
else
    echo "✗ 镜像构建失败!"
    exit 1
fi