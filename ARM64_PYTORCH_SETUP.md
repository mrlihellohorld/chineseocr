# ARM64 PyTorch 容器安装指南

本指南帮助您在aarch64架构的容器中安装您提供的PyTorch wheels。

## 前提条件

1. **系统架构**: aarch64
2. **宿主机CUDA版本**: 12.6
3. **所需的wheel文件**:
   - `torch-2.3.0-cp310-cp310-linux_aarch64.whl`
   - `torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl`  
   - `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`

## 快速开始

### 1. 准备wheel文件
确保所有PyTorch wheel文件都在项目根目录中：

```bash
ls -la *.whl
# 应该看到：
# torch-2.3.0-cp310-cp310-linux_aarch64.whl
# torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
# torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

### 2. 构建容器
使用提供的构建脚本：

```bash
./build-arm64.sh
```

或者手动构建：

```bash
docker build -f Dockerfile.arm64 -t chineseocr-arm64:latest .
```

### 3. 运行容器

#### 基本运行
```bash
docker run -p 8080:8080 chineseocr-arm64:latest
```

#### 交互式运行（用于调试）
```bash
docker run -it -p 8080:8080 chineseocr-arm64:latest bash
```

#### 带GPU支持运行（如果可用）
```bash
docker run --gpus all -p 8080:8080 chineseocr-arm64:latest
```

### 4. 验证安装
在容器内运行验证脚本：

```bash
python3 test_pytorch.py
```

## 容器特性

### 基础镜像
- **Base**: `arm64v8/ubuntu:20.04`
- **Python版本**: Python 3.8+
- **架构**: aarch64

### 安装的包
- **PyTorch**: 2.3.0 (来自您的wheel文件)
- **TorchVision**: 0.18.0a0+6043bc2 (来自您的wheel文件)  
- **TorchAudio**: 2.3.0+952ea74 (来自您的wheel文件)
- **OpenCV**: opencv-contrib-python
- **其他依赖**: 根据requirements.txt安装

### 系统依赖
- 图像处理库 (libgl1-mesa-glx, libglib2.0-0 等)
- 编译工具 (gcc, g++, python3-dev)
- 开发工具 (git, wget, curl)

## 故障排除

### 常见问题

1. **wheel文件找不到**
   ```
   错误: 找不到 torch-2.3.0-cp310-cp310-linux_aarch64.whl
   ```
   **解决**: 确保所有wheel文件在项目根目录

2. **架构不匹配**
   ```
   ERROR: torch-2.3.0-cp310-cp310-linux_aarch64.whl is not supported on this platform
   ```
   **解决**: 确保在aarch64架构上构建容器

3. **Python版本不匹配**
   ```
   ERROR: torch-2.3.0-cp310-cp310-linux_aarch64.whl requires Python 3.10
   ```
   **解决**: 容器会自动安装Python 3.10

### 验证步骤

1. **检查架构**:
   ```bash
   docker run chineseocr-arm64:latest uname -m
   # 应该输出: aarch64
   ```

2. **检查PyTorch**:
   ```bash
   docker run chineseocr-arm64:latest python3 -c "import torch; print(torch.__version__)"
   # 应该输出: 2.3.0
   ```

3. **运行完整测试**:
   ```bash
   docker run chineseocr-arm64:latest python3 test_pytorch.py
   ```

## 注意事项

- 此容器专为aarch64架构优化
- CUDA支持需要宿主机有兼容的GPU驱动
- wheel文件必须与Python版本匹配（cp310 = Python 3.10）
- 容器使用国内镜像源加速包安装

## 文件说明

- `Dockerfile.arm64`: ARM64专用Dockerfile
- `build-arm64.sh`: 自动化构建脚本
- `test_pytorch.py`: PyTorch安装验证脚本
- `ARM64_PYTORCH_SETUP.md`: 本说明文档

## 支持

如果遇到问题，请检查：
1. 系统架构是否为aarch64
2. wheel文件是否完整且在正确位置
3. Docker版本是否支持多架构构建