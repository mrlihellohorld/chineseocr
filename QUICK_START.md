# ARM64 PyTorch 容器 - 快速开始

## 📋 使用步骤

### 1. 准备wheel文件
将以下文件放在项目根目录：
```
torch-2.3.0-cp310-cp310-linux_aarch64.whl
torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl  
torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

### 2. 构建容器
```bash
./build-arm64.sh
```

### 3. 运行容器
```bash
# 基本运行
docker run -p 8080:8080 chineseocr-arm64:latest

# 交互式运行
docker run -it -p 8080:8080 chineseocr-arm64:latest bash

# 带GPU支持（如果可用）
docker run --gpus all -p 8080:8080 chineseocr-arm64:latest
```

### 4. 验证安装
```bash
# 在容器内运行
python3 test_pytorch.py
python3 demo.py
```

## 🗂️ 文件说明

| 文件 | 描述 |
|------|------|
| `Dockerfile.arm64` | ARM64专用Dockerfile |
| `build-arm64.sh` | 自动化构建脚本 |
| `test_pytorch.py` | PyTorch安装验证 |
| `demo.py` | 功能演示脚本 |
| `ARM64_PYTORCH_SETUP.md` | 详细安装指南 |

## ⚡ 快速测试

```bash
# 检查架构
docker run chineseocr-arm64:latest uname -m

# 检查PyTorch版本
docker run chineseocr-arm64:latest python3 -c "import torch; print(torch.__version__)"

# 运行完整测试
docker run chineseocr-arm64:latest python3 test_pytorch.py
```

## 🔧 自定义配置

如需修改配置，编辑 `Dockerfile.arm64` 中的相应部分：
- Python版本：修改 `python3.10` 相关行
- 依赖包：修改 `pip3 install` 命令
- 基础镜像：修改 `FROM` 行

## 📝 注意事项

- 确保在aarch64架构上运行
- wheel文件必须在项目根目录
- 容器已优化使用国内镜像源
- 支持CUDA但在ARM64上主要使用CPU