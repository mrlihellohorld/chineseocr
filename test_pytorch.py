#!/usr/bin/env python3
"""
PyTorch ARM64 安装验证脚本
"""

import sys
import platform

def test_pytorch_installation():
    """测试PyTorch安装"""
    print("=" * 50)
    print("PyTorch ARM64 安装验证")
    print("=" * 50)
    
    # 系统信息
    print(f"Python版本: {sys.version}")
    print(f"平台架构: {platform.machine()}")
    print(f"系统: {platform.system()} {platform.release()}")
    print()
    
    # 测试PyTorch
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"✓ PyTorch 构建信息: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
        
        # 测试基本操作
        x = torch.rand(3, 3)
        print(f"✓ 创建张量成功: {x.shape}")
        
        # 测试CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA可用: {torch.cuda.get_device_name()}")
            print(f"✓ CUDA版本: {torch.version.cuda}")
            # 测试GPU张量
            x_gpu = x.cuda()
            print("✓ GPU张量创建成功")
        else:
            print("⚠ CUDA不可用（这在ARM架构上是正常的）")
            
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    # 测试TorchVision
    try:
        import torchvision
        print(f"✓ TorchVision 版本: {torchvision.__version__}")
        
        # 测试基本功能
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor()])
        print("✓ TorchVision变换功能正常")
        
    except ImportError as e:
        print(f"✗ TorchVision导入失败: {e}")
        return False
    
    # 测试TorchAudio
    try:
        import torchaudio
        print(f"✓ TorchAudio 版本: {torchaudio.__version__}")
    except ImportError as e:
        print(f"✗ TorchAudio导入失败: {e}")
        return False
    
    # 测试基本计算
    try:
        print("\n测试基本计算功能:")
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = a + b
        print(f"✓ 向量加法: {a.tolist()} + {b.tolist()} = {c.tolist()}")
        
        # 矩阵乘法
        m1 = torch.randn(2, 3)
        m2 = torch.randn(3, 2)
        result = torch.mm(m1, m2)
        print(f"✓ 矩阵乘法: {m1.shape} × {m2.shape} = {result.shape}")
        
    except Exception as e:
        print(f"✗ 计算测试失败: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ 所有测试通过! PyTorch ARM64安装成功!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_pytorch_installation()
    sys.exit(0 if success else 1)