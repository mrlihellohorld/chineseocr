#!/usr/bin/env python3
"""
ARM64 PyTorch 功能演示脚本
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io

def create_sample_image():
    """创建一个示例图像"""
    # 创建一个简单的渐变图像
    width, height = 224, 224
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            image_array[y, x] = [
                int(255 * x / width),      # Red gradient
                int(255 * y / height),     # Green gradient  
                int(255 * (x + y) / (width + height))  # Blue gradient
            ]
    
    return Image.fromarray(image_array)

def demo_pytorch_features():
    """演示PyTorch的各种功能"""
    print("🚀 ARM64 PyTorch 功能演示")
    print("=" * 50)
    
    # 1. 基本张量操作
    print("1. 基本张量操作")
    print("-" * 30)
    
    # 创建张量
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    print(f"张量 x 形状: {x.shape}")
    print(f"张量 y 形状: {y.shape}")
    
    # 矩阵乘法
    z = torch.mm(x, y)
    print(f"矩阵乘法结果形状: {z.shape}")
    
    # 2. 图像处理
    print("\n2. 图像处理和变换")
    print("-" * 30)
    
    # 创建示例图像
    img = create_sample_image()
    print(f"原始图像大小: {img.size}")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 应用变换
    img_tensor = transform(img)
    print(f"变换后张量形状: {img_tensor.shape}")
    print(f"张量数据类型: {img_tensor.dtype}")
    
    # 3. 简单的神经网络
    print("\n3. 简单神经网络")
    print("-" * 30)
    
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = torch.nn.Linear(224*224*3, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 10)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x):
            x = x.view(x.size(0), -1)  # 展平
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 创建网络
    net = SimpleNet()
    print(f"网络参数数量: {sum(p.numel() for p in net.parameters())}")
    
    # 前向传播
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = net(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 4. 损失函数和优化器
    print("\n4. 损失函数和优化器")
    print("-" * 30)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 模拟标签
    labels = torch.randint(0, 10, (batch_size,))
    
    # 计算损失
    loss = criterion(output, labels)
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("完成一次反向传播和参数更新")
    
    # 5. GPU测试（如果可用）
    print("\n5. GPU加速测试")
    print("-" * 30)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"✓ GPU设备数量: {torch.cuda.device_count()}")
        print(f"✓ 当前GPU: {torch.cuda.get_device_name()}")
        
        # 将张量移到GPU
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = torch.mm(x_gpu, y_gpu)
        print(f"✓ GPU矩阵乘法完成，结果形状: {z_gpu.shape}")
        
        # 将网络移到GPU
        net_gpu = net.cuda()
        input_gpu = input_tensor.cuda()
        output_gpu = net_gpu(input_gpu)
        print(f"✓ GPU推理完成，输出形状: {output_gpu.shape}")
        
    else:
        print("⚠ CUDA不可用，使用CPU计算")
        print("  这在ARM64架构上是正常的")
    
    # 6. 保存和加载模型
    print("\n6. 模型保存和加载")
    print("-" * 30)
    
    # 保存模型
    model_path = "/tmp/simple_net.pth"
    torch.save(net.state_dict(), model_path)
    print(f"✓ 模型已保存到: {model_path}")
    
    # 加载模型
    new_net = SimpleNet()
    new_net.load_state_dict(torch.load(model_path))
    print("✓ 模型加载成功")
    
    # 验证模型一致性
    with torch.no_grad():
        output1 = net(input_tensor)
        output2 = new_net(input_tensor)
        diff = torch.max(torch.abs(output1 - output2))
        print(f"✓ 模型输出差异: {diff.item():.6f} (应该接近0)")
    
    print("\n" + "=" * 50)
    print("🎉 ARM64 PyTorch 演示完成!")
    print("所有功能正常工作!")
    print("=" * 50)

if __name__ == "__main__":
    demo_pytorch_features()