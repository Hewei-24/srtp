import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

# 1. 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Available (显卡加速可用?): {cuda_available}")

# 2. 如果可用，打印显卡名字
if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    x = torch.tensor([1.0, 2.0]).cuda()
    print("CUDA Test Tensor: ", x)
    print("✅ 恭喜！PyTorch 可以识别并使用你的显卡。")
else:
    print("❌ 警告！你的 PyTorch 只能在 CPU 上运行。")
    print("原因可能是：安装了 CPU 版的 torch，或者显卡驱动没装好。")