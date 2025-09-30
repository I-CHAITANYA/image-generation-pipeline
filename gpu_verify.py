# verify_pytorch.py
import torch
import sys

print("🧪 PyTorch Verification")
print("=" * 40)

# Basic info
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# CUDA info
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # Test GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {gpu_memory:.1f} GB")
else:
    print("❌ CUDA not available - will use CPU")

# Test basic tensor operations
print("\n🧠 Testing tensor operations...")
x = torch.rand(3, 3)
if torch.cuda.is_available():
    x = x.cuda()
    print("✅ GPU tensor operations working")
else:
    print("ℹ️  Using CPU for tensor operations")

print("✅ PyTorch verification complete!")