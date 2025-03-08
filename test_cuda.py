import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Please check your installation.")
    
    # Check if CUDA is installed but not detected by PyTorch
    try:
        import ctypes
        cuda = ctypes.CDLL("nvcuda.dll")
        print("CUDA driver is installed but not detected by PyTorch.")
    except:
        print("CUDA driver does not appear to be installed.") 