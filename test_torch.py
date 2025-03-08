import torch
print(torch.backends.cudnn.enabled)  # Should print True
print(torch.backends.cudnn.version())  # Should match cuDNN Version
print(torch.cuda.get_device_name(0))  # Should show "RTX 4070 Ti"
