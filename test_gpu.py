import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    gpu_id = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB")
    print(f"Reserved Memory: {torch.cuda.memory_reserved(gpu_id) / 1e9:.2f} GB")
    print(f"Free Memory (Approx.): {(torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)) / 1e9:.2f} GB")
else:
    print("CUDA is not available. Using CPU.")
