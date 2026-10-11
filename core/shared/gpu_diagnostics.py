import torch

def verify_cuda():
    """
    Verifies CUDA is available and prints diagnostic info.
    Falls back gracefully to CPU without terminating the process.
    """
    if not torch.cuda.is_available():
        print("[Hardware Diagnostics] CUDA is not available. Operating in CPU fallback mode.")
        return False
        
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"CUDA Available: True")
    print(f"Device Count: {device_count}")
    print(f"Current Device: {current_device} ({device_name})")
    
    vram_gb = round(torch.cuda.get_device_properties(current_device).total_memory / (1024**3), 2)
    print(f"GPU VRAM: {vram_gb} GB")
    return True
