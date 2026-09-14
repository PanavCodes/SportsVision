import torch
import sys

def verify_cuda():
    """
    Verifies CUDA is available and prints diagnostic info.
    """
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA is not available. Ensure NVIDIA drivers and PyTorch with CUDA are installed.")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"CUDA Available: True")
    print(f"Device Count: {device_count}")
    print(f"Current Device: {current_device} ({device_name})")
    
    # Optional: check if it's the RTX 4060
    if "4060" not in device_name:
        print("WARNING: Expected RTX 4060, but found another GPU.")
        
    return True
