import os
import urllib.request
import torch

# Initialize Config to set environment variables
import config

print("="*50)
print("Downloading all required models for offline portability...")
print("="*50)

os.makedirs(os.path.join(config.WORKSPACE_ROOT, "models"), exist_ok=True)

# 1. YOLOv8m (Player Detector)
yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
yolo_path = config.PLAYER_MODEL_PATH
if not os.path.exists(yolo_path):
    print(f"Downloading YOLOv8m to {yolo_path}...")
    urllib.request.urlretrieve(yolo_url, yolo_path)
else:
    print("YOLOv8m already exists.")

# 2. SAM2 (Segmentation)
sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_s.pt"
sam2_path = config.SAM2_MODEL_PATH
if not os.path.exists(sam2_path):
    print(f"Downloading SAM2.1_s to {sam2_path}...")
    urllib.request.urlretrieve(sam2_url, sam2_path)
else:
    print("SAM2 already exists.")

# 3. SigLIP (Team Classification)
print("Downloading/Verifying SigLIP (HuggingFace)...")
from transformers import AutoModel, AutoProcessor
AutoProcessor.from_pretrained(config.SIGLIP_MODEL_ID)
AutoModel.from_pretrained(config.SIGLIP_MODEL_ID)

# 4. EasyOCR (Player ID)
print("Downloading/Verifying EasyOCR...")
import easyocr
# This will download the english model into config.EASYOCR_MODULE_PATH
reader = easyocr.Reader(['en'], gpu=False)

# 5. RF-DETR (Roboflow detection fallback)
print("Downloading/Verifying RF-DETR...")
try:
    from rfdetr import RFDETRMedium
    model = RFDETRMedium() # Triggers download
except ImportError:
    print("rfdetr not installed, skipping.")

print("="*50)
print("All models downloaded and verified!")
print("="*50)
