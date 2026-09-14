import os
import sys
import urllib.request
import torch

# Initialize Config to set environment variables and directory paths
import config

print("=" * 65)
print(" SportsVision Multi-Sport: Model Verification & Setup")
print("=" * 65)
print(f" Workspace Root: {config.WORKSPACE_ROOT}")
print(f" Compute Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (Fallback Mode)'}")
print("=" * 65)

os.makedirs(config.SHARED_MODELS_DIR, exist_ok=True)
os.makedirs(config.BASKETBALL_MODELS_DIR, exist_ok=True)
os.makedirs(config.CRICKET_MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Shared Models (Player Detector & Caches)
# -------------------------------------------------------------
print("\n[1/3] Verifying Shared Base Models...")

# 1.1 YOLOv8m (Shared Player/Person Detector)
yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
yolo_path = config.PLAYER_MODEL_PATH
if os.path.exists(yolo_path):
    size_mb = os.path.getsize(yolo_path) / (1024 * 1024)
    print(f"  [OK] YOLOv8m verified: {yolo_path} ({size_mb:.1f} MB)")
else:
    print(f"  Downloading YOLOv8m to {yolo_path}...")
    try:
        urllib.request.urlretrieve(yolo_url, yolo_path)
        print("  [OK] YOLOv8m downloaded successfully.")
    except Exception as e:
        print(f"  [FAIL] Failed to download YOLOv8m: {e}")

# 1.2 SAM2.1 Small (Player Segmentation)
sam2_path = config.SAM_MODEL_PATH
if not os.path.exists(sam2_path):
    print(f"  Downloading Ultralytics SAM2.1 Small to {sam2_path}...")
    try:
        from ultralytics import SAM
        SAM("sam2.1_s.pt")
        downloaded_path = os.path.abspath("sam2.1_s.pt")
        if os.path.exists(downloaded_path):
            os.replace(downloaded_path, sam2_path)
            print("  [OK] SAM2.1 Small moved to models/shared/ successfully.")
        else:
            print("  [WARN] SAM2.1 checkpoint was not found at root.")
    except Exception as e:
        print(f"  [FAIL] Failed to download SAM2.1: {e}")
else:
    size_mb = os.path.getsize(sam2_path) / (1024 * 1024)
    print(f"  [OK] SAM2.1 Small verified: {sam2_path} ({size_mb:.1f} MB)")

# Clean up any accidental root downloads
for stray in ["yolov8m.pt", "sam2.1_s.pt"]:
    root_stray = os.path.join(config.WORKSPACE_ROOT, stray)
    if os.path.exists(root_stray):
        try:
            os.remove(root_stray)
        except OSError:
            pass

# 1.3 SigLIP (Sport Router & Zero-Shot Vision)
print("  Downloading/Verifying SigLIP (Sport Router)...")
try:
    from transformers import AutoModel, AutoProcessor
    AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    AutoModel.from_pretrained("google/siglip-base-patch16-224")
    print("  [OK] SigLIP cached successfully.")
except Exception as e:
    print(f"  [WARN] SigLIP check warning: {e}")

# 1.4 EasyOCR (Player ID Jersey Number OCR)
if config.USE_OCR:
    print("  Downloading/Verifying EasyOCR weights...")
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("  [OK] EasyOCR models verified.")
    except Exception as e:
        print(f"  [WARN] EasyOCR check warning: {e}")
else:
    print("  [-] EasyOCR: Disabled in config.py.")

# -------------------------------------------------------------
# 2. Basketball Models
# -------------------------------------------------------------
print("\n[2/3] Verifying Basketball Pipeline Models...")
bball_weights = config.BASKETBALL_BALL_MODEL_PATH
court_weights = config.BASKETBALL_COURT_MODEL_PATH

if os.path.exists(bball_weights):
    size_mb = os.path.getsize(bball_weights) / (1024 * 1024)
    print(f"  [OK] Basketball ball model verified: {bball_weights} ({size_mb:.1f} MB)")
else:
    print(f"  [WARN] Basketball ball model missing at {bball_weights}")

if os.path.exists(court_weights):
    size_mb = os.path.getsize(court_weights) / (1024 * 1024)
    print(f"  [OK] Basketball court keypoint model verified: {court_weights} ({size_mb:.1f} MB)")
else:
    print(f"  [WARN] Basketball court keypoint model missing at {court_weights}")

# -------------------------------------------------------------
# 3. Cricket Models
# -------------------------------------------------------------
print("\n[3/3] Verifying Cricket Pipeline Models...")
cricket_ball = config.CRICKET_BALL_MODEL_PATH
cricket_cbd = os.path.join(config.CRICKET_MODELS_DIR, "CBDbest.pt")

if os.path.exists(cricket_cbd):
    size_mb = os.path.getsize(cricket_cbd) / (1024 * 1024)
    print(f"  [OK] Cricket ball detector (CBD) verified: {cricket_cbd} ({size_mb:.1f} MB)")

if os.path.exists(cricket_ball):
    size_mb = os.path.getsize(cricket_ball) / (1024 * 1024)
    print(f"  [OK] Cricket ball model verified: {cricket_ball} ({size_mb:.1f} MB)")
elif not os.path.exists(cricket_cbd):
    print(f"  [WARN] No cricket ball model found at {cricket_ball} or {cricket_cbd}")

print("\n" + "=" * 65)
print(" SportsVision Model Setup Complete!")
print("=" * 65)
