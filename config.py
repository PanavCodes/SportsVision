import os
import torch

# Base Paths
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECTS_ROOT = os.path.dirname(WORKSPACE_ROOT)

# Extracted Repositories Paths (Migrated into external/)
EXTERNAL_DIR = os.path.join(WORKSPACE_ROOT, "external")
REPO1_DIR = os.path.join(EXTERNAL_DIR, "AI_BasketBall_Analysis_v1-main")
REPO2_DIR = os.path.join(EXTERNAL_DIR, "Basketball-Shot-Detection-main")
REPO3_DIR = os.path.join(EXTERNAL_DIR, "Finetuned-YOLO-and-Supervision-Basketball-Video-Tracker-main")
REPO4_DIR = os.path.join(EXTERNAL_DIR, "Automated-Basketball-Highlights-with-Deep-Learning-main")

# Output and Data
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware Configuration (RTX 4060, 8GB VRAM)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

# --- Dual-Model Detector ---
PLAYER_MODEL_PATH = os.path.join(WORKSPACE_ROOT, "models", "yolov8m.pt")       # COCO model — excellent at detecting people
BALL_MODEL_PATH = os.path.join(WORKSPACE_ROOT, "models", "basketball_best.pt")  # Fine-tuned — detects ball + referee
PLAYER_CONFIDENCE = 0.25   # Min confidence for player detections
BALL_CONFIDENCE = 0.15     # Min confidence for ball detections (low because ball is small/blurry)
TRACKER_CONFIDENCE = 0.5

# --- Team Classifier (Repo 3 - K-Means) ---
TEAM_1_COLOR = (255, 0, 0) # Red
TEAM_2_COLOR = (0, 255, 0) # Green

# --- Court Mapper (Repo 1 - Homography) ---
COURT_IMAGE_PATH = os.path.join(REPO1_DIR, "tactical_view", "court_images", "basketball_court.png")
COURT_MODEL_PATH = os.path.join(REPO1_DIR, "models", "court_keypoint_detector.pt")

# --- Shot Detector (Repo 2 / Repo 4) ---
# We will use Repo 2's geometric approach as the base, supplemented by Repo 4's ResNet50 if weights exist.
RESNET_MODEL_PATH = os.path.join(REPO4_DIR, "models", "resnet50_cropped.pth")
YOLO_GENERAL_MODEL = os.path.join(WORKSPACE_ROOT, "models", "yolov8l.pt")

# --- Highlights (Repo 4) ---
HIGHLIGHT_CLIP_BEFORE_SEC = 4
HIGHLIGHT_CLIP_AFTER_SEC = 2

# --- Performance Tuning (RTX 4060, 8GB VRAM) ---
BATCH_SIZE = 250                # Frames per batch (tuned for 8GB VRAM — 500 causes OOM)
COURT_KEYPOINT_STRIDE = 5      # Run court keypoint model every N frames (interpolate rest)
FRAME_SKIP = 1                  # Process every Nth frame (1 = all, 2 = half speed)
OUTPUT_CODEC = "mp4v"           # Video codec: "mp4v" (universal) or "avc1" (smaller, needs ffmpeg)
USE_ONNX = False                 # Export YOLO to .engine for GPU TensorRT speedup
OUTPUT_RESOLUTION = (1280, 720) # Upscale to 720p to improve quality and ball detection

# --- Analytics Output ---
ANALYTICS_DIR = os.path.join(WORKSPACE_ROOT, "data", "output")
os.makedirs(ANALYTICS_DIR, exist_ok=True)
