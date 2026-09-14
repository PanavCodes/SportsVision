import os
import torch

# Base Paths
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECTS_ROOT = os.path.dirname(WORKSPACE_ROOT)

# Localize Cache Directories for portability
os.environ["HF_HOME"] = os.path.join(WORKSPACE_ROOT, "models", "shared", "hf_cache")
os.environ["EASYOCR_MODULE_PATH"] = os.path.join(WORKSPACE_ROOT, "models", "shared", "easyocr")

# Output & Video Directories (Auto-created on fresh clones)
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "data", "output")
VIDEOS_DIR = os.path.join(WORKSPACE_ROOT, "data", "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(VIDEOS_DIR, "basketball"), exist_ok=True)
os.makedirs(os.path.join(VIDEOS_DIR, "cricket"), exist_ok=True)

# Hardware Configuration (NVIDIA CUDA accelerated, CPU fallback)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_FP16 = True if torch.cuda.is_available() else False

# ==========================================
# Shared Models & Global Performance Tuning
# ==========================================
SHARED_MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models", "shared")
os.makedirs(SHARED_MODELS_DIR, exist_ok=True)

# Common human detector (COCO person class)
PLAYER_MODEL_PATH = os.path.join(SHARED_MODELS_DIR, "yolov8m.pt")
if not os.path.exists(PLAYER_MODEL_PATH):
    # Fallback to repo root or models directory if present
    alt_player = os.path.join(WORKSPACE_ROOT, "models", "yolov8m.pt")
    if os.path.exists(alt_player):
        PLAYER_MODEL_PATH = alt_player

# SAM2 segmentation
USE_SAM2 = True
SAM_MODEL_PATH = os.path.join(SHARED_MODELS_DIR, "sam2.1_s.pt")
if not os.path.exists(SAM_MODEL_PATH):
    alt_sam = os.path.join(WORKSPACE_ROOT, "sam2.1_s.pt")
    if os.path.exists(alt_sam):
        SAM_MODEL_PATH = alt_sam

# Feature Toggles
USE_SIGLIP = False
USE_OCR = True
USE_RF_DETR = False  # Set to True if rf-detr is installed in environment

# Performance Tuning (Tuned for RTX 4060 8GB VRAM)
BATCH_SIZE = 250                 # Frames per batch
FRAME_SKIP = 1                   # Process every Nth frame
OUTPUT_CODEC = "mp4v"            # Video codec
USE_ONNX = False                 # Export YOLO to .engine for TensorRT
OUTPUT_RESOLUTION = (1280, 720)  # Standard 720p output

# ==========================================
# Basketball Pipeline Configuration
# ==========================================
BASKETBALL_MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models", "basketball")
BASKETBALL_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "basketball")
BASKETBALL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "basketball")
os.makedirs(BASKETBALL_OUTPUT_DIR, exist_ok=True)

BASKETBALL_BALL_MODEL_PATH = os.path.join(BASKETBALL_MODELS_DIR, "basketball_best.pt")
BASKETBALL_COURT_MODEL_PATH = os.path.join(BASKETBALL_MODELS_DIR, "court_keypoint_detector.pt")
BASKETBALL_COURT_IMAGE_PATH = os.path.join(BASKETBALL_DATA_DIR, "basketball_court.png")

PLAYER_CONFIDENCE = 0.25
BALL_CONFIDENCE = 0.15
TRACKER_CONFIDENCE = 0.5

TEAM_1_COLOR = (255, 0, 0)
TEAM_2_COLOR = (0, 255, 0)
COURT_KEYPOINT_STRIDE = 5
YOLO_GENERAL_MODEL = os.path.join(SHARED_MODELS_DIR, "yolov8l.pt")
HIGHLIGHT_CLIP_BEFORE_SEC = 4
HIGHLIGHT_CLIP_AFTER_SEC = 2

# Legacy / Backward Compatibility Aliases for Basketball
BALL_MODEL_PATH = BASKETBALL_BALL_MODEL_PATH
COURT_MODEL_PATH = BASKETBALL_COURT_MODEL_PATH
COURT_IMAGE_PATH = BASKETBALL_COURT_IMAGE_PATH
ANALYTICS_DIR = BASKETBALL_OUTPUT_DIR

# ==========================================
# Cricket Pipeline Configuration
# ==========================================
CRICKET_MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models", "cricket")
CRICKET_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "cricket")
CRICKET_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cricket")
CRICKET_STATS_DIR = os.path.join(CRICKET_OUTPUT_DIR, "stats")
os.makedirs(CRICKET_STATS_DIR, exist_ok=True)

CRICKET_BALL_MODEL_PATH = os.path.join(CRICKET_MODELS_DIR, "cricket_ball_best.pt")
CRICKET_YOLO_MODEL_PATH = os.path.join(CRICKET_MODELS_DIR, "cricket_yolov8_best.pt")
CRICKET_PITCH_IMAGE_PATH = os.path.join(CRICKET_DATA_DIR, "cricket_pitch.png")
CRICKET_GROUND_IMAGE_PATH = os.path.join(CRICKET_DATA_DIR, "cricket_ground.png")

CRICKET_BALL_CONFIDENCE = 0.20
CRICKET_PLAYER_CONFIDENCE = 0.30
CRICKET_STUMPS_CONFIDENCE = 0.35

# 22-yard pitch real-world metric dimensions
CRICKET_PITCH_LENGTH_METERS = 20.12  # 22 yards between bowling creases
CRICKET_PITCH_WIDTH_METERS = 3.05    # 10 feet width

# 10 Standard CricShot10 shot classes
CRICKET_SHOT_CLASSES = [
    "cover_drive",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight_drive",
    "sweep"
]

# ==========================================
# Sport Router Configuration
# ==========================================
ROUTER_SAMPLE_FRAMES = 5
ROUTER_CONFIDENCE_THRESHOLD = 0.60
