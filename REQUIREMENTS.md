# SportsVision — System Requirements & Cross-Platform Setup Guide

This guide details the hardware specifications, software dependencies, and step-by-step setup procedures to run **SportsVision** seamlessly on any PC (Windows, Linux, or macOS).

---

## 💻 Hardware Requirements

| Component | Minimum Specification | Recommended Specification (Broadcast Grade) |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit), Ubuntu 20.04+, macOS 12+ | Windows 11 (64-bit) or Ubuntu 22.04 LTS |
| **Processor (CPU)** | Intel Core i5 (8th gen+) or AMD Ryzen 5 | Intel Core i7/i9 (12th gen+) or AMD Ryzen 7/9 |
| **System Memory (RAM)** | 16 GB | 32 GB DDR4 / DDR5 |
| **Graphics (GPU)** | CPU Fallback Supported (or GTX 1660 6GB) | NVIDIA RTX 3060 / 4060 (8GB VRAM) or higher |
| **CUDA Compute** | CUDA 11.8 or 12.1+ | CUDA 12.1+ with FP16 Tensor Cores |
| **Disk Storage** | 10 GB free space (SSD recommended) | 25 GB free space (for match footage & high-res exports) |

> [!NOTE]
> **CPU Fallback Mode**: SportsVision natively supports running on CPU-only machines. If an NVIDIA GPU is not detected, the system will automatically fall back to CPU execution. Processing will simply be slower than with GPU acceleration.

---

## 🛠️ Software Prerequisites

1. **Python**: Python **3.10**, **3.11**, or **3.12** (64-bit) is recommended.
   - *Note: Python 3.13 is very recent and certain pre-compiled binary wheels (PyTorch, EasyOCR, UMAP) may require build tools on Windows. Python 3.10–3.12 is tested and fully stable.*
2. **Git**: Installed and accessible in your system terminal / Command Prompt.
3. **NVIDIA GPU Drivers** (Optional, for GPU acceleration): Driver version **>= 535.xx** for CUDA 12.x support.

---

## ⚡ Quick Setup (Automated)

### Windows
Double-click [setup.bat](file:///setup.bat) or open Command Prompt / PowerShell in the repository root and run:
```cmd
setup.bat
```

### Linux / macOS
Open a terminal in the repository root and run:
```bash
chmod +x setup.sh
./setup.sh
```

The automated script will:
1. Create a clean Python virtual environment (`venv`).
2. Upgrade `pip`, `setuptools`, and `wheel`.
3. Install all production dependencies from `requirements.txt`.
4. Run `setup_models.py` to verify fine-tuned weights and download pre-trained foundation models (`yolov8m.pt`, `sam2.1_s.pt`, EasyOCR weights).

---

## 🔧 Manual Step-by-Step Installation

If you prefer setting up manually or need a customized CUDA configuration:

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/SportsVision.git
cd SportsVision
```

### 2. Create and Activate Virtual Environment

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade Core Packaging Tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch Tailored to Your Hardware

#### A. NVIDIA GPU with CUDA 12.1 (Recommended for RTX 30/40 Series)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### B. NVIDIA GPU with CUDA 11.8 (For Older Drivers / GTX 10/20 Series)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### C. CPU Only (No Dedicated NVIDIA GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### D. Apple Silicon (M1 / M2 / M3 / M4 Mac)
```bash
pip install torch torchvision
```

### 5. Install SportsVision Dependencies
```bash
pip install -r requirements.txt
```

### 6. Verify and Download Model Weights
Run the automated verification script:
```bash
python setup_models.py
```
This script validates:
- `models/basketball/basketball_best.pt` (Fine-tuned basketball detector)
- `models/basketball/court_keypoint_detector.pt` (Court keypoint locator)
- `models/cricket/CBDbest.pt` / `cricket_ball_best.pt` (Cricket ball detectors)
- Shared base models (`yolov8m.pt`, `sam2.1_s.pt`, EasyOCR reader)

---

## 📁 Model Storage Architecture

```text
models/
├── basketball/
│   ├── basketball_best.pt          # Fine-tuned ball & player weights (~22 MB)
│   └── court_keypoint_detector.pt  # Court homography keypoints (~8 MB)
│
├── cricket/
│   ├── CBDbest.pt                  # Lightweight cricket ball detector (~22 MB)
│   └── cricket_ball_best.pt        # High-precision ball detector (~50 MB)
│
└── shared/                         # Auto-downloaded by setup_models.py
    ├── yolov8m.pt                  # General human / person detector (~50 MB)
    ├── sam2.1_s.pt                 # SAM2 player instance segmentation (~88 MB)
    ├── easyocr/                    # CRAFT & English OCR weights
    └── hf_cache/                   # HuggingFace cache for vision models
```

---

## 🔍 Troubleshooting Common Setup Issues

### 1. Windows CUDA DLL Initialization Error
**Symptom**: `OSError: [WinError 126] The specified module could not be found` or `CUDA available: False` despite having an NVIDIA GPU.
**Solution**:
1. Ensure your NVIDIA driver is up to date ([nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)).
2. `main_pipeline.py` automatically searches standard CUDA paths and `CUDA_PATH`. If your toolkit is installed in a custom path, set the environment variable:
   ```cmd
   set CUDA_PATH=C:\Your\Custom\CUDA\Path
   ```

### 2. Out of Memory (CUDA OOM) During Long Match Videos
**Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory`.
**Solution**:
- SportsVision features streaming batch execution. Decrease the batch size using the CLI:
  ```bash
  python main_pipeline.py video.mp4 --batch-size 100
  ```
- Or edit [config.py](file:///config.py):
  ```python
  BATCH_SIZE = 100  # Default is 250 (calibrated for 8GB VRAM)
  USE_SAM2 = False  # Disable SAM2 instance masks to conserve VRAM
  ```

### 3. Missing Video Codec for Output MP4
**Symptom**: OpenCV throws `OpenCV: FFMPEG: tag 0x7634706d/'mp4v' is not supported with codec id 12`.
**Solution**:
- In [config.py](file:///config.py), change `OUTPUT_CODEC`:
  ```python
  OUTPUT_CODEC = "avc1"  # or "H264" or "XVID"
  ```

### 4. Windows Unicode Console Display Issues
**Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character...`
**Solution**:
- SportsVision scripts use clean ASCII status indicators (`[OK]`, `[WARN]`, `[FAIL]`) to avoid Windows `cp1252` encoding errors.
- You can also force Python to use UTF-8 globally on Windows by setting:
  ```cmd
  set PYTHONUTF8=1
  ```

---

## 🏃 Testing Your Installation

Verify the entire multi-sport pipeline with a quick 30-frame test run:

```bash
# Test Cricket Pipeline
python main_pipeline.py data/videos/cricket/cricket_sample.mp4 --max-frames 30

# Test Basketball Pipeline
python main_pipeline.py data/videos/basketball/test_clip.mp4 --max-frames 30
```
