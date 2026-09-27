# SportsVision: Multi-Sport AI Computer Vision & Spatial Analytics Engine 🏀 🏏

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%2012.1)-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Studio-009688.svg)](https://fastapi.tiangolo.com/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20Accelerated-76b900.svg)](https://www.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SportsVision** is an end-to-end, broadcast-grade computer vision and spatial analytics engine for team sports. It features an intelligent **Sport-Detection Router** that ingests raw game footage, automatically identifies whether the sport is **Basketball** or **Cricket**, and dispatches the execution pipeline with zero manual configuration.

---

## 🌟 Key Capabilities

```text
                                Raw Video Input (.mp4 / .mov / .avi)
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  Sport-Detection      │
                                    │  Router (Zero-Shot)   │
                                    └───────────┬───────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │     🏀 BASKETBALL ENGINE    │                   │      🏏 CRICKET ENGINE      │
        ├─────────────────────────────┤                   ├─────────────────────────────┤
        │ • BoT-SORT / ByteTrack      │                   │ • Ballistic Kalman Tracking │
        │ • SAM2 Instance Masks       │                   │ • DRS Hawk-Eye 3D Projector │
        │ • K-Means Jersey Classifier │                   │ • 22-Yard Pitch Homography  │
        │ • EasyOCR Jersey Number ID  │                   │ • Metric Line & Length      │
        │ • Parabolic Shot Detector   │                   │ • Speed Gun Telemetry       │
        │ • Tactical Court Minimap    │                   │ • CricShot10 Classifier     │
        │ • Thermal Team Heatmaps     │                   │ • Ground Radar Minimap      │
        │ • Highlight Reel Generator  │                   │ • Auto Commentary & Stats   │
        └─────────────────────────────┘                   └─────────────────────────────┘
```

### 1. 🤖 Intelligent Sport Router
- **Zero-Manual-Configuration**: Analyzes representative video frames using surface chromatic distribution (outfield turf grass vs. indoor hardwood floor) and zero-shot visual embeddings.
- Automatically selects and executes either the Basketball or Cricket pipeline.
- Optional CLI override (`--sport {auto, basketball, cricket}`) for direct execution.

### 2. 🏏 Cricket Spatial Analytics Engine (Hawk-Eye & DRS)
- **Physics-Aware Ball Tracking**: YOLOv8 ball detector paired with an adaptive ballistic Kalman filter ($[x, y, v_x, v_y]^T$ with gravity modeling) to bridge dropped and motion-blurred frames.
- **DRS Hawk-Eye & Stumps Projection**: Detects pitch bounce inflections, calculates 3D incident/rebound angles, and extrapolates the post-bounce ballistic arc towards the stumps for automated LBW review.
- **22-Yard Pitch Homography**: Perspective transformation mapping camera pixels to real-world pitch metric coordinates ($20.12\text{m} \times 3.05\text{m}$).
- **Line & Length Analytics**: Classifies deliveries into *Yorker, Full, Good Length, Short of Length, Bouncer* and lateral lines (*Outside Off, On Stumps, Leg Line, Wide*).
- **Speed Gun Telemetry**: Computes release speed, bounce speed, and pitch deceleration percentage in km/h.
- **Batting Shot Recognition**: Classifies batting strokes using the 10 standard **CricShot10** classes (*Cover Drive, Defense, Flick, Hook, Late Cut, Lofted, Pull, Square Cut, Straight Drive, Sweep*).
- **Field Radar & Pitch Minimap**: Dual overlays showing delivery pitch location on the 22-yard strip and ball flight trajectory across the entire circular cricket ground.
- **Event Detection & Commentary**: Automated detection of wickets, boundaries (4s and 6s), dot balls, and no-balls, with automated ball-by-ball commentary generation and JSON export.

### 3. 🏀 Basketball Spatial Analytics Engine (CourtVision)
- **Player & Ball Tracking**: High-precision YOLOv8m player detection with BoT-SORT/ByteTrack and fine-tuned basketball detection.
- **SAM2 Player Segmentation**: Pixel-perfect instance segmentation masks for players on court.
- **Dynamic Team Classification**: Pixel-level K-Means clustering identifying Team 1 vs. Team 2 jersey colors without manual labeling.
- **Player Identification (EasyOCR)**: Reads and persists jersey numbers across camera cuts and occlusions.
- **Tactical Minimap & Homography**: Transforms court keypoints to a 2D top-down tactical court view, plotting real-time player positions and ball possession.
- **Parabolic Shot Engine**: Fits a 2nd-degree polynomial curve to the ball trajectory to detect shot releases, make/miss events, and shooting percentages.
- **Passing & Possession Analytics**: Proximity-based ball possession tracking, completed passes, and defensive interceptions.
- **Thermal Heatmaps & Highlights**: Generates 2D thermal spatial heatmaps for each team and auto-cuts game highlights.

---

## 📁 Repository Structure

```text
SportsVision/
├── main_pipeline.py                  # Master entrypoint & multi-sport dispatcher
├── config.py                         # Unified configuration with hardware & sport tuning
├── setup_models.py                   # Automated model downloader & integrity checker
├── requirements.txt                  # Python dependencies
├── REQUIREMENTS.md                   # Complete hardware & cross-PC setup guide
├── setup.bat                         # Automated 1-click setup script for Windows
├── setup.sh                          # Automated 1-click setup script for Linux / macOS
│
├── core/
│   ├── basketball/                   # Basketball analytics engine
│   │   ├── pipeline.py               # Basketball pipeline runner
│   │   ├── detector.py               # Player & ball detector
│   │   ├── team_classifier.py        # K-Means jersey color clustering
│   │   ├── player_id.py              # EasyOCR jersey number detection
│   │   ├── court_mapper.py           # Court homography & tactical minimap
│   │   ├── ball_possession.py        # Proximity possession estimation
│   │   ├── pass_detector.py          # Pass & interception analyzer
│   │   ├── speed_calculator.py       # Player speed & distance tracking
│   │   ├── shot_detector.py          # Parabolic make/miss shot detector
│   │   ├── visualizer.py             # Broadcast graphics & player badges
│   │   ├── highlight_generator.py    # Highlight video compiler
│   │   ├── analytics_report.py       # Basketball JSON report exporter
│   │   ├── segmentation.py           # SAM2 player segmentation
│   │   ├── basketball_tracker.yaml   # BoT-SORT tracking configuration
│   │   └── minimap/                  # Homography, keypoints, and tactical drawer
│   │
│   ├── cricket/                      # Cricket analytics engine
│   │   ├── pipeline.py               # Cricket pipeline runner
│   │   ├── detector.py               # Ball, player, stumps detection & role assignment
│   │   ├── ball_tracker.py           # ByteTrack + Kalman ballistic filter
│   │   ├── trajectory.py             # 3D parabolic curves, bounce angles & Hawk-Eye
│   │   ├── pitch_mapper.py           # 22-yard pitch homography & Ground Radar
│   │   ├── speed_calculator.py       # Release & bounce speed gun (km/h)
│   │   ├── shot_classifier.py        # CricShot10 batting shot action classifier
│   │   ├── event_detector.py         # Milestones: bounce, impact, wicket, 4, 6, dot
│   │   ├── delivery_manager.py       # Ball-by-ball delivery cycle manager
│   │   ├── commentary_generator.py   # Automated match commentary generator
│   │   ├── stats_extractor.py        # Per-delivery & match statistics aggregator
│   │   ├── visualizer.py             # Broadcast HUD (speed gun, pitch minimap, radar)
│   │   ├── highlight_generator.py    # Auto-compiler for wickets and boundaries
│   │   └── analytics_report.py       # Cricket JSON report exporter
│   │
│   ├── shared/                       # Shared utilities
│   │   ├── gpu_diagnostics.py        # CUDA verification & device reporting
│   │   └── video_io.py               # Threaded video writer & time formatters
│   │
│   └── sport_router/                 # Automatic sport detector
│       ├── frame_sampler.py          # Evenly spaced video frame sampling
│       ├── classifier.py             # Field chromatic analysis & zero-shot classifier
│       └── router.py                 # Majority voting & dispatch logic
│
├── models/
│   ├── basketball/                   # basketball_best.pt, court_keypoint_detector.pt
│   ├── cricket/                      # CBDbest.pt, cricket_ball_best.pt
│   └── shared/                       # Auto-downloaded: yolov8m.pt, sam2.1_s.pt, easyocr/
│
├── data/
│   ├── basketball/                   # basketball_court.png (tactical court template)
│   ├── cricket/                      # cricket_pitch.png, cricket_ground.png (templates)
│   ├── videos/                       # Place input match videos here
│   └── output/                       # Generated annotated videos, reports & highlights
│
└── training/                         # Model training & fine-tuning
    ├── train_cricket_yolov8.py       # YOLOv8 training script for cricket
    └── optuna_tuning_config.yaml     # Optuna hyperparameter tuning configuration
```

---

## ⚡ Quick Start

### 1. Automated Setup (Recommended)

**Windows:**
```cmd
setup.bat
```

**Linux / macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Manual Setup
See [REQUIREMENTS.md](file:///REQUIREMENTS.md) for detailed hardware specifications, CUDA matrix, and step-by-step installation instructions.

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate | Linux: source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify & download foundation models
python setup_models.py
```

### 3. Launch Web Studio Frontend 🌐 (Recommended for Presentations & Demos)
Run the broadcast-grade web interface to view live telemetry, interactive video playback, tactical minimaps, and post-game analytics without using terminal commands:
- **Windows One-Click:** Double-click [run_app.bat](file:///run_app.bat)
- **Linux / macOS:**
  ```bash
  chmod +x run_app.sh
  ./run_app.sh
  ```
- **Manual Launch:**
  ```bash
  python app.py
  ```
  The browser will automatically open to `http://localhost:8000`.

#### Web Studio Features:
- **Multi-Sport Adaptive UI**: Automatically switches telemetry, minimaps, and metric cards between Cricket and Basketball.
- **Live Pipeline Monitor**: Real-time log console via SSE, stage progress indicators, GPU VRAM tracking, and dynamic ETA.
- **Cricket Analytics Suite**: 22-Yard Pitch Homography Map with impact rings, Full Ground Radar Minimap, Speed Gun telemetry, ball-by-ball commentary with filterable events (Wickets, Boundaries, Dots), and auto-cut highlight reels.
- **Basketball Analytics Suite**: 2D Tactical Court Minimap, Team possession breakdown, parabolic shot tracking (makes/misses), and player speed/distance metrics.
- **Web-Optimized Transcoding**: Built-in background H.264 transcoding for instantaneous in-browser video playback.

---

## 🚀 Running via Command-Line Pipeline

SportsVision automatically detects the sport from the input video:

```bash
# Process a Cricket Match Video
python main_pipeline.py data/videos/cricket/cricket_sample.mp4

# Process a Basketball Game Video
python main_pipeline.py data/videos/basketball/test_clip.mp4
```

### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `video_path` | `str` | *Required* | Path to the input video file. |
| `--sport` | `str` | `auto` | Force a pipeline (`auto`, `basketball`, `cricket`). |
| `--max-frames` | `int` | `None` | Process only the first $N$ frames (ideal for quick tests). |
| `--frame-skip` | `int` | `1` | Process every $N$-th frame (`1` = all frames, `2` = half). |
| `--batch-size` | `int` | `250` | Batch size for streaming memory management (tuned for 8GB VRAM). |
| `--resume` | `flag` | `False` | Resume processing from the last saved batch checkpoint. |

#### Example Commands

```bash
# Quick test on first 100 frames
python main_pipeline.py match.mp4 --max-frames 100

# Force Cricket pipeline with custom batch size
python main_pipeline.py match.mp4 --sport cricket --batch-size 150

# Fast preview at half frame rate
python main_pipeline.py game.mp4 --frame-skip 2
```

---

## 📊 Deliverables & Export Formats

### Cricket Deliverables
1. **Annotated Broadcast Video**: `data/output/cricket/<video_name>_annotated.mp4`
   - HUD speed gun badge (Release & Bounce speed in km/h)
   - Real-time 22-yard pitch minimap with ball bounce impact rings
   - Ground radar overlay showing ball travel path across the outfield
   - 3D Hawk-Eye trajectory trail
2. **Match Analytics JSON**: `data/output/cricket/stats/<video_name>.json`
   - Delivery-by-delivery speeds, pitch metric $(X, Y)$ coordinates, Line & Length classification
   - Bounce incident & rebound angles
   - CricShot10 shot played and confidence score
   - Outcome classification (*Wicket, Four, Six, Dot Ball, Single, Double*)
   - Ball-by-ball commentary transcript
3. **Highlights Reel**: `data/output/cricket/<video_name>_highlights.mp4` (auto-clipped wickets and boundaries).

### Basketball Deliverables
1. **Annotated Broadcast Video**: `data/output/basketball/<video_name>_annotated.mp4`
   - Player bounding boxes and jersey number badges
   - Dynamic team color indicators
   - 2D tactical minimap with player coordinates and ball possession
2. **Game Analytics JSON**: `data/output/basketball/<video_name>_analytics.json`
   - Team possession percentages
   - Pass counts and interceptions
   - Parabolic shot attempts and make/miss detections
   - Player distance traveled and average speeds
3. **Thermal Team Heatmaps**: `data/output/basketball/<video_name>_team1_heatmap.jpg` and `team2_heatmap.jpg`.
4. **Highlights Reel**: `data/output/basketball/<video_name>_highlights.mp4`.

---

## ⚙️ Hardware Tuning & Performance

SportsVision is calibrated for consumer GPUs (e.g., **NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM**) and scales to enterprise workstations:

- **Streaming Batch Processing**: Processes video in batches (default: 250 frames) to maintain constant memory usage regardless of video duration.
- **CUDA FP16 Half-Precision**: Enabled by default for 2x faster inference and 50% lower VRAM footprint.
- **CPU Fallback**: Automatically activates on machines without an NVIDIA GPU.

To tune performance, adjust settings in [config.py](file:///config.py) or via CLI arguments:
```python
BATCH_SIZE = 250         # Lower to 100-150 for 4GB-6GB GPUs
USE_FP16 = True          # Half-precision inference
USE_SAM2 = True          # Set False to disable SAM2 segmentation and boost FPS
USE_OCR = True           # EasyOCR jersey number detection
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](file:///LICENSE) file for details.

*Developed by the SportsVision Multi-Sport AI Engineering Team.*
