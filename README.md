# Basketball CourtVision Spatial Analytics Engine 🏀

An end-to-end computer vision and spatial analytics pipeline for analyzing basketball games from broadcast video. This project seamlessly integrates state-of-the-art YOLO object tracking, SAM2 segmentation, SigLIP zero-shot classification, EasyOCR player identification, and pure geometric parabolic shot detection into a single automated pipeline.

## 🌟 Key Features

1. **Player & Ball Tracking (RF-DETR + Custom YOLO)** 🏃‍♂️
   - Tracks players using RF-DETR (Roboflow DETR) and basketballs using a custom YOLOv8 model.
   - Utilizes `roboflow/sports` BallTracker for physics-aware ball path smoothing.
   
2. **Player Segmentation (SAM2)** ✂️
   - Generates pixel-perfect segmentation masks for players on the court using Meta's Segment Anything Model 2 (SAM2).

3. **Dynamic Team Classification (Pixel K-Means)** 👕
   - Employs Pixel-level K-Means clustering to isolate dominant jersey colors.
   - Dynamically discovers and assigns Team 1 vs Team 2 colors without hardcoding, perfectly separating opposing teams even during extreme broadcast zooms where UMAP struggles.

4. **Player Identification (EasyOCR)** 🔢
   - Reads jersey numbers in real-time utilizing EasyOCR to persist player identities throughout the broadcast.

5. **Parabolic Shot Detection (Make/Miss Logic)** 🎯
   - Pure geometric trajectory mapper that traces the basketball's path over time.
   - Fits a 2nd-degree polynomial to the ball's coordinates to identify a shot arc and determines Make/Miss events by calculating intersections with hoop regions.

6. **Broadcast Visuals & Minimap** 📺
   - Draws glowing, fading colored trails behind the ball using `sports.common.ball.BallAnnotator`.
   - Maps player coordinates via Homography to a 2D top-down tactical minimap.
   - Generates spatial heatmaps showing team movement and defensive pressure over time.

## 📁 Architecture

The project features a highly modular structure:

```text
CourtVision/
├── main_pipeline.py          # Master entry point. Executes the entire pipeline.
├── config.py                 # Configuration variables, paths, and hardware settings.
├── setup_models.py           # Auto-downloads all model weights to models/ directory.
├── core/
│   ├── detector.py           # RF-DETR and YOLO tracking.
│   ├── segmentation.py       # SAM2 integration.
│   ├── team_classifier.py    # SigLIP + UMAP + KMeans clustering.
│   ├── player_id.py          # EasyOCR jersey number detection.
│   ├── shot_detector.py      # Parabolic Make/Miss trajectory engine.
│   ├── court_mapper.py       # Handles Homography transformation for the Minimap.
│   └── visualizer.py         # Main video renderer (HUD, ID tags, ball trails).
├── models/                   # Contains all downloaded ML weights (portable offline!).
└── data/
    ├── videos/               # Raw input video files.
    └── output/               # Processed videos, heatmaps, and JSON reports.
```

## 🚀 Usage & Portability

**Installation & Setup:**

### 1. Clone & Install
```bash
git clone https://github.com/PanavCodes/CourtVision.git
cd CourtVision
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Offline Models
CourtVision requires several large AI models (YOLOv8, SAM2, SigLIP, etc.). Download them automatically by running:
```bash
python setup_models.py
```

### 3. Run Pipeline
```bash
chmod +x setup.sh
./setup.sh
```

**Running the Pipeline:**
Make sure you activate your environment first!
*Windows:* `venv\Scripts\activate`
*Linux/macOS:* `source venv/bin/activate`
```bash
python main_pipeline.py data/videos/fiba_first_half.mp4
```

*Optional Arguments:*
- `--max-frames 100`: Stop processing after 100 frames (useful for testing).

---
*Developed by the CourtVision AI Engineering Team.*
