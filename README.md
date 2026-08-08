# Basketball CourtVision Spatial Analytics Engine 🏀

An end-to-end computer vision and spatial analytics pipeline for analyzing basketball games from broadcast video. This project seamlessly integrates state-of-the-art YOLO object tracking, homography-based minimap rendering, machine-learning-driven team classification, and geometric shot detection into a single automated pipeline.

## 🌟 Key Features

1. **Player & Ball Tracking (BoT-SORT)** 🏃‍♂️
   - Tracks players and basketballs simultaneously using dual YOLOv8 models (`yolov8m` for players, custom fine-tuned `basketball_best` for ball/referee).
   - Utilizes advanced BoT-SORT algorithms to maintain identities through occlusions and fast movements.
   
2. **Dynamic Team Classification (K-Means)** 👕
   - Automatically discovers the primary colors of both teams playing based on jersey extraction (no hardcoded team colors).
   - Leverages a custom heuristic that targets the center chest patch to extract reliable jersey pixels even on heavily stylized courts.
   - Accurately classifies players to Team 1, Team 2, or Referee.

3. **Tactical Minimap Overlay (Homography)** 🗺️
   - Maps player coordinates from the broadcast camera view to a 2D top-down tactical minimap.
   - Utilizes `CourtKeypointDetector` to identify court lines and compute a planar homography matrix.
   - *Note on limitations:* Perspective-heavy FIBA broadcast angles may produce distorted keypoints which the engine correctly identifies as invalid, safely disabling the minimap tracking for that segment.

4. **Automated Analytics & Heatmaps** 📊
   - Tracks ball possession percentages (Team 1 vs Team 2).
   - Generates spatial heatmaps showing team movement and defensive pressure over time.
   - Exports all analytics to `test_clip_analytics.json` for external dashboard integration.

## 📁 Architecture

The project has been refactored into a clean, modular structure:

```text
CourtVision/
├── main_pipeline.py          # Master entry point. Executes the 8-stage pipeline.
├── config.py                 # Configuration variables, paths, and hardware settings.
├── core/
│   ├── detector.py           # Wraps YOLO tracking for ball, referee, and players.
│   ├── team_classifier.py    # K-Means logic for dynamic color extraction and assignment.
│   ├── court_mapper.py       # Handles Homography transformation for the Minimap.
│   ├── visualizer.py         # Main video renderer (ellipses, HUD, ID tags).
│   └── ...                   # Other core analytical engines.
├── drawers/                  # UI components for rendering the minimap and graphics.
├── external/                 # Integrated dependencies (Court Keypoints, ResNet50, etc.).
├── models/                   # Contains ML weights (.pt files).
└── data/
    ├── videos/               # Raw input video files.
    └── output/               # Processed videos, heatmaps, and JSON reports.
```

## 🚀 Usage

**Requirements:**
- Python 3.10+
- `pip install -r requirements.txt`

**Running the Pipeline:**
```bash
python main_pipeline.py data/videos/fiba_first_half.mp4
```

*Optional Arguments:*
- `--max-frames 100`: Stop processing after 100 frames (useful for testing).

## 🛠️ Recent Fixes & Improvements

- **Dynamic Color Injection**: The K-Means team colors are now automatically injected into both the HUD Visualizer and the Minimap Tactical Drawer.
- **Improved Referee Exclusion**: Bounding box spatial matching thresholds for referees were tightened to ensure players fighting for the ball are no longer accidentally excluded from tracking.
- **FIBA Court Resilience**: Jersey pixel extraction now focuses purely on a tighter chest crop, preventing grey/wood FIBA court colors from bleeding into the team classification clustering.

---
*Developed by the CourtVision AI Engineering Team.*
