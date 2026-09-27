# PROJECT_CONTEXT.md — SportsVision Research & System Reference

**Repository:** `PanavCodes/SportsVision` (formerly *CourtVision*)  
**Evaluated Branch / Commit:** `main` @ commit `33e4769` (*"Add CourtVision Spatial Analytics Web Studio frontend and launcher"*)  
**Working Tree State:** Clean (working directory identical to `origin/main`; 0 uncommitted changes)  
**Evaluation Date:** 2026-09-27  

---

## 1. Research-Paper-Ready Summary (System Overview)

> **Abstract / System Overview Paragraph (Direct Paper Extraction):**  
> We present **SportsVision**, an automated multi-sport computer vision and spatial analytics engine designed for continuous broadcast video analysis without manual calibration or game-type configuration. The system introduces an intelligent front-end **Sport-Detection Router** that dynamically classifies input footage into domain-specific processing graphs—namely **Basketball** or **Cricket**—using a hybrid representation combining outfield chromatic surface distributions (turf grass vs. hardwood varnish) with zero-shot vision-transformer embeddings (SigLIP / ViT-B/16). Core detection and spatial localization leverage a unified **Ultralytics YOLOv8** stack across both sports, eliminating heterogeneous model runtime overheads while ensuring deterministic memory footprints on consumer hardware (NVIDIA RTX 4060 8GB VRAM). In the basketball pipeline, dual-detector tracking (COCO YOLOv8m human tracking via BoT-SORT combined with fine-tuned ball and referee models) is coupled with SAM2.1 prompt-based instance segmentation, unsupervised K-Means and SigLIP jersey color clustering, EasyOCR player number persistence, and a RANSAC-optimized planar homography projection onto a metric 2D tactical court. In the cricket pipeline, the engine implements physics-constrained 3D ballistic trajectory reconstruction ($X, Y, Z$) integrating aerodynamic drag and pitch restitution ($e \approx 0.62$), an official ICC Law 36 Hawk-Eye Decision Review System (DRS) projecting delivery paths through the batsman's stumps at $Y = 20.12\text{ m}$, 22-yard pitch and ground radar homography, automated metric line and length classification, a kinematic CricShot10 stroke classifier, and multimodal audio-visual highlight extraction fusing acoustic energy spikes with visual boundary/dismissal events. The entire architecture operates in streaming batch execution (250-frame windows with keypoint striding), yielding end-to-end telemetry, automated natural language commentary, and structured post-match JSON analytics suitable for broadcast graphics and coaching analysis.

---

## 2. Complete Repository Directory & Module Index

Every file in the repository tree was inspected directly from source code. The table below provides the verified one-line operational purpose for each module, inferred from imports, class/function definitions, and executable logic.

| File Path | Verified Operational Purpose |
| :--- | :--- |
| `config.py` | Central configuration defining file paths, hardware execution flags (CUDA/FP16), model weight paths, performance tuning parameters (batch size, frame skip), and sport-specific physical constants. |
| `main_pipeline.py` | Top-level CLI entry point; handles dynamic Windows CUDA DLL loading, runs the automatic sport router, and dispatches video execution to either the basketball or cricket pipeline. |
| `app.py` | Asynchronous FastAPI web application providing a local Web Studio UI, pipeline process execution management, real-time log streaming via SSE, output transcoding (H.264), and analytics retrieval. |
| `setup_models.py` | Setup and verification script; validates local weights, automatically downloads shared base weights (`yolov8m.pt`, `sam2.1_s.pt`), pre-caches SigLIP, and initializes EasyOCR models. |
| `requirements.txt` | Explicit Python dependency manifest specifying versions for PyTorch (CUDA 12.1), Ultralytics, Supervision, OpenCV, Transformers, MoviePy, EasyOCR, Optuna, and FastAPI. |
| `setup.bat` / `setup.sh` | Automated shell setup scripts for Windows and POSIX systems; initializes virtual environments, installs requirements, and runs `setup_models.py`. |
| `run_app.bat` | One-click launcher script for Windows that activates the virtual environment, launches `uvicorn app:app`, and opens the Web Studio in the default browser. |
| `PLAN.md` | Architectural specification and migration plan detailing the transition from single-sport CourtVision to multi-sport SportsVision, external reference repos, and DRS kinematics. |
| `README.md` | User-facing documentation containing project features, architectural diagrams, quick start instructions, and model storage hierarchies. |
| `REQUIREMENTS.md` | Comprehensive system requirements, CUDA compatibility matrix, and hardware tier guide (RTX 4060 vs. CPU fallback). |
| **`core/sport_router/`** | |
| `core/sport_router/__init__.py` | Package initialization exposing the sport routing package namespace. |
| `core/sport_router/frame_sampler.py` | Class `FrameSampler`; extracts $N$ evenly spaced video frames across the 10%–90% temporal timeline of a video to prevent boundary fade-to-black frames. |
| `core/sport_router/classifier.py` | Class `SportClassifier`; executes surface chromatic analysis (HSV grass vs. hardwood ratios) and zero-shot SigLIP classification to return predicted sport and confidence. |
| `core/sport_router/router.py` | Class `SportRouter` and functional wrapper `detect_sport()`; aggregates frame-level classifications via majority voting to route videos to "basketball", "cricket", or "unknown". |
| **`core/shared/`** | |
| `core/shared/__init__.py` | Package initialization exposing shared cross-sport utilities. |
| `core/shared/gpu_diagnostics.py` | Function `verify_cuda()`; verifies CUDA availability, queries device count/name, warns if non-RTX 4060 GPU is detected, and halts execution if CUDA is missing. |
| `core/shared/video_io.py` | Video I/O helpers: `read_video()`, `save_video()`, threaded non-blocking queue writer `VideoWriterThread`, and timestamp formatter `format_time()`. |
| **`core/basketball/`** | |
| `core/basketball/__init__.py` | Package initialization for basketball pipeline modules. |
| `core/basketball/pipeline.py` | Function `run_basketball_pipeline()`; coordinates the 8-stage basketball spatial analytics pipeline across streaming video batches. |
| `core/basketball/detector.py` | Class `CourtVisionDetector`; dual-model detection loading YOLOv8m (players + BoT-SORT) and `basketball_best.pt` (ball + referee), with referee spatial voting and ball smoothing. |
| `core/basketball/segmentation.py` | Class `PlayerSegmenter`; executes Ultralytics SAM2.1 Small (`sam2.1_s.pt`) prompted by player bounding boxes to generate pixel-level binary instance masks. |
| `core/basketball/team_classifier.py` | Class `CourtVisionTeamClassifier`; dynamically extracts jersey colors using K-Means clustering (default) or SigLIP embeddings + UMAP, assigning player tracks to teams via temporal voting. |
| `core/basketball/player_id.py` | Class `CourtVisionPlayerID`; applies unsharp masking and EasyOCR text recognition on player upper-torso crops to identify jersey numbers with majority vote persistence. |
| `core/basketball/court_mapper.py` | Class `CourtVisionMapper`; optimizes court keypoint CNN inference using stride-based keyframe sampling, estimates homography, transforms player tracks, and renders tactical view. |
| `core/basketball/ball_possession.py` | Class `CourtVisionBallPossession`; determines ball possession per frame using bounding box containment ratio, key anchor point distances, and temporal voting with grace periods. |
| `core/basketball/pass_detector.py` | Class `CourtVisionPassDetector`; tracks ball holder transitions across frames to detect intra-team passes and inter-team defensive interceptions. |
| `core/basketball/speed_calculator.py` | Class `CourtVisionSpeedCalculator`; converts top-down tactical coordinates to real-world meters ($28\text{m} \times 15\text{m}$) to compute player distances and sliding-window speeds in km/h. |
| `core/basketball/shot_detector.py` | Class `CourtVisionShotDetector`; fits 2nd-degree polynomial trajectories ($y(t) = at^2 + bt + c$) to ball centroids over a 45-frame window to classify shot attempts and make/miss outcomes. |
| `core/basketball/visualizer.py` | Class `CourtVisionVisualizer`; renders broadcast overlays including team-colored ellipses, track IDs, jersey numbers, possession triangles, and ball-control/passing HUD cards. |
| `core/basketball/highlight_generator.py` | Class `CourtVisionHighlightGenerator`; cuts video clips around high-confidence shot events using MoviePy and concatenates them into an annotated highlight reel. |
| `core/basketball/analytics_report.py` | Class `CourtVisionAnalyticsReport`; aggregates game-wide possession percentages, passes, speeds, and distances into a JSON report and exports 2D thermal heatmaps. |
| `core/basketball/basketball_tracker.yaml` | Tracker configuration file for BoT-SORT defining tracking thresholds, lost-track buffers (60 frames), and camera motion compensation (`sparseOptFlow`). |
| **`core/basketball/minimap/`** | |
| `core/basketball/minimap/bbox_utils.py` | Geometric helper functions: bounding box center calculation, width calculation, Euclidean distance, and bottom-center foot position extraction. |
| `core/basketball/minimap/court_keypoint_detection.py` | Class `CourtKeypointDetector`; runs fine-tuned YOLO pose/keypoint detector (`court_keypoint_detector.pt`) to locate 14 court keypoints with confidence filtering and fallback memory. |
| `core/basketball/minimap/homography.py` | Class `Homography` and utilities; computes robust perspective transform via OpenCV RANSAC, checks determinant for horizontal flip correction, evaluates reprojection error, and blends matrices. |
| `core/basketball/minimap/tactical_view.py` | Class `TacticalViewConverter`; manages the 18 reference court landmark coordinates ($200 \times 107\text{ px}$), validates detected keypoint aspect ratios, and transforms foot positions. |
| `core/basketball/minimap/tactical_view_drawer.py` | Class `TacticalViewDrawer`; renders the top-down 2D court graphic, interpolates missing player positions across frames, and draws team-coded player indicators on the minimap. |
| `core/basketball/minimap/stubs_utils.py` | Pickle serialization caching helpers (`save_stub`, `read_stub`) for saving intermediate keypoint detections during development. |
| **`core/cricket/`** | |
| `core/cricket/__init__.py` | Package initialization exposing cricket pipeline modules. |
| `core/cricket/pipeline.py` | Function `run_cricket_pipeline()`; orchestrates the 7-stage cricket spatial analytics pipeline, multi-over delivery state machine, and DRS adjudication across video batches. |
| `core/cricket/detector.py` | Class `CricketDetector`; dual-model cricket detection loading YOLOv8 ball detector (`CBDbest.pt`/`cricket_ball_best.pt`) with CLAHE lighting and streak heuristics, YOLOv8m + ByteTrack for players, and Sobel edge-gradient stump localization. |
| `core/cricket/ball_tracker.py` | Classes `CricketBallKalmanFilter` and `CricketBallTracker`; implements a 4-state $[x, y, v_x, v_y]^T$ Kalman filter with gravity transition to smooth ball paths, reject jump outliers, and interpolate missing frames. |
| `core/cricket/trajectory.py` | Classes `Kalman3DSmoother` (6-state 3D Kalman) and `CricketTrajectoryAnalyzer`; detects pitch bounce inflections, fits Bezier trails, and extrapolates 3D ballistic arcs with air drag towards stumps. |
| `core/cricket/pitch_mapper.py` | Class `CricketPitchMapper`; computes homography mapping camera pixels to real-world 22-yard pitch metric meters ($20.12\text{m} \times 3.05\text{m}$), 2D pitch minimap ($600 \times 320$), and circular ground radar ($538 \times 530$). |
| `core/cricket/speed_calculator.py` | Class `CricketSpeedCalculator`; reconstructs metric 3D ballistic positions ($X, Y, Z$) to calculate true 3D spatial delivery release speed, pitch bounce speed, and surface deceleration in km/h. |
| `core/cricket/shot_classifier.py` | Class `CricketShotClassifier`; classifies batting strokes into the 10 CricShot10 taxonomy classes using kinematic exit vectors (velocity magnitude, elevation angle, lateral displacement). |
| `core/cricket/event_detector.py` | Class `CricketEventDetector`; detects delivery lifecycle events including release, pitch bounce, bat impact vs. pad impact, bowled wickets, LBW appeals, and boundary fours/sixes. |
| `core/cricket/drs_engine.py` | Classes `DRSEngine`, `DRSVerdict`, and supporting enums; implements complete ICC Law 36 LBW adjudication (Pitching, Impact, Wickets, 3-meter distance rule, close-proximity rule) and renders broadcast DRS cards. |
| `core/cricket/delivery_manager.py` | Class `DeliveryLifecycleManager`; delivery segmentation state machine distinguishing active ball flight from between-delivery dead time, tracking over counts (e.g. 0.1, 0.2 ... 1.1). |
| `core/cricket/audio_processor.py` | Class `CricketAudioProcessor`; extracts audio soundtracks via MoviePy, computes RMS short-time energy envelopes, and detects acoustic spikes (crowd roars, bat clicks, appeals). |
| `core/cricket/commentary_generator.py` | Class `CricketCommentaryGenerator`; synthesizes natural language ball-by-ball commentary and end-of-over recaps based on telemetry (speed, line, length, shot, outcome, xD score). |
| `core/cricket/stats_extractor.py` | Class `CricketStatsExtractor`; records structured per-delivery dictionaries and aggregates match-level statistics (overs, runs, wickets, speed distributions, shot frequencies). |
| `core/cricket/analytics_engine.py` | Class `ExpectedDismissalEngine`; computes Expected Dismissal (xD) probability scores ($0.00$ to $1.00$) based on length zone, line zone, speed multipliers, and stump intersection bonuses. |
| `core/cricket/umpiring.py` | Classes `NoBallDetector` (front-foot popping crease overstep margin) and `UltraEdgeWaveformSimulator` (synthetic resonant harmonic audio spikes for bat edge vs. pad contact). |
| `core/cricket/visualizer.py` | Class `CricketVisualizer`; renders comprehensive broadcast graphics including ball flight trails, predictive Hawkeye paths, bounce landing rings, speed gun badges, pitch minimaps, ground radar, and DRS overlays. |
| `core/cricket/highlight_generator.py` | Class `CricketHighlightGenerator`; multi-modal audio-visual fusion combining visual event timestamps (wickets, boundaries) with acoustic energy peaks to compile highlight reels via MoviePy or OpenCV. |
| `core/cricket/analytics_report.py` | Class `CricketAnalyticsReport`; aggregates ball-by-ball delivery telemetry and commentary transcripts into structured JSON reports at `data/output/cricket/stats/<video_name>.json`. |
| **`training/`** | |
| `training/train_cricket_yolov8.py` | CLI training script for fine-tuning YOLOv8 models on cricket datasets with CUDA FP16 mixed precision on RTX 4060 GPUs. |
| `training/optuna_tuning_config.yaml` | Optuna hyperparameter optimization configuration specifying search bounds for learning rates, momentum, loss gains (box, cls, dfl), and augmentations. |
| **`web/static/`** | |
| `web/static/index.html` | Frontend markup for CourtVision Web Studio featuring pipeline control panels, real-time stage progression stepper, live terminal log viewer, video playback, and telemetry charts. |
| `web/static/css/style.css` | Modern dark-mode stylesheet utilizing CSS custom properties, glassmorphism containers, responsive grid layouts, and custom status badge styling. |
| `web/static/js/app.js` | Frontend controller establishing Server-Sent Events (SSE) connections with FastAPI, managing real-time progress bars, rendering Chart.js telemetry charts, and handling video selection. |

---

## 3. Component-by-Component Technical Deep Dive

### 3.1 Sport Router Subsystem

```
                           Input Video
                                │
                                ▼
                     ┌─────────────────────┐
                     │    FrameSampler     │  (5 evenly spaced frames across 10%-90%)
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   SportClassifier   │
                     ├─────────────────────┤
                     │ 1. HSV Color Ratio  │  Grass (>0.22) vs Hardwood (>0.25)
                     │ 2. SigLIP Zero-Shot │  google/siglip-base-patch16-224
                     │ 3. Chromatic Fallbk │  Argmax(grass_ratio, wood_ratio)
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     SportRouter     │  Majority vote >= 3/5 & conf >= 0.60
                     └──────────┬──────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        "basketball" Pipeline          "cricket" Pipeline
```

- **File Path:** [`core/sport_router/classifier.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L4-L111), [`core/sport_router/router.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/router.py#L6-L57), [`core/sport_router/frame_sampler.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/frame_sampler.py#L4-L43)
- **Model(s) & Algorithm(s):**
  1. *Chromatic Surface Distribution:* Evaluates HSV pixel masks:
     - Outfield grass mask: $H \in [32, 85], S \in [40, 255], V \in [40, 255]$ ([classifier.py:L49-L52](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L49-L52))
     - Hardwood court mask: $H \in [8, 28], S \in [60, 255], V \in [60, 240]$ ([classifier.py:L55-L58](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L55-L58))
  2. *Zero-Shot Vision Transformer:* `google/siglip-base-patch16-224` loaded via HuggingFace `transformers` ([classifier.py:L23-L33](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L23-L33)).
  3. *Ensemble Aggregation:* Majority voting over 5 sampled frames with confidence gating ($\ge 0.60$) ([router.py:L45-L50](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/router.py#L45-L50)).
- **Implementation Status:** **Fully Implemented**. Chromatic and SigLIP branches are active. (Note: Geometric contour detection mentioned in docstrings is not implemented in code; see Section 6).
- **Input / Output:**
  - *Input:* Video filepath string.
  - *Output:* String: `"basketball"`, `"cricket"`, or `"unknown"`.
- **Constants & Hardware Constraints:**
  - `ROUTER_SAMPLE_FRAMES = 5`, `ROUTER_CONFIDENCE_THRESHOLD = 0.60` ([config.py:L125-L126](file:///c:/important%20files/main%20files/projects/SportsVision/config.py#L125-L126)).
  - Runs SigLIP in `torch.no_grad()` on device specified by `config.DEVICE`.

---

### 3.2 Basketball Pipeline Components

#### A. Dual-Model Player & Ball Tracking
- **File Path:** [`core/basketball/detector.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L19-L261)
- **Model(s) & Algorithm(s):**
  - Player Tracker: `yolov8m.pt` (COCO class 0: person) tracking via native BoT-SORT ([detector.py:L38-L48](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L38-L48)). Optional RF-DETR Medium integration supported if installed ([detector.py:L31-L35](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L31-L35)).
  - Ball Detector: Custom fine-tuned YOLO `basketball_best.pt` evaluating class 0 (ball) and class 2 (referee) at $1280\text{px}$ inference ([detector.py:L50-L54](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L50-L54)).
  - Ball Smoothing: `sports.common.ball.BallTracker` (buffer size 10) if `sports` library is present, falling back to Exponential Moving Average (EMA, $\alpha=0.5$, max coast 30 frames) ([detector.py:L202-L254](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L202-L254)).
- **Implementation Status:** **Fully Implemented**.
- **Input / Output:**
  - *Input:* List of BGR image frames ($N \times H \times W \times 3$).
  - *Output:* Tuple `(player_tracks, final_ball_tracks)` where:
    - `player_tracks`: List of dicts `{track_id: {'bbox': [x1, y1, x2, y2]}}` per frame.
    - `ball_tracks`: List of dicts `{1: {'bbox': [x1, y1, x2, y2], 'conf': float}}` per frame.
- **Constants & Hardware Constraints:**
  - Inference resolution: $640\text{px}$ for player model, $1280\text{px}$ for ball model ([detector.py:L129, L159](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L129-L159)).
  - Confidence thresholds: Player = $0.25$, Ball = $0.15$ ([config.py:L69-L70](file:///c:/important%20files/main%20files/projects/SportsVision/config.py#L69-L70)).
  - Crowd filtering heuristic: Discards player bounding boxes whose bottom edge lies in the upper 25% of the frame: `box[3] < frame_h * 0.25` ([detector.py:L119, L144](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L119-L144)).

#### B. Instance Player Segmentation
- **File Path:** [`core/basketball/segmentation.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/segmentation.py#L7-L57)
- **Model(s) & Algorithm(s):** Ultralytics SAM2.1 Small (`sam2.1_s.pt`) prompted by player bounding boxes ([segmentation.py:L15-L17](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/segmentation.py#L15-L17)).
- **Implementation Status:** **Fully Implemented** (controlled by `config.USE_SAM2 = True`).
- **Input / Output:**
  - *Input:* BGR frame array and list of bounding boxes `[[x1, y1, x2, y2], ...]`.
  - *Output:* List of binary boolean masks (`np.ndarray` of shape $[H, W]$).
- **Constants & Hardware Constraints:** Runs half-precision (`half=True`) on GPU ([segmentation.py:L43](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/segmentation.py#L43)).

#### C. Dynamic Team Classification
- **File Path:** [`core/basketball/team_classifier.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/team_classifier.py#L8-L299)
- **Model(s) & Algorithm(s):**
  - Primary (Default): K-Means clustering ($k=2$, `n_init=5`) over center-chest pixel samples (up to 15,000 samples) with BGR Euclidean distance assignment ([team_classifier.py:L51-L70](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/team_classifier.py#L51-L70)).
  - Secondary (Feature Flag `USE_SIGLIP`): `google/siglip-base-patch16-224` vision embeddings reduced via UMAP ($2\text{D}$) and clustered with K-Means ([team_classifier.py:L30-L47, L72-L107](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/team_classifier.py#L30-L47)).
  - Temporal Smoothing: Check interval of 5 frames per track with majority voting ([team_classifier.py:L26, L236-L296](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/team_classifier.py#L26-L296)). Referee IDs are strictly assigned Team 0 ([team_classifier.py:L205-L217](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/team_classifier.py#L205-L217)).
- **Implementation Status:** **Fully Implemented**.
- **Input / Output:** Modifies `player_tracks` in-place, adding `'team': 1 | 2 | 0`.

#### D. Jersey Number Optical Character Recognition (Player ID)
- **File Path:** [`core/basketball/player_id.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/player_id.py#L6-L79)
- **Model(s) & Algorithm(s):** `easyocr.Reader(['en'])` with unsharp masking preprocessing (`cv2.GaussianBlur` + `cv2.addWeighted`), numeric allowlist `0123456789`, and cumulative probability thresholding ($> 1.0$) ([player_id.py:L12, L53-L75](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/player_id.py#L12-L75)).
- **Implementation Status:** **Fully Implemented** (controlled by `config.USE_OCR = True`).
- **Input / Output:** Appends `'jersey_number': str` to track data.

#### E. Court Keypoints, Homography & Tactical Minimap
- **File Path:** [`core/basketball/court_mapper.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/court_mapper.py#L12-L145), [`core/basketball/minimap/`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/)
- **Model(s) & Algorithm(s):**
  - Keypoint Detection: Fine-tuned YOLO pose model `court_keypoint_detector.pt` locating 14 court keypoints ([court_keypoint_detection.py:L11-L64](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/court_keypoint_detection.py#L11-L64)).
  - Performance Optimization: **Keypoint Striding**—runs the keypoint detector every $N=5$ frames (`COURT_KEYPOINT_STRIDE`), repeating nearest keyframe coordinates for intermediate frames ([court_mapper.py:L38-L71](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/court_mapper.py#L38-L71)).
  - Geometric Validation: Ratios of pairwise detected keypoint distances compared against tactical model proportions; points with $>80\%$ error are pruned ([tactical_view.py:L54-L112](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/tactical_view.py#L54-L112)).
  - Homography: OpenCV `findHomography` with RANSAC (`ransacReprojThreshold=5.0`, `maxIters=2000`), determinant-based flip correction, and exponential smoothing with previous homography ($\alpha=0.85$) ([homography.py:L43-L115](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/homography.py#L43-L115)).
  - Tactical Smoothing & Sideline Filter: EMA position smoothing ($\alpha=0.6$) and sideline filter discarding entities outside $[-200, 3000] \times [-200, 1700]$ coordinate space ([court_mapper.py:L92-L113](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/court_mapper.py#L92-L113)).
- **Implementation Status:** **Fully Implemented**.
- **Input / Output:** Produces transformed coordinates and rendered minimap overlays.

#### F. Ball Possession, Passing & Interceptions
- **File Path:** [`core/basketball/ball_possession.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/ball_possession.py#L5-L163), [`core/basketball/pass_detector.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/pass_detector.py#L1-L76)
- **Model(s) & Algorithm(s):**
  - Possession: Evaluates intersection-over-ball-area containment ratio ($>0.80$) and Euclidean distance to 10 player boundary/alignment anchor points (threshold $= 120\text{px}$). Uses a 5-frame sliding window majority vote with a 3-frame grace period ([ball_possession.py:L18-L20, L67-L113, L126-L161](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/ball_possession.py#L18-L20)).
  - Passes & Interceptions: State transition analyzer; intra-team possession shifts register as passes; inter-team shifts register as interceptions ([pass_detector.py:L11-L75](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/pass_detector.py#L11-L75)).
- **Implementation Status:** **Fully Implemented**.

#### G. Speed & Distance Telemetry
- **File Path:** [`core/basketball/speed_calculator.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/speed_calculator.py#L4-L119)
- **Model(s) & Algorithm(s):** Tactical pixel-to-meter scaling ($28\text{m} \times 15\text{m}$ court) with a hardcoded $0.4\times$ noise-reduction scale factor. Speed computed in km/h via a 5-frame sliding window ([speed_calculator.py:L19-L23, L52, L72-L99](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/speed_calculator.py#L19-L23)).
- **Implementation Status:** **Fully Implemented**.

#### H. Parabolic Shot Detection
- **File Path:** [`core/basketball/shot_detector.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/shot_detector.py#L6-L132)
- **Model(s) & Algorithm(s):** 2nd-degree polynomial curve fitting ($y(t) = at^2 + bt + c$) via `np.polyfit` over a 45-frame rolling deque. Active shot arc requires $a > 0.5$ (upward-opening in image coordinates where $Y$ points downward) and $R^2 > 0.8$. Determines apex $t_{\text{apex}} = -b / (2a)$ and checks intersection against hardcoded left/right hoop bounding boxes to classify "MAKE" ($1.0$), "MISS" ($0.5$), or ascending/descending ($0.6$) ([shot_detector.py:L54-L107](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/shot_detector.py#L54-L107)).
- **Implementation Status:** **Fully Implemented** (algorithmic polynomial fit).
- **Constants & Hardware Constraints:** Hoop bounding regions hardcoded assuming $1920 \times 1080$ frame dimensions: left hoop `[50, 250, 250, 450]`, right hoop `[1650, 250, 1850, 450]` ([shot_detector.py:L21-L24](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/shot_detector.py#L21-L24)).

#### I. Highlights & Post-Game Analytics
- **File Path:** [`core/basketball/highlight_generator.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/highlight_generator.py#L10-L51), [`core/basketball/analytics_report.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/analytics_report.py#L9-L247)
- **Model(s) & Algorithm(s):**
  - Highlight Generator: MoviePy video slicing around shot confidence peaks ($\ge 0.5$) with buffer windows ($4\text{s}$ before, $2\text{s}$ after) ([highlight_generator.py:L15-L48](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/highlight_generator.py#L15-L48)).
  - Analytics Report: Exports structured JSON summaries and renders 2D Gaussian-blurred thermal heatmaps ($2800 \times 1500\text{ px}$, $\sigma=35$) with `cv2.COLORMAP_JET` overlaid onto court templates ([analytics_report.py:L116-L198, L200-L246](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/analytics_report.py#L116-L246)).
- **Implementation Status:** **Fully Implemented**.

---

### 3.3 Cricket Pipeline Components

```
                           Cricket Video
                                │
                                ▼
                     ┌─────────────────────┐
                     │   CricketDetector   │  YOLOv8 Ball + YOLOv8m ByteTrack Players
                     └──────────┬──────────┘  + Sobel Vertical Edge Stumps Calibration
                                │
                                ▼
                     ┌─────────────────────┐
                     │  CricketBallTracker │  4-State Ballistic Kalman Filter [x,y,vx,vy]
                     └──────────┬──────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
┌───────────────┐       ┌───────────────┐       ┌────────────────┐
│ PitchMapper   │       │Trajectory/Spd │       │DeliveryManager │
├───────────────┤       ├───────────────┤       ├────────────────┤
│ 22-yd Metric  │       │ 3D Ballistics │       │ In-Flight vs   │
│ Pitch Minimap │       │ Release/Bounce│       │ Dead-Time      │
│ Ground Radar  │       │ Stumps Proj.  │       │ Over Counting  │
└───────┬───────┘       └───────┬───────┘       └────────┬───────┘
        └───────────────────────┼────────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Delivery Lifecycle │  Triggered on ball completion:
                     ├─────────────────────┤
                     │ • DRSEngine (LBW)   │  ICC Law 36 (Pitching, Impact, Wickets)
                     │ • CricShot10        │  Kinematic exit vector classification
                     │ • Speed Gun Calc    │  True 3D speed (Release, Bounce, % Loss)
                     │ • Expected Dismissal│  xD threat index (0-100%)
                     │ • NoBallDetector    │  Front-foot crease margin (cm)
                     │ • Commentary Gen    │  Natural language ball-by-ball recap
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Outputs & Reporting │  Annotated Video + Multimodal Highlights
                     └─────────────────────┘  (Audio-Visual) + Stats JSON Report
```

#### A. Multi-Class Detection & Stumps Calibration
- **File Path:** [`core/cricket/detector.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L12-L311)
- **Model(s) & Algorithm(s):**
  - Ball Detector: Fine-tuned YOLOv8 (`CBDbest.pt` or `cricket_ball_best.pt`) preceded by CLAHE local contrast normalization on the HSV luminance channel ($V$) (`clipLimit=2.5, tileGridSize=(8,8)`). Includes motion-blur streak detection (aspect ratio $>1.3$ or $<0.77$) and kinematic distance gating ($<220\text{px}$) ([detector.py:L42-L72, L87-L96, L113-L158](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L42-L158)).
  - Player Detector: COCO `yolov8m.pt` tracking person class with Supervision `ByteTrack` (`track_activation_threshold=0.25`, `lost_track_buffer=30`) ([detector.py:L35-L40, L61-L67, L172-L198](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L35-L198)).
  - Stumps & Pads Localization: Vertical Sobel edge filter (`cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)`) computing horizontal gradient energy peaks to locate vertical stump poles behind the batsman's feet. Lower 48% of the batsman's bounding box is segmented as pads ([detector.py:L209-L251](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L209-L251)).
  - Role Assignment: Automatic temporal voting based on median $Y$ coordinates and movement (batsman, wicketkeeper, bowler) ([detector.py:L254-L311](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L254-L311)).
- **Implementation Status:** **Fully Implemented**.

#### B. Ballistic Kalman Tracking
- **File Path:** [`core/cricket/ball_tracker.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/ball_tracker.py#L3-L178)
- **Model(s) & Algorithm(s):** 4-State linear Kalman filter:
  $$\mathbf{x} = [x, y, v_x, v_y]^T, \quad \mathbf{z} = [x, y]^T$$
  Incorporates gravitational acceleration into state transition matrix $\mathbf{F}$. Rejects unphysical velocity jumps ($>150\text{px/frame}$) and extrapolates missing detections for up to 10 consecutive frames ([ball_tracker.py:L11-L88, L96-L177](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/ball_tracker.py#L11-L177)).
- **Implementation Status:** **Fully Implemented**.

#### C. Trajectory Analysis, Bounce Detection & 3D Extrapolation
- **File Path:** [`core/cricket/trajectory.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L33-L315)
- **Model(s) & Algorithm(s):**
  - 3D State Filter: 6-State 3D Kalman Filter $[X, Y, Z, v_x, v_y, v_z]^T$ with control input matrix $\mathbf{B}\mathbf{u}$ for gravity ($g=9.81\text{ m/s}^2$) ([trajectory.py:L33-L101](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L33-L101)).
  - Pitch Bounce Detection: Detects inflection points in the vertical trajectory ($dy_1 > 0, dy_2 \ge 0, dy_4 < dy_2$), computing incident and rebound angles and angular deflection ([trajectory.py:L151-L191](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L151-L191)).
  - Ballistic Stumps Extrapolation: Integrates forward trajectory from pad contact to the stumps plane at $Y = 20.12\text{ m}$ under gravity and aerodynamic drag ($\gamma = 0.0070$), computing the exact $(X, Z)$ coordinate at the stumps plane and estimating a 95% confidence error cone ([trajectory.py:L232-L287, L301-L315](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L232-L315)).
- **Implementation Status:** **Fully Implemented**.

#### D. Pitch Homography & Ground Radar Mapping
- **File Path:** [`core/cricket/pitch_mapper.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L7-L250)
- **Model(s) & Algorithm(s):**
  - Camera-to-Metric Homography: Planar perspective transformation mapping broadcast trapezoids to a $20.12\text{m} \times 3.05\text{m}$ coordinate space ([pitch_mapper.py:L39-L71, L96-L112](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L39-L112)).
  - 2D Pitch Minimap: Projects ball flight and bounce locations onto a $600 \times 320\text{ px}$ pitch template image (`data/cricket/cricket_pitch.png`) ([pitch_mapper.py:L114-L158](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L114-L158)).
  - Circular Ground Radar: Maps player positions and extended ball travel onto a top-down ground graphic ($538 \times 530\text{ px}$, $R=254\text{px}$) with boundary clamping ([pitch_mapper.py:L160-L205](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L160-L205)).
  - Line & Length Classification: Rule-based geometric categorization:
    - Length: Yorker ($<2\text{m}$), Full ($2\text{–}4\text{m}$), Good Length ($4\text{–}6.5\text{m}$), Short of Length ($6.5\text{–}8.5\text{m}$), Bouncer ($>8.5\text{m}$) ([pitch_mapper.py:L223-L232](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L223-L232)).
    - Line: Wide Off ($<-0.6\text{m}$), Outside Off ($-0.6\text{ to }-0.15\text{m}$), On Stumps ($-0.15\text{ to }+0.15\text{m}$), Leg Stump ($+0.15\text{ to }+0.6\text{m}$), Down Leg ($>+0.6\text{m}$) ([pitch_mapper.py:L234-L248](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L234-L248)).
- **Implementation Status:** **Fully Implemented**.

#### E. 3D Speed Gun Kinematics
- **File Path:** [`core/cricket/speed_calculator.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/speed_calculator.py#L4-L193)
- **Model(s) & Algorithm(s):** Reconstructs true 3D coordinates ($X, Y, Z$) by solving vertical projectile motion before bounce ($z(0) = 2.15\text{m}$) and applying turf coefficient of restitution ($e=0.62$). Computes spatial velocity magnitude $v = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2} / \Delta t$, enforcing physical monotonicity so that bounce speed never exceeds release speed ([speed_calculator.py:L22-L72, L74-L100, L101-L192](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/speed_calculator.py#L22-L192)).
- **Implementation Status:** **Fully Implemented**.

#### F. Batting Shot Action Recognition (CricShot10)
- **File Path:** [`core/cricket/shot_classifier.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/shot_classifier.py#L7-L108)
- **Model(s) & Algorithm(s):**
  - Deep Learning Path: Optional PyTorch weight loader for `models/cricket/shot_classifier.pt` ([shot_classifier.py:L37-L51](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/shot_classifier.py#L37-L51)).
  - Kinematic Fallback Path (Active): Classifies into the 10 CricShot10 classes based on post-impact ball exit velocity, elevation angle ($\theta = \arctan2(-\Delta y, |\Delta x|)$), and lateral deflection ($\Delta x$). Differentiates between defensive blocks, lofted shots, straight drives, off-side cuts/drives, and leg-side pulls/hooks/sweeps ([shot_classifier.py:L60-L108](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/shot_classifier.py#L60-L108)).
- **Implementation Status:** **Partially Implemented / Algorithmic Fallback Active** (`shot_classifier.pt` checkpoint does not exist; rule-based kinematic vector classifier runs deterministically).

#### G. Event Detection, ICC Law 36 DRS & Hawk-Eye
- **File Path:** [`core/cricket/event_detector.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/event_detector.py#L6-L179), [`core/cricket/drs_engine.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L68-L302)
- **Model(s) & Algorithm(s):**
  - Event Detector: Detects bat deflection via vector dot products ($dx_1 dx_2 + dy_1 dy_2 < 0$), pad impacts via pad bbox intersection, bowled wickets via stump proximity ($<40\text{px}$), and boundary 4s/6s via exit velocity ([event_detector.py:L80-L178](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/event_detector.py#L80-L178)).
  - DRS Engine: Implements ICC Law 36 LBW adjudication:
    - Pitching: In-Line, Outside Off, Outside Leg (strictly NOT OUT), Full Toss ([drs_engine.py:L140-L182](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L140-L182)).
    - Impact: In-Line, Umpire's Call ($7.2\text{cm}$ outer margin), Outside Off (factors in Law 36.1(e) shot-offered condition) ([drs_engine.py:L184-L226](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L184-L226)).
    - Wickets: Hitting, Umpire's Call (clipping bails or outer half-ball margin), Missing ([drs_engine.py:L228-L260](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L228-L260)).
    - ICC Regulations: Enforces the 3-meter distance rule ($20.12 - Y_{\text{pad}} \ge 3.0\text{m}$) and close proximity rule ($\Delta Y < 0.40\text{m}$) ([drs_engine.py:L262-L281](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L262-L281)).
    - Broadcast Overlay: `render_drs_banner()` generates an ICC-style 3-box overlay card ([drs_engine.py:L201-L302](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L201-L302)).
- **Implementation Status:** **Fully Implemented**.

#### H. Delivery Segmentation & Multi-Over Management
- **File Path:** [`core/cricket/delivery_manager.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/delivery_manager.py#L10-L126)
- **Model(s) & Algorithm(s):** State machine transitioning across `IDLE`, `IN_FLIGHT`, `COMPLETED`, and `DEAD_TIME`. Segments deliveries by requiring active ball flight with a batsman on screen, triggering delivery completion after 16 frames of lost tracking, and tracking overs ($0.1, \dots, 0.6 \to 1.1$) ([delivery_manager.py:L50-L113](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/delivery_manager.py#L50-L113)).
- **Implementation Status:** **Fully Implemented**.

#### I. Match Commentary & Statistical Reporting
- **File Path:** [`core/cricket/commentary_generator.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/commentary_generator.py#L4-L111), [`core/cricket/stats_extractor.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/stats_extractor.py#L21-L142), [`core/cricket/analytics_report.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_report.py#L8-L145)
- **Model(s) & Algorithm(s):**
  - Commentary: Dynamic natural language generation producing contextual ball-by-ball text and over summaries conditioned on speed, line, length, shot, outcome, and xD score ([commentary_generator.py:L15-L110](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/commentary_generator.py#L15-L110)).
  - Stats Extractor: Ingests delivery data and aggregates innings summaries ([stats_extractor.py:L43-L141](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/stats_extractor.py#L43-L141)).
  - Analytics Report: Exports complete match records to `data/output/cricket/stats/<video_name>.json` ([analytics_report.py:L74-L145](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_report.py#L74-L145)).
- **Implementation Status:** **Fully Implemented**.

#### J. Expected Dismissal (xD) Probability Engine
- **File Path:** [`core/cricket/analytics_engine.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_engine.py#L4-L130)
- **Model(s) & Algorithm(s):** Computes delivery dismissal probability $xD \in [0.02, 0.95]$ by combining base length weights (Good Length $= 0.42$, Yorker $= 0.38$), line multipliers (Corridor of Uncertainty $= 1.35\times$), pace multipliers ($>140\text{ km/h}$), seam deviation bonuses, and stump intersection bonuses ([analytics_engine.py:L47-L129](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_engine.py#L47-L129)).
- **Implementation Status:** **Fully Implemented**.

#### K. Umpiring Aids (No-Ball & Snicko Simulation)
- **File Path:** [`core/cricket/umpiring.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/umpiring.py#L30-L125)
- **Model(s) & Algorithm(s):**
  - NoBallDetector: Compares front foot landing $Y$ against the popping crease ($Y = 1.22\text{m}$), returning overstep margin in cm ([umpiring.py:L30-L63](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/umpiring.py#L30-L63)).
  - UltraEdge Simulator: Synthesizes acoustic time-series waveforms ($1000\text{ Hz}$) with high-frequency resonant spikes ($320\text{ Hz}$ for bat edge, $90\text{ Hz}$ for pad impact, $180\text{ Hz}$ for glove) ([umpiring.py:L65-L125](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/umpiring.py#L65-L125)).
- **Implementation Status:** **Fully Implemented**.

#### L. Multimodal Audio-Visual Highlight Generation
- **File Path:** [`core/cricket/audio_processor.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/audio_processor.py#L5-L105), [`core/cricket/highlight_generator.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/highlight_generator.py#L7-L153)
- **Model(s) & Algorithm(s):** Extracts video soundtrack at $22,050\text{ Hz}$ via MoviePy, computes short-time RMS energy ($0.25\text{s}$ windows), detects crowd roars ($\text{energy} > \mu + 2\sigma$ with $2.0\text{s}$ minimum separation), and fuses them with visual boundary/wicket timestamps within a $2.5\text{s}$ temporal tolerance window to render highlight reels ([audio_processor.py:L31-L95](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/audio_processor.py#L31-L95), [highlight_generator.py:L20-L58, L60-L113](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/highlight_generator.py#L20-L113)).
- **Implementation Status:** **Fully Implemented**.

---

## 4. Pipeline Execution & Data Flow Overview

### 4.1 Master Entrypoint (`main_pipeline.py`)

Execution flow is governed by [`main_pipeline.py`](file:///c:/important%20files/main%20files/projects/SportsVision/main_pipeline.py#L24-L74):
1. **CUDA Dynamic Initialization:** Searches system paths and `CUDA_PATH` on Windows to link NVIDIA DLLs ([main_pipeline.py:L4-L20](file:///c:/important%20files/main%20files/projects/SportsVision/main_pipeline.py#L4-L20)).
2. **Sport Routing:** If `--sport auto` (default), calls `detect_sport(video_path)` ([main_pipeline.py:L44-L53](file:///c:/important%20files/main%20files/projects/SportsVision/main_pipeline.py#L44-L53)).
3. **Dispatch:**
   - If `"cricket"`: calls `core.cricket.pipeline.run_cricket_pipeline(...)` ([main_pipeline.py:L56-L64](file:///c:/important%20files/main%20files/projects/SportsVision/main_pipeline.py#L56-L64)).
   - Else (`"basketball"`): calls `core.basketball.pipeline.run_basketball_pipeline(...)` ([main_pipeline.py:L65-L73](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/pipeline.py#L65-L73)).

---

### 4.2 Basketball Execution Pipeline (8 Stages)

```
[1/8] Initializing Hardware (verify_cuda)
  │
[2/8] Initializing AI Engines (CourtVisionDetector, SAM2, TeamClassifier, CourtMapper, etc.)
  │
[3/8] Opening Video Stream & Initializing Checkpointing
  │
[4/8] Streaming Batch Processing Loop (Batch size = 250 frames)
  ├── Stage A: Dual-model Tracking (YOLOv8m BoT-SORT + basketball_best.pt Ball)
  ├── Stage A.1: SAM2 Instance Mask Segmentation
  ├── Stage B: K-Means / SigLIP Jersey Color Clustering & Team Assignment
  ├── Stage B.1: EasyOCR Player Number Identification
  ├── Stage C: Court Keypoint Striding CNN & RANSAC Homography Mapping
  ├── Stage D: Proximity & Bounding Box Ball Possession Detection
  ├── Stage E: Intra-Team Pass & Defensive Interception Detection
  ├── Stage F: Metric Speed (km/h) & Distance Calculation
  ├── Stage G: Threaded Broadcast Graphics Drawing (Ellipses, Badges, HUDs)
  ├── Stage H: 2D Tactical Minimap Overlay
  ├── Stage I: Parabolic 2nd-degree Polynomial Shot Arc Fitting (Make/Miss)
  └── Stage J: Telemetry & Statistical Accumulation
  │
[5/8] Video Processing Complete (Annotated MP4 Output)
  │
[6/8] Shot Detection Summary
  │
[7/8] MoviePy Highlight Video Compilation
  │
[8/8] Analytics JSON Report & Thermal Heatmap Export (2800x1500 Jet Colormap)
```

---

### 4.3 Cricket Execution Pipeline (7 Stages)

```
[1/7] Initializing Hardware (verify_cuda)
  │
[2/7] Initializing Cricket Analytics Engines (CricketDetector, Kalman, Trajectory, DRS, etc.)
  │
[3/7] Opening Video Source & Initializing Delivery Lifecycle Manager
  │
[4/7] Streaming Batch Processing Loop (Batch size = 250 frames)
  ├── A. Detection: YOLOv8 Ball (CLAHE + Streak) + YOLOv8m Players + Sobel Stumps
  ├── B. Ball Tracking: 4-State Ballistic Kalman Filter ([x,y,vx,vy] + Gravity)
  ├── C. Lifecycle Update: DeliveryLifecycleManager (Active Flight vs. Dead Time)
  ├── D. Spatial Mapping: Camera to 22-yard Pitch Minimap & Circular Ground Radar
  ├── E. Rendering: Live Speedometer Badge, Hawkeye Trail, Landing Rings, Player Roles
  └── F. Delivery Finalization (Invoked per completed ball):
         ├── Trajectory Inflection Bounce Analysis (Incident & Rebound Angles)
         ├── 3D Ballistic Projection through Stumps Plane (Y = 20.12m)
         ├── ICC Law 36 DRS LBW Adjudication (Pitching, Impact, Wickets, 3m Rule)
         ├── True 3D Delivery Speeds (Release, Bounce, % Deceleration)
         ├── Expected Dismissal (xD) Probability Threat Index
         ├── CricShot10 Kinematic Batting Stroke Classification
         ├── Bowler Front-Foot Crease No-Ball Evaluation
         ├── Natural Language Ball-by-Ball Commentary Generation
         └── Stats Telemetry Recording & DRS Banner Trigger
  │
[5/7] Video Processing Complete (Annotated MP4 Output)
  │
[6/7] Multimodal Audio-Visual Highlight Reel Extraction (MoviePy / OpenCV)
  │
[7/7] Cricket Match Analytics & Commentary JSON Report Export
```

---

## 5. Progress, Implementation Maturity & Discrepancies

### 5.1 Pipeline Stage Status Matrix

| Subsystem | Pipeline Stage | Code Location | Primary Model / Algorithm | Maturity Status |
| :--- | :--- | :--- | :--- | :--- |
| **Router** | Frame Sampling | [`frame_sampler.py:L12-L42`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/frame_sampler.py#L12-L42) | Linspace temporal sampling (10%–90%) | **Complete** |
| **Router** | Chromatic Classification | [`classifier.py:L43-L70`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L43-L70) | HSV turf/wood pixel ratio scoring | **Complete** |
| **Router** | Zero-Shot Embedding | [`classifier.py:L72-L104`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L72-L104) | `google/siglip-base-patch16-224` | **Complete** |
| **Router** | Majority Dispatch | [`router.py:L17-L51`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/router.py#L17-L51) | Majority vote + confidence threshold | **Complete** |
| **Basketball** | Player Detection & Tracking | [`detector.py:L104-L148`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L104-L148) | YOLOv8m COCO + BoT-SORT | **Complete** |
| **Basketball** | Ball & Referee Detection | [`detector.py:L152-L200`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L152-L200) | `basketball_best.pt` + spatial voting | **Complete** |
| **Basketball** | Ball Trajectory Smoothing | [`detector.py:L202-L254`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L202-L254) | `BallTracker` / EMA fallback ($\alpha=0.5$) | **Complete** |
| **Basketball** | Player Instance Masks | [`segmentation.py:L19-L56`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/segmentation.py#L19-L56) | SAM2.1 Small (`sam2.1_s.pt`) | **Complete** |
| **Basketball** | Jersey Color Discovery | [`team_classifier.py:L51-L107`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/team_classifier.py#L51-L107) | K-Means ($k=2$) or SigLIP + UMAP | **Complete** |
| **Basketball** | Jersey Number Recognition | [`player_id.py:L17-L78`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/player_id.py#L17-L78) | EasyOCR + Unsharp Masking + Voting | **Complete** |
| **Basketball** | Court Keypoint Detection | [`court_mapper.py:L38-L71`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/court_mapper.py#L38-L71) | `court_keypoint_detector.pt` with stride | **Complete** |
| **Basketball** | Court Homography | [`homography.py:L43-L115`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/homography.py#L43-L115) | OpenCV RANSAC + Flip Fix + Blending | **Complete** |
| **Basketball** | Tactical Minimap Rendering | [`tactical_view_drawer.py:L79-L160`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/tactical_view_drawer.py#L79-L160) | 2D court image overlay + interpolation | **Complete** |
| **Basketball** | Ball Possession Detection | [`ball_possession.py:L115-L163`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/ball_possession.py#L115-L163) | Containment ratio + Anchor proximity | **Complete** |
| **Basketball** | Passes & Interceptions | [`pass_detector.py:L11-L75`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/pass_detector.py#L11-L75) | Possession state-machine transitions | **Complete** |
| **Basketball** | Speed & Distance Telemetry | [`speed_calculator.py:L29-L99`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/speed_calculator.py#L29-L99) | Tactical coordinate metric scaling | **Complete** |
| **Basketball** | Shot Detection (Make/Miss) | [`shot_detector.py:L32-L128`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/shot_detector.py#L32-L128) | 2nd-degree polynomial ($y=at^2+bt+c$) | **Complete** |
| **Basketball** | Broadcast Graphics | [`visualizer.py:L213-L292`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/visualizer.py#L213-L292) | Feet ellipses, HUDs, possession badges | **Complete** |
| **Basketball** | Highlight Video Extraction | [`highlight_generator.py:L17-L50`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/highlight_generator.py#L17-L50) | MoviePy subclip slicing ($\ge 0.5$ conf) | **Complete** |
| **Basketball** | JSON Report & Heatmaps | [`analytics_report.py:L116-L246`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/analytics_report.py#L116-L246) | JSON generation + 2800x1500 Jet heatmap | **Complete** |
| **Cricket** | Ball Detection (YOLOv8) | [`detector.py:L111-L170`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L111-L170) | `CBDbest.pt` + CLAHE + Streak AR | **Complete** |
| **Cricket** | Player Tracking & Roles | [`detector.py:L172-L198, L254-L311`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L172-L311) | YOLOv8m + ByteTrack + Spatial voting | **Complete** |
| **Cricket** | Stumps & Pads Localization | [`detector.py:L209-L251`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L209-L251) | Sobel vertical gradient edge energy | **Complete** |
| **Cricket** | Ballistic Kalman Tracking | [`ball_tracker.py:L105-L177`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/ball_tracker.py#L105-L177) | 4-state $[x, y, v_x, v_y]^T$ + Gravity | **Complete** |
| **Cricket** | 3D Trajectory & Bounce | [`trajectory.py:L151-L191`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L151-L191) | Trajectory inflection + incident angles | **Complete** |
| **Cricket** | 3D Stumps Extrapolation | [`trajectory.py:L232-L287`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L232-L287) | Ballistic forward integration with drag | **Complete** |
| **Cricket** | Pitch & Ground Homography | [`pitch_mapper.py:L39-L158, L160-L205`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L39-L205) | 22-yd metric + minimap + ground radar | **Complete** |
| **Cricket** | Metric Line & Length | [`pitch_mapper.py:L207-L250`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L207-L250) | Geometric distance & lateral offsets | **Complete** |
| **Cricket** | 3D Speed Gun Kinematics | [`speed_calculator.py:L101-L192`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/speed_calculator.py#L101-L192) | True 3D spatial velocity + restitution | **Complete** |
| **Cricket** | CricShot10 Classification | [`shot_classifier.py:L52-L108`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/shot_classifier.py#L52-L108) | Kinematic exit vector analysis | **Partial (Heuristic)** |
| **Cricket** | Match Event Detection | [`event_detector.py:L21-L178`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/event_detector.py#L21-L178) | Trajectory deflection + boundaries | **Complete** |
| **Cricket** | ICC Law 36 DRS Engine | [`drs_engine.py:L68-L302`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L68-L302) | Pitching/Impact/Wickets + 3m rule | **Complete** |
| **Cricket** | Multi-Over Lifecycle | [`delivery_manager.py:L50-L125`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/delivery_manager.py#L50-L125) | 4-state delivery lifecycle machine | **Complete** |
| **Cricket** | Expected Dismissal (xD) | [`analytics_engine.py:L47-L129`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_engine.py#L47-L129) | Weighted threat index (0-100%) | **Complete** |
| **Cricket** | No-Ball Detection | [`umpiring.py:L30-L63`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/umpiring.py#L30-L63) | Popping crease ($1.22\text{m}$) overstep | **Complete** |
| **Cricket** | Snicko / UltraEdge Sim | [`umpiring.py:L65-L125`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/umpiring.py#L65-L125) | Synthetic acoustic harmonic spikes | **Complete** |
| **Cricket** | Automated Commentary | [`commentary_generator.py:L15-L110`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/commentary_generator.py#L15-L110) | Natural language telemetry templates | **Complete** |
| **Cricket** | Audio Energy Extraction | [`audio_processor.py:L17-L104`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/audio_processor.py#L17-L104) | MoviePy 22kHz RMS short-time energy | **Complete** |
| **Cricket** | Multimodal Highlights | [`highlight_generator.py:L20-L113`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/highlight_generator.py#L20-L113) | Audio-visual co-occurrence fusion | **Complete** |
| **Cricket** | Match Statistics Export | [`analytics_report.py:L74-L145`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_report.py#L74-L145) | Structured JSON export + console recap | **Complete** |
| **Web Studio** | FastAPI Backend | [`app.py:L31-L300`](file:///c:/important%20files/main%20files/projects/SportsVision/app.py#L31-L300) | Pipeline subprocess execution & SSE | **Complete** |
| **Web Studio** | Spatial UI Frontend | `web/static/` | HTML5, Vanilla CSS, Chart.js, SSE | **Complete** |

---

### 5.2 Discrepancies Between Stated Design and Actual Implementation

The following concrete discrepancies exist between design documents (`PLAN.md`, `README.md`, docstrings) and actual codebase implementation:

1. **Router Geometric Analysis Omission:**
   - *Stated Design:* [`core/sport_router/classifier.py:L10-L12`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L10-L12) docstring specifies "2. Line & Court Geometry: Basketball courts feature concentric arcs... Cricket fields feature a central elongated pitch rectangle".
   - *Actual Implementation:* No line or geometric edge analysis exists in `classify_frame()`. Only chromatic surface distribution (HSV grass/wood ratios) and zero-shot SigLIP are implemented ([classifier.py:L43-L111](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L43-L111)).

2. **Absence of Shared Batching Module:**
   - *Stated Design:* [`PLAN.md:L64`](file:///c:/important%20files/main%20files/projects/SportsVision/PLAN.md#L64) lists `core/shared/batching.py` for generic frame-batch iteration.
   - *Actual Implementation:* `core/shared/batching.py` does not exist in the codebase. Frame batching loops are implemented independently within [`core/basketball/pipeline.py:L124-L142`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/pipeline.py#L124-L142) and [`core/cricket/pipeline.py:L277-L295`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pipeline.py#L277-L295).

3. **CricShot10 Learned Checkpoint vs. Kinematic Fallback:**
   - *Stated Design:* [`PLAN.md:L76`](file:///c:/important%20files/main%20files/projects/SportsVision/PLAN.md#L76) and [`core/cricket/shot_classifier.py:L40`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/shot_classifier.py#L40) describe loading `models/cricket/shot_classifier.pt`.
   - *Actual Implementation:* The file `shot_classifier.pt` does not exist in the repository. The classifier permanently executes its rule-based kinematic exit-vector fallback ([shot_classifier.py:L60-L108](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/shot_classifier.py#L60-L108)).

4. **Cricket Pitch Keypoint CNN vs. Perspective Homography:**
   - *Stated Design:* [`PLAN.md:L75`](file:///c:/important%20files/main%20files/projects/SportsVision/PLAN.md#L75) lists `models/cricket/pitch_keypoint_detector.pt` for automatic pitch landmark detection.
   - *Actual Implementation:* No pitch keypoint neural network is present. Pitch mapping in [`core/cricket/pitch_mapper.py:L39-L71`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pitch_mapper.py#L39-L71) relies on a default nominal broadcast trapezoid with manual corner calibration support.

5. **Basketball Highlight "ResNet" Docstring Relic:**
   - *Stated Design:* [`core/basketball/highlight_generator.py:L18`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/highlight_generator.py#L18) logs: `print("Extracting highlights based on ResNet shot confidences...")`.
   - *Actual Implementation:* Shot confidences are produced by 2nd-degree polynomial curve fitting in [`core/basketball/shot_detector.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/shot_detector.py#L54-L107), not a ResNet architecture.

6. **Basketball Shot Detector Resolution Mismatch:**
   - *Stated Design:* [`core/basketball/shot_detector.py:L20-L24`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/shot_detector.py#L20-L24) defines hoop regions assuming $1920 \times 1080$ frame dimensions: left hoop `[50, 250, 250, 450]`, right hoop `[1650, 250, 1850, 450]`.
   - *Actual Implementation:* [`config.py:L55`](file:///c:/important%20files/main%20files/projects/SportsVision/config.py#L55) sets `OUTPUT_RESOLUTION = (1280, 720)`, and [`core/basketball/pipeline.py:L137`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/pipeline.py#L137) resizes incoming video frames to $1280 \times 720$. Consequently, the right hoop coordinate ($X \ge 1650$) falls outside the resized image boundary, causing right-basket shots to be misclassified as misses.

7. **Basketball Speed Calculator vs. Tactical Minimap Dimensions:**
   - *Stated Design:* [`core/basketball/speed_calculator.py:L13`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/speed_calculator.py#L13) initializes with `width_px=300, height_px=161`.
   - *Actual Implementation:* [`core/basketball/minimap/tactical_view.py:L17-L18`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/tactical_view.py#L17-L18) generates tactical positions mapped to `width = 200, height = 107`. Furthermore, `speed_calculator.py:L52` applies an arbitrary $0.4\times$ multiplier to distance measurements.

8. **Strict GPU Requirement vs. Documented CPU Fallback:**
   - *Stated Design:* [`REQUIREMENTS.md:L18-L20`](file:///c:/important%20files/main%20files/projects/SportsVision/REQUIREMENTS.md#L18-L20) states: "SportsVision natively supports running on CPU-only machines. If an NVIDIA GPU is not detected, the system will automatically fall back to CPU execution."
   - *Actual Implementation:* Both [`core/basketball/pipeline.py:L37`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/pipeline.py#L37) and [`core/cricket/pipeline.py:L199`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pipeline.py#L199) invoke `verify_cuda()`, which executes `sys.exit(1)` if `torch.cuda.is_available()` is `False` ([gpu_diagnostics.py:L8-L10](file:///c:/important%20files/main%20files/projects/SportsVision/core/shared/gpu_diagnostics.py#L8-L10)).

9. **Web Studio Single-Sport Regex Parsing:**
   - *Stated Design:* [`app.py`](file:///c:/important%20files/main%20files/projects/SportsVision/app.py) was updated to serve the Web Studio frontend for the project.
   - *Actual Implementation:* Title is hardcoded as `"CourtVision Basketball Spatial Analytics Engine"` ([app.py:L31](file:///c:/important%20files/main%20files/projects/SportsVision/app.py#L31)), and stage regex parsers ([app.py:L108-L140](file:///c:/important%20files/main%20files/projects/SportsVision/app.py#L108-L140)) match only the 8 basketball stages (`[1/8]` through `[8/8]`), failing to update stage badges when the 7-stage cricket pipeline runs.

---

## 6. Technical Context & System Constraints

### 6.1 Full Tech Stack & Dependencies

| Library / Tool | Specified Version | Purpose in Codebase |
| :--- | :--- | :--- |
| **Python** | `>= 3.10, < 3.13` | Runtime environment (tested stable on 3.10–3.12). |
| **PyTorch & Torchvision** | `torch >= 2.1.0`, `torchvision >= 0.16.0` | Deep learning backend, tensor operations, CUDA FP16 inference. |
| **Ultralytics** | `>= 8.1.0` | YOLOv8 object detection, keypoint pose detection, SAM2 wrapper. |
| **Supervision** | `>= 0.19.0` | ByteTrack player tracking and ball detection abstractions. |
| **OpenCV** | `opencv-python >= 4.8.0` | Video frame decoding/encoding, RANSAC homography, CLAHE, Sobel filters. |
| **Transformers** | `>= 4.30.0` | `google/siglip-base-patch16-224` vision-language model for sport routing. |
| **EasyOCR** | `>= 1.7.0` | CRAFT + PyTorch text recognition for player jersey numbers. |
| **MoviePy** | `>= 1.0.3` | Video subclip slicing, highlight concatenation, 22kHz audio extraction. |
| **Scikit-learn** | `>= 1.3.0` | K-Means clustering for dynamic team jersey color discovery. |
| **SciPy** | `>= 1.11.0` | Peak finding, numerical signal processing, scientific functions. |
| **UMAP-learn** | `>= 0.5.3` | Non-linear dimensionality reduction for SigLIP player embeddings. |
| **Optuna** | `>= 3.0.0` | Bayesian hyperparameter optimization for YOLO training. |
| **FastAPI & Uvicorn** | `fastapi >= 0.110.0`, `uvicorn >= 0.28.0` | Web Studio backend server and SSE real-time streaming. |

---

### 6.2 Hardware Constraints & Memory Tuning

All execution parameters in [`config.py`](file:///c:/important%20files/main%20files/projects/SportsVision/config.py) and pipeline runners are calibrated for an **NVIDIA GeForce RTX 4060 Mobile / Laptop GPU (8GB VRAM)**:
- **Batch Processing:** Videos are processed in windows of `BATCH_SIZE = 250` frames ([config.py:L51](file:///c:/important%20files/main%20files/projects/SportsVision/config.py#L51)). This bounds GPU tensor allocations while preventing unbounded RAM growth during full-game videos ($30\text{–}60\text{ minutes}$).
- **Precision:** `USE_FP16 = True` ([config.py:L21](file:///c:/important%20files/main%20files/projects/SportsVision/config.py#L21)), converting model weights to half-precision (`float16`) to halve VRAM bandwidth and footprint.
- **Keypoint Striding:** `COURT_KEYPOINT_STRIDE = 5` ([config.py:L75](file:///c:/important%20files/main%20files/projects/SportsVision/config.py#L75)), running the court pose network only once every 5 frames and interpolating intermediate frames to reduce compute by $\approx 80\%$.
- **Explicit Garbage Collection:** Pipelines invoke `torch.cuda.empty_cache()` between batch iterations and after keypoint inference ([pipeline.py:L448](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/pipeline.py#L448), [court_mapper.py:L69](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/court_mapper.py#L69)).

---

### 6.3 External Reference Repositories & Datasets Catalog

The following external repositories and datasets are documented or cited across codebase comments, training configs, and `PLAN.md`:

| Source / Repository | Original Author / Source | Usage in SportsVision | Integration Type |
| :--- | :--- | :--- | :--- |
| `Cricket-Ball-Trajectory-Prediction` | `kushagra3204` | YOLOv8 ball tracking dataset and bounce-point angle detection math. | **Adapted:** Math rewritten into `core/cricket/trajectory.py`. |
| `cricket-ball-detection-and-tracking` | `mig9mili` | Kalman filter state estimation math for missing ball detections. | **Adapted:** Converted to 4-state $[x, y, v_x, v_y]^T$ filter in `core/cricket/ball_tracker.py`. |
| `cricket_computer_vision_sports` | `siddharthksah` | Multi-class scheme and Optuna hyperparameter configuration. | **Adapted:** Reused in `training/optuna_tuning_config.yaml`. |
| `CricShot10` | `ascuet` | 10-class batting stroke action recognition taxonomy. | **Adapted:** Taxonomy utilized in `core/cricket/shot_classifier.py`. |
| `computer-vision-football-analysis` | `AhmedQassemDev2004` | Camera motion compensation and homography patterns. | **Reference Only:** Studied for homography structure. |
| `Cricket-Ball-and-Stumps-Detection` | `sanjusabu` | Vertical Sobel filter gradient analysis for stumps localization. | **Adapted:** Implemented in `core/cricket/detector.py:L209-L251`. |
| `lbw_drs_ai` | `dschandra` | 3-Class DRS detection scheme (`[0: ball, 1: stump, 2: pad]`). | **Adapted:** Schema documented in `core/cricket/detector.py:L25-L30`. |
| `cricket-ball-tracking` | `sathviknalla` | ICC Law 36 LBW logic and 3D Hawkeye projection principles. | **Adapted & Refined:** Fully implemented in `core/cricket/drs_engine.py`. |
| `High-Spot--Cricket-Highlights-Generator` | `vijay-2012` | Short-time RMS acoustic energy thresholding for cricket highlights. | **Adapted:** Implemented in `core/cricket/audio_processor.py`. |
| `Analysis-of-Cricket-Commentary` | `Psychellic` | Commentary phrase generation structures and narrative flow. | **Adapted:** Implemented in `core/cricket/commentary_generator.py`. |
| `CBDbest.pt` Weights | `saral7293` | Lightweight fine-tuned YOLOv8 cricket ball detector weights. | **Directly Harvested:** Saved at `models/cricket/CBDbest.pt`. |

---

## 7. Verifiable Implementation Claims & Line References

To ensure absolute rigor and prevent hallucination in research reporting, every technical claim made in this document maps directly to verified line ranges in the source tree:

1. **Hardware Enforcements & FP16 Execution:**
   - CUDA device selection: [`config.py:L20-L21`](file:///c:/important%20files/main%20files/projects/SportsVision/config.py#L20-L21)
   - CUDA check and mandatory halt: [`core/shared/gpu_diagnostics.py:L8-L10`](file:///c:/important%20files/main%20files/projects/SportsVision/core/shared/gpu_diagnostics.py#L8-L10)
   - Dynamic Windows DLL discovery: [`main_pipeline.py:L4-L20`](file:///c:/important%20files/main%20files/projects/SportsVision/main_pipeline.py#L4-L20)
2. **Sport Routing Mechanism:**
   - 5-frame linspace extraction: [`core/sport_router/frame_sampler.py:L29-L31`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/frame_sampler.py#L29-L31)
   - Grass HSV mask `[32, 40, 40]` to `[85, 255, 255]`: [`core/sport_router/classifier.py:L49-L52`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L49-L52)
   - Hardwood HSV mask `[8, 60, 60]` to `[28, 255, 240]`: [`core/sport_router/classifier.py:L55-L58`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L55-L58)
   - SigLIP model loading and zero-shot scoring: [`core/sport_router/classifier.py:L23-L33, L72-L104`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/classifier.py#L23-L104)
   - Majority voting decision: [`core/sport_router/router.py:L38-L50`](file:///c:/important%20files/main%20files/projects/SportsVision/core/sport_router/router.py#L38-L50)
3. **Basketball Pipeline Verification:**
   - Dual-model loading: [`core/basketball/detector.py:L38-L54`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L38-L54)
   - BoT-SORT person tracking & crowd filter: [`core/basketball/detector.py:L123-L148`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L123-L148)
   - Referee voting distance threshold ($< 1.5\%$ frame width): [`core/basketball/detector.py:L177-L195`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/detector.py#L177-L195)
   - SAM2.1 instance mask inference: [`core/basketball/segmentation.py:L38-L56`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/segmentation.py#L38-L56)
   - K-Means jersey color discovery: [`core/basketball/team_classifier.py:L51-L70`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/team_classifier.py#L51-L70)
   - EasyOCR unsharp masking & score voting: [`core/basketball/player_id.py:L50-L76`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/player_id.py#L50-L76)
   - Court keypoint CNN striding: [`core/basketball/court_mapper.py:L38-L71`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/court_mapper.py#L38-L71)
   - RANSAC homography flip correction: [`core/basketball/minimap/homography.py:L4-L12, L99`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/minimap/homography.py#L4-L99)
   - Possession containment ratio ($>0.80$) & proximity ($<120\text{px}$): [`core/basketball/ball_possession.py:L18-L20, L67-L113`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/ball_possession.py#L18-L113)
   - Parabolic shot polynomial fit ($a > 0.5, R^2 > 0.8$): [`core/basketball/shot_detector.py:L64-L107`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/shot_detector.py#L64-L107)
   - Thermal heatmap generation ($2800 \times 1500$, Gaussian $\sigma=35$): [`core/basketball/analytics_report.py:L200-L246`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/analytics_report.py#L200-L246)
4. **Cricket Pipeline Verification:**
   - CLAHE luminance enhancement & streak detection: [`core/cricket/detector.py:L87-L96, L139-L153`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L87-L153)
   - Sobel vertical edge filter for stumps: [`core/cricket/detector.py:L236-L250`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/detector.py#L236-L250)
   - 4-State ballistic Kalman filter: [`core/cricket/ball_tracker.py:L20-L88`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/ball_tracker.py#L20-L88)
   - 3D flight reconstruction with turf restitution ($e=0.62$): [`core/cricket/speed_calculator.py:L49-L72`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/speed_calculator.py#L49-L72)
   - Trajectory bounce inflection detection: [`core/cricket/trajectory.py:L151-L191`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L151-L191)
   - Ballistic stumps extrapolation with air drag ($0.0070$): [`core/cricket/trajectory.py:L232-L287`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/trajectory.py#L232-L287)
   - ICC Law 36 LBW adjudication & 3-meter rule: [`core/cricket/drs_engine.py:L106-L281`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/drs_engine.py#L106-L281)
   - Delivery lifecycle segmentation (16 lost frames threshold): [`core/cricket/delivery_manager.py:L69-L113`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/delivery_manager.py#L69-L113)
   - Expected Dismissal (xD) score calculation: [`core/cricket/analytics_engine.py:L47-L129`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_engine.py#L47-L129)
   - Front-foot popping crease check ($1.22\text{m}$): [`core/cricket/umpiring.py:L35-L63`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/umpiring.py#L35-L63)
   - MoviePy 22kHz RMS acoustic energy peak detection: [`core/cricket/audio_processor.py:L31-L95`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/audio_processor.py#L31-L95)
   - Multimodal audio-visual highlight fusion: [`core/cricket/highlight_generator.py:L20-L58`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/highlight_generator.py#L20-L58)
   - JSON report schema generation: [`core/cricket/analytics_report.py:L94-L130`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_report.py#L94-L130)
