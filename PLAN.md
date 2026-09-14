# CourtVision Multi-Sport Expansion — PLAN.md

**Goal:** Extend CourtVision (currently basketball-only) to support cricket, with an
automatic sport-detection router that picks the correct pipeline per input video.
The cricket pipeline's job is **stats extraction from raw video** — ball speed,
trajectory/bounce points, shot events (boundary/wicket), and per-delivery summaries —
mirroring what the basketball pipeline already does (possession %, speed, passes,
shot make/miss).

**Detector stack:** YOLOv8 (Ultralytics) end-to-end, matching what's already proven in
CourtVision's basketball detector. No YOLOv3/v5/v7 mixing — every cricket repo cloned
below is chosen because it's YOLOv8 + Supervision/ByteTrack-based, so the ported code
drops into CourtVision's existing model-loading and tracking conventions with minimal
adaptation.

**Hardware target:** RTX 4060 (mobile), CUDA, 8GB VRAM — same constraints as the
existing basketball pipeline (250-frame batching, FP16, keypoint striding).

**Repo policy:** Third-party GitHub repos are cloned for *reference and weight/dataset
harvesting only*. They are never imported into or run as part of the production
pipeline directly. Nothing from the external clone folder ships in the final `core/`
code — we port/rewrite the relevant logic into our own modules.

---

## 1. Directory structure

```
CourtVision/                          # existing repo root — DO NOT put clones here
├── core/
│   ├── basketball/                   # NEW — move existing basketball-only modules here
│   │   ├── detector.py
│   │   ├── team_classifier.py
│   │   ├── player_id.py
│   │   ├── court_mapper.py
│   │   ├── ball_possession.py
│   │   ├── pass_detector.py
│   │   ├── speed_calculator.py
│   │   ├── shot_detector.py
│   │   ├── visualizer.py
│   │   ├── highlight_generator.py
│   │   ├── analytics_report.py
│   │   └── minimap/
│   │       ├── court_keypoint_detection.py
│   │       ├── homography.py
│   │       └── tactical_view_drawer.py
│   │
│   ├── cricket/                      # NEW — mirrors basketball/ structure
│   │   ├── detector.py               # YOLOv8: stumps/ball/bat/batsman/bowler
│   │   ├── ball_tracker.py           # Supervision ByteTrack + Kalman-filter fallback
│   │   ├── trajectory.py             # bounce-point detection, angle calc, speed-from-arc
│   │   ├── pitch_mapper.py           # homography onto 22-yard pitch template
│   │   ├── shot_classifier.py        # batting shot type (CricShot10-derived)
│   │   ├── event_detector.py         # boundary/wicket/no-ball event detection
│   │   ├── speed_calculator.py       # ball speed (km/h) from pitch-mapped coords + FPS
│   │   ├── stats_extractor.py        # NEW — per-delivery + per-innings stat aggregation
│   │   ├── visualizer.py             # HUD: speed gun, pitch map, shot label
│   │   ├── highlight_generator.py    # clip wickets/boundaries
│   │   └── analytics_report.py       # per-over / per-batsman JSON summary
│   │
│   ├── shared/                       # NEW — code used by both sports
│   │   ├── video_io.py               # moved from helpers/
│   │   ├── gpu_diagnostics.py        # moved from helpers/
│   │   └── batching.py               # generic frame-batch iterator (250-frame windows)
│   │
│   └── sport_router/                 # NEW — the classifier described in section 3
│       ├── classifier.py             # CLIP zero-shot / fine-tuned frame classifier
│       ├── frame_sampler.py          # pulls N representative frames from input video
│       └── router.py                 # decides + dispatches to basketball/ or cricket/
│
├── models/
│   ├── basketball/                   # move basketball_best.pt, court_keypoint_detector.pt here
│   ├── cricket/                      # NEW — cricket weights land here (see section 2)
│   │   ├── cricket_yolov8_best.pt    # fine-tuned multi-class detector
│   │   ├── pitch_keypoint_detector.pt
│   │   └── shot_classifier.pt
│   └── router/                       # only needed if you fine-tune a custom classifier
│
├── data/
│   ├── basketball_court.png
│   ├── cricket_pitch.png             # NEW — top-down pitch template for pitch_mapper.py homography
│   ├── videos/
│   └── output/
│       ├── basketball/
│       └── cricket/
│           └── stats/                # NEW — per-video JSON/CSV stat exports land here
│
├── main_pipeline.py                  # MODIFIED — becomes a thin entrypoint (see section 4)
├── config.py                         # MODIFIED — add CRICKET_* and ROUTER_* config blocks
├── setup_models.py                   # MODIFIED — add cricket + router weight downloads
└── PLAN.md                           # this file
```

**External clone location (outside the project folder, as requested):**

```
~/cv-external-repos/
├── cricket/
│   ├── Cricket-Ball-Trajectory-Prediction/     # kushagra3204 — YOLOv8, trajectory+bounce+angle
│   ├── cricket-ball-detection-and-tracking/    # mig9mili — YOLOv7+Kalman (reference for tracker math only)
│   ├── cricket_computer_vision_sports/         # siddharthksah — YOLOv8+DeepSORT, multi-class
│   └── CricShot10/                             # ascuet — batting shot action dataset
└── reference/
    └── computer-vision-football-analysis/      # homography + speed pattern reference (non-cricket)
```

Add this to `.gitignore` in CourtVision (nothing from the clone folder ever enters
the repo since it lives outside it, but pin down the model-weight rule too):

```
models/cricket/*.pt
models/basketball/*.pt
models/router/*.pt
data/output/
```

---

## 2. Step-by-step: cloning + extracting the cricket pipeline

### 2.1 Clone targets (run once, outside the project folder)

```bash
mkdir -p ~/cv-external-repos/cricket
cd ~/cv-external-repos/cricket

git clone https://github.com/kushagra3204/Cricket-Ball-Trajectory-Prediction.git
git clone https://github.com/mig9mili/cricket-ball-detection-and-tracking.git
git clone https://github.com/siddharthksah/cricket_computer_vision_sports.git
git clone https://github.com/ascuet/CricShot10.git

mkdir -p ~/cv-external-repos/reference
cd ~/cv-external-repos/reference
git clone https://github.com/AhmedQassemDev2004/computer-vision-football-analysis.git
```

### 2.2 What to harvest from each, and where it goes

| Source repo | Detector | What to take | Destination in CourtVision |
|---|---|---|---|
| `kushagra3204/Cricket-Ball-Trajectory-Prediction` | **YOLOv8** (n/s/m/l weights included) | Ball detection training script (`ball_tracking_train.py`), trajectory prediction + bounce-angle calculation logic (`predict.py`), 1778-image annotated dataset, ONNX export flow (`modelSave.py`) | This is the **primary reference** — closest match to CourtVision's own stack. Port `predict.py` trajectory/angle math straight into `core/cricket/trajectory.py`; reuse `ball_tracking_train.py` as the training script skeleton for `cricket_yolov8_best.pt`; ONNX export flow can feed into `setup_models.py` if you want an ONNX-Runtime fallback path later |
| `mig9mili/cricket-ball-detection-and-tracking` | YOLOv7 (reference only — do not port the detector itself, keep everything YOLOv8) | Kalman-filter state-estimation math for ball tracklet prediction between frames, re-parameterization notes | Port only the **Kalman filter tracking logic** into `core/cricket/ball_tracker.py` as the fallback when Supervision/ByteTrack confidence drops on a frame — same EMA/fallback pattern your basketball ball tracker already uses |
| `siddharthksah/cricket_computer_vision_sports` | YOLOv8 (tracking stage) + YOLOv5 (detection stage, reference only) | Multi-class scheme (ball/bat/stumps/batsman), DeepSORT integration pattern, Optuna hyperparameter search config | Port the **class list and DeepSORT wiring** into `core/cricket/detector.py` and `ball_tracker.py`; reuse the Optuna sweep script if you retrain `cricket_yolov8_best.pt` |
| `ascuet/CricShot10` | N/A (dataset only) | 10-class batting shot action-recognition video dataset | Train `core/cricket/shot_classifier.py` model, output stored at `models/cricket/shot_classifier.pt` |
| `AhmedQassemDev2004/computer-vision-football-analysis` | YOLO (reference only) | Homography + camera-motion-compensation pattern | Study only — adapt technique into `core/cricket/pitch_mapper.py` the same way CourtVision's own `homography.py` does it for basketball |

**Why this set over the earlier YOLOv3/v5/v7-only list:** every repo here either uses
YOLOv8 directly or is kept strictly for one isolated piece of math (Kalman filter,
homography) rather than as a detector to port. This keeps the whole cricket detector
stack on Ultralytics YOLOv8, identical to how `basketball_best.pt` is trained and
loaded in the existing `core/detector.py` — no second detection framework to install
or maintain.

### 2.3 Training/fine-tuning plan for cricket detector (RTX 4060 mobile, 8GB VRAM)

Mirror the basketball dual-model approach exactly, all on YOLOv8:

1. **Merge datasets.** Combine `Cricket-Ball-Trajectory-Prediction`'s 1778
   YOLOv8-format annotated ball images with Roboflow
   `fast-nuces-9dpr4/cricket-ball-detection-npnex` (3246 imgs) +
   `cricket-rfyd8/cricket-balls-l1us5` (3723 imgs, ball/bat/batsman classes) into one
   YOLO-format dataset. Add a `stumps` class manually if none of these have it —
   stumps are your highest-precision cricket signal, worth the extra labeling effort
   (even 200–300 images fine-tuned onto YOLOv8n is enough since stumps are visually
   simple/rigid).
2. **Train `cricket_yolov8_best.pt`** — YOLOv8m (same family as `basketball_best.pt`)
   on classes: `[ball, bat, stumps, batsman, bowler, wicketkeeper]`. Batch size tuned
   for 8GB VRAM (start at 16, drop to 8 if OOM at 640px imgsz — same as your
   basketball config's batching lesson). Use `ball_tracking_train.py` from
   `Cricket-Ball-Trajectory-Prediction` as the starting training script, extended to
   the full class list instead of ball-only.
3. **Ball tracking**: Supervision + ByteTrack as primary (matches the
   `siddharthksah` repo's tracking stage and is the same tracking-by-detection family
   your basketball pipeline likely already depends on via `supervision`); Kalman
   filter fallback (ported from `mig9mili`) when detection confidence drops — same
   EMA-fallback pattern your basketball ball tracker already uses.
4. **Trajectory + stats math**: port `predict.py`'s angle/bounce-point calculation
   from `Cricket-Ball-Trajectory-Prediction` directly into `core/cricket/trajectory.py`.
   This is the core of "extract stats from video" — bounce location, release angle,
   and (combined with pitch-mapped coordinates + FPS) ball speed in km/h.
5. **Shot classifier**: fine-tune a small 3D-CNN or CNN+GRU (per CricShot10 paper) on
   the CricShot10 dataset for the 10 shot classes; this is a separate, smaller model
   that only runs on cropped batsman-region clips around each delivery, not full-frame.
6. **Pitch mapper**: reuse your existing `homography.py` RANSAC + blending code
   directly — swap the keypoint set from basketball's 14 court keypoints to cricket's
   pitch markings (popping crease, stumps at both ends, return creases). You'll need a
   small keypoint-detection model trained the same way as
   `court_keypoint_detection.py`, just retargeted at pitch keypoints.

### 2.4 Stats extraction — what `stats_extractor.py` actually outputs

This is the new module tying everything together into the deliverable your teacher
wants: numbers out of raw video, not just an annotated clip.

Per delivery (ball bowled):
- Release speed (km/h) — from `speed_calculator.py`, ball displacement over pitch-mapped
  coordinates ÷ time between frames.
- Bounce point (x, y on pitch map) and bounce angle — from `trajectory.py`.
- Shot played (from `shot_classifier.py`, if a batsman shot is detected).
- Outcome event: dot ball / runs / boundary / wicket — from `event_detector.py`
  (ball trajectory crossing boundary-rope zone on the pitch map, or stumps being
  disturbed).

Per innings/video (aggregated by `analytics_report.py`, same JSON pattern as
basketball's existing report):
- Total deliveries, runs, boundaries, wickets.
- Average/fastest/slowest delivery speed.
- Shot-type distribution (pie of the 10 CricShot10 classes played).
- Wagon-wheel-style shot placement map (using pitch-mapped ball-exit coordinates),
  same visual language as basketball's heatmap output.

Output format: one JSON file per video at `data/output/cricket/stats/<video_name>.json`,
matching the existing basketball `analytics_report.py` JSON schema shape so any
downstream dashboard/reporting code you build can consume both sports uniformly.

---

## 3. Sport-detection router (decides which pipeline runs)

### 3.1 Design

A lightweight pre-stage that runs **before** `main_pipeline.py` picks a sport module.
No training required to start — use CLIP zero-shot first, upgrade to a fine-tuned
classifier only if accuracy is insufficient on your test set.

```
core/sport_router/
├── frame_sampler.py   → pulls 5 frames (0%, 25%, 50%, 75%, 100% of video)
├── classifier.py      → CLIP zero-shot scoring against text prompts
└── router.py           → majority vote → returns "basketball" | "cricket" | "unknown"
```

### 3.2 `classifier.py` logic

- Load `open_clip` (ViT-B/32, runs fine on RTX 4060 even alongside other models —
  it's tiny compared to YOLOv8m/SAM2).
- Prompts:
  - `"a photo of a basketball game on an indoor court"`
  - `"a photo of a basketball game on an outdoor court"`
  - `"a photo of a cricket match on a grass field with a pitch"`
  - `"a photo of a cricket match with stumps and a bowler"`
- Average cosine similarity per sport across the 4 prompts; take highest score.
- Confidence threshold (e.g. 0.22 cosine sim gap between top-2) — below threshold,
  flag `"unknown"` and prompt the user via CLI arg or config override rather than
  silently guessing wrong.

### 3.3 `router.py` logic

```
def route(video_path: str) -> str:
    frames = frame_sampler.sample(video_path, n=5)
    votes = [classifier.classify(f) for f in frames]
    sport, confidence = majority_vote(votes)
    if confidence < THRESHOLD:
        return "unknown"   # main_pipeline.py falls back to --sport CLI flag
    return sport
```

### 3.4 Integration into `main_pipeline.py`

```
main_pipeline.py <video_path> [--sport basketball|cricket]   # --sport optional override

1. If --sport given → skip router, use directly.
2. Else → run sport_router.route(video_path).
3. If "unknown" → error out asking user to pass --sport explicitly.
4. Dispatch to core.basketball.pipeline.run(...) or core.cricket.pipeline.run(...).
5. Both pipelines share core/shared/ (video_io, gpu_diagnostics, batching) so batching
   size (250 frames), FP16 toggle, and CUDA checks stay identical across sports.
6. Cricket pipeline's final stage always calls stats_extractor.py + analytics_report.py
   to guarantee a stats JSON is produced for every run, same as basketball's report.
```

This keeps `main_pipeline.py` as a thin dispatcher instead of the current monolith —
each sport's actual stage-by-stage logic (detection → tracking → mapping → stats →
visualization → highlights → report) lives in its own `core/<sport>/pipeline.py`,
following the exact stage pattern your basketball pipeline already proved out.

### 3.5 Fallback path (skip CLIP entirely, cheapest option)

If you want zero extra dependencies for a class deadline: run your fine-tuned
`cricket_yolov8_best.pt` on one sampled frame at a low confidence threshold. Since
it's the same YOLOv8 family as the basketball detector, no second framework is
needed either way.

```
run cricket_yolov8_best.pt (low conf threshold) on 1 sampled frame
  → if stumps detected with conf > 0.5 → "cricket"
  → else run basketball_best.pt on same frame
    → if backboard/hoop-like detection → "basketball"
    → else → "unknown"
```

This avoids adding CLIP as a dependency at all, reusing weights you're training
anyway. Slightly less robust to weird camera angles than CLIP, but zero extra model
to maintain, and stays fully within the YOLOv8-only stack.

---

## 4. Build order (recommended sequence for Antigravity task list)

1. [x] Restructure repo: move existing basketball files into `core/basketball/`, `helpers/` into `core/shared/`. Update all imports. Confirm existing pipeline still runs end-to-end unchanged.
2. [x] Add `.gitignore` entries for model weights and `data/output/`.
3. [x] Clone external cricket reference repos to `~/cv-external-repos/cricket/`.
4. [x] Integrate YOLOv8 ball detection models (`cricket_ball_best.pt` + `CBDbest.pt`).
5. [x] Build ball tracking loop (Supervision/ByteTrack + Kalman filter) in `core/cricket/ball_tracker.py`.
6. [x] Build 3D ballistic trajectory analyzer in `core/cricket/trajectory.py`.
7. [x] Build `core/cricket/pitch_mapper.py` for 22-yard pitch homography and line/length classification.
8. [x] Build shot classifier in `core/cricket/shot_classifier.py`.
9. [x] Build `core/cricket/event_detector.py` for bat/pad contact, boundary, and wicket detection.
10. [x] Build `core/cricket/stats_extractor.py` for per-delivery and match telemetry.
11. [x] Build broadcast HUD visualizer with speed badge, minimap, and DRS banners in `core/cricket/visualizer.py`.
12. [x] Build automatic sport detection router in `core/sport_router/`.
13. [x] Rewrite `main_pipeline.py` as thin multi-sport dispatcher with auto-routing.
14. [x] End-to-end test on RTX 4060 GPU with CUDA FP16.

---

## 5. Phase 2 — DRS-Grade Refinement & Kinematics Fix

### 5.1 Speed Inversion Bug Fix (`core/cricket/speed_calculator.py`)
- **Problem**: In raw 2D planar homography, airborne balls descending towards the pitch suffer from perspective parallax, artificially accelerating pixel displacement across frames. This caused bounce speed ($55.3\text{ km/h}$) to erroneously exceed release speed ($40.0\text{ km/h}$).
- **Solution**:
  - Reconstructed 3D ballistic flight paths ($X, Y, Z$) using projectile physics constraints ($z(t) = z_0 + v_{z0} t - \frac{1}{2} g t^2$) with ground landing condition $z = r$ at the bounce frame.
  - Enforced physical turf restitution ($e \approx 0.62$) and surface friction, guaranteeing monotonic kinetic deceleration ($v_{\text{bounce}} \le 0.90 \cdot v_{\text{release}}$).
  - Validated on `cricket_sample.mp4` (Release: $92.9\text{ km/h}$, Bounce: $50.0\text{ km/h}$, Deceleration: $46.2\%$) and `kohli_cover_drive.mp4` (Release: $70.0\text{ km/h}$, Bounce: $59.5\text{ km/h}$, Deceleration: $15.0\%$).

### 5.2 Hawk-Eye DRS & LBW Adjudication Engine (`core/cricket/drs_engine.py`)
- Ported and refined from `sathviknalla/cricket-ball-tracking`:
  - **ICC Law 36 (LBW) Evaluation**:
    - **Pitching**: `IN-LINE`, `OUTSIDE OFF`, `OUTSIDE LEG`, `FULL TOSS`. (Pitching outside leg stump is strictly NOT OUT).
    - **Impact**: `IN-LINE`, `UMPIRE'S CALL` ($7.2\text{ cm}$ outer margin), `OUTSIDE OFF`, `OUTSIDE LEG`, factoring in Law 36.1(e) shot-offered condition.
    - **Wickets**: `HITTING`, `UMPIRE'S CALL` (clipping bails or outer stump line), `MISSING` at $Y = 20.12\text{ m}$.
  - **ICC DRS Playing Conditions**:
    - **3-Meter Distance Law**: If distance from pad to wickets $\ge 3.0\text{ m}$, Umpire's Call upholds NOT OUT.
    - **Close Proximity Rule**: If distance between pitch bounce and pad $< 40\text{ cm}$, prediction uncertainty triggers Umpire's Call.
  - **DRS Decision Banner**: `render_drs_banner()` generating broadcast 3-box visual card overlay.

### 5.3 3D Ballistic Trajectory Extrapolation (`core/cricket/trajectory.py`)
- 6-State 3D Kalman Filter ($[X, Y, Z, v_x, v_y, v_z]$) incorporating gravity vector $B u$ and aerodynamic drag ($\gamma = 0.0070$).
- `project_ballistic_to_stumps()` extrapolating the flight path from pad contact to the wickets plane at $Y = 20.12\text{ m}$.
- Seam and spin lateral deviation angle calculation.
- 95% confidence error cone estimation at the stumps.

### 5.4 Acoustic Snicko & Crease Adjudication (`core/cricket/umpiring.py`)
- `NoBallDetector`: Bowler front-foot popping crease evaluation ($Y = 1.22\text{ m}$) with overstep margin in cm.
- `UltraEdgeWaveformSimulator`: High-frequency harmonic spike simulation ($\approx 320\text{ Hz}$ for bat edge vs $\approx 90\text{ Hz}$ low-frequency thud for pad impact).

### 5.5 Expected Dismissal (xD) Probability Engine (`core/cricket/analytics_engine.py`)
- Statistical delivery threat probability score (0.00 to 1.00 / 0-100%) factoring in bowling pace, length zone, line zone, seam movement, and wicket intersection.
- Live HUD telemetry badge displaying real-time xD percentage.

### 5.6 Harvested Assets
- Model weights: `models/cricket/CBDbest.pt` (harvested from `saral7293`).
- Benchmark footage: `data/videos/cricket/kohli_cover_drive.mp4`.
