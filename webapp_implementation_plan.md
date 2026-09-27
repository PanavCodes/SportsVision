# SportsVision — Full Webapp Implementation Plan

> **Goal:** Transform the existing basketball-only "CourtVision Web Studio" into a proper **SportsVision** multi-sport webapp with video upload → processing → results display (heatmaps, annotated video, stats dashboard) for both basketball and cricket.

---

## Current State Assessment

### What Already Exists

| Layer | Status | Key Files |
|:---|:---|:---|
| **Backend (FastAPI)** | ✅ Functional but basketball-only | [`app.py`](file:///c:/important%20files/main%20files/projects/SportsVision/app.py) — video upload, pipeline process execution, SSE log streaming, output serving |
| **Frontend (Static HTML/CSS/JS)** | ⚠️ Basketball-only, outdated branding | [`web/static/index.html`](file:///c:/important%20files/main%20files/projects/SportsVision/web/static/index.html), [`style.css`](file:///c:/important%20files/main%20files/projects/SportsVision/web/static/css/style.css), [`app.js`](file:///c:/important%20files/main%20files/projects/SportsVision/web/static/js/app.js) |
| **Pipeline Backend** | ✅ Fully implemented for both sports | [`main_pipeline.py`](file:///c:/important%20files/main%20files/projects/SportsVision/main_pipeline.py) dispatches to basketball/cricket pipelines |
| **Analytics Output** | ✅ JSON reports for both sports | Basketball: [`analytics_report.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/basketball/analytics_report.py), Cricket: [`analytics_report.py`](file:///c:/important%20files/main%20files/projects/SportsVision/core/cricket/analytics_report.py) |

### Critical Gaps to Fix

1. **Frontend is basketball-only** — no cricket stats/views, no DRS cards, no pitch minimaps, no bowling analysis
2. **Branding still says "CourtVision"** — needs full rebrand to "SportsVision"
3. **Backend doesn't handle sport-specific outputs** — cricket stats are in a different JSON schema and different output directory
4. **No sport detection feedback** — user doesn't know what sport was detected
5. **Upload UX is basic** — no drag-and-drop, no upload progress, no file validation
6. **No cricket-specific visualizations** — wagon wheel, pitch map, speed gun distribution, DRS verdict display
7. **Output paths are inconsistent** — basketball outputs go to `data/output/basketball/`, cricket to `data/output/cricket/stats/`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SportsVision Web App                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 FRONTEND (Static)                     │  │
│  │  index.html  ←→  app.js  ←→  style.css               │  │
│  │                                                       │  │
│  │  Pages/Tabs:                                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │ Upload & │ │  Video   │ │  Game    │ │ Spatial  │ │  │
│  │  │ Pipeline │ │ Theater  │ │Analytics │ │ Heatmaps │ │  │
│  │  │ Studio   │ │          │ │Dashboard │ │& Visuals │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │  │
│  │  ┌──────────┐                                         │  │
│  │  │Highlight │                                         │  │
│  │  │  Reel    │                                         │  │
│  │  └──────────┘                                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↕ HTTP/SSE                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              BACKEND (FastAPI + Uvicorn)               │  │
│  │                                                       │  │
│  │  /api/upload      → Save video to data/videos/        │  │
│  │  /api/run         → Spawn main_pipeline.py subprocess │  │
│  │  /api/stream-logs → SSE real-time progress & stages   │  │
│  │  /api/status      → Pipeline state + detected sport   │  │
│  │  /api/results/:id → Load sport-specific JSON output   │  │
│  │  /api/video/*     → Stream raw/annotated/highlight    │  │
│  │  /api/image/*     → Serve heatmaps/pitch maps         │  │
│  │  /api/system      → GPU/CUDA hardware info            │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↕ subprocess                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           PIPELINE (main_pipeline.py)                  │  │
│  │                                                       │  │
│  │  SportRouter → auto-detect → dispatch to:             │  │
│  │    • run_basketball_pipeline() [8 stages]             │  │
│  │    • run_cricket_pipeline()    [7 stages]             │  │
│  │                                                       │  │
│  │  Outputs: annotated video, analytics JSON,            │  │
│  │           heatmaps, highlights, pitch maps             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Backend Upgrades (`app.py`)

### 1.1 Sport-Aware Output Resolution

> [!IMPORTANT]
> The basketball and cricket pipelines produce outputs in different directories and different JSON schemas. The backend must unify this.

**Changes to [`app.py`](file:///c:/important%20files/main%20files/projects/SportsVision/app.py):**

```diff
 # Add sport detection to PipelineManager
 class PipelineManager:
     def __init__(self):
+        self.detected_sport = "unknown"
         ...
     
     def _parse_line(self, line: str):
+        # Detect sport from pipeline output
+        if "Auto-detected sport: BASKETBALL" in line:
+            self.detected_sport = "basketball"
+            self.status["detected_sport"] = "basketball"
+        elif "Auto-detected sport: CRICKET" in line:
+            self.detected_sport = "cricket"
+            self.status["detected_sport"] = "cricket"
         ...
```

**New helper — `load_results_for_video()` sport-aware version:**

| What | How |
|:---|:---|
| Detect which sport produced the output | Check for `data/output/basketball/<name>_analytics.json` vs `data/output/cricket/stats/<name>.json` |
| Basketball results | Annotated video, heatmaps (team1/team2), analytics JSON, highlights |
| Cricket results | Annotated video, pitch map images, ground radar, DRS banners, delivery-by-delivery JSON, highlights |
| Unified response schema | `{ sport, video_name, annotated_video, highlights_video, analytics, heatmaps[], images[], ... }` |

### 1.2 New API Endpoints

| Endpoint | Method | Purpose |
|:---|:---|:---|
| `GET /api/results/{video_name}` | GET | **Enhanced** — returns sport-specific data with `sport` field |
| `GET /api/video/output/{sport}/{filename}` | GET | Sport-scoped output video serving |
| `GET /api/image/{sport}/{filename}` | GET | Sport-scoped image serving (heatmaps, pitch maps, radar) |
| `POST /api/upload` | POST | **Enhanced** — add upload progress tracking, file type validation |
| `GET /api/available-sports` | GET | Returns `["basketball", "cricket"]` for UI sport filter |

### 1.3 Upload Enhancement

```python
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # Validate file type
    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    
    # Validate file size (max 2GB)
    MAX_SIZE = 2 * 1024 * 1024 * 1024
    
    # Stream to disk with progress tracking
    dest_path = DATA_DIR / "videos" / file.filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_written = 0
    with open(dest_path, "wb") as buffer:
        while content := await file.read(1024 * 1024 * 5):
            total_written += len(content)
            if total_written > MAX_SIZE:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large (max 2GB)")
            buffer.write(content)
    
    return {
        "status": "success",
        "filename": file.filename,
        "path": str(dest_path),
        "size_mb": round(total_written / (1024*1024), 1)
    }
```

### 1.4 Sport-Specific Results Loader

```python
def load_results_for_video(video_name: str) -> dict:
    """Load results with auto sport detection from output directories."""
    
    # Check basketball outputs
    bb_analytics = Path(config.BASKETBALL_OUTPUT_DIR) / f"{video_name}_analytics.json"
    # Check cricket outputs
    ck_analytics = Path(config.CRICKET_STATS_DIR) / f"{video_name}.json"
    
    sport = "unknown"
    analytics_data = None
    
    if bb_analytics.exists():
        sport = "basketball"
        with open(bb_analytics) as f:
            analytics_data = json.load(f)
    elif ck_analytics.exists():
        sport = "cricket"
        with open(ck_analytics) as f:
            analytics_data = json.load(f)
    
    # Build sport-specific output manifest
    if sport == "basketball":
        output_dir = Path(config.BASKETBALL_OUTPUT_DIR)
        return {
            "sport": "basketball",
            "video_name": video_name,
            "annotated_video": _find_video(output_dir, video_name),
            "highlights_video": _find_file(output_dir, f"{video_name}_highlights.mp4"),
            "team1_heatmap": _find_file(output_dir, f"{video_name}_team1_heatmap.jpg"),
            "team2_heatmap": _find_file(output_dir, f"{video_name}_team2_heatmap.jpg"),
            "analytics": analytics_data
        }
    elif sport == "cricket":
        output_dir = Path(config.CRICKET_OUTPUT_DIR)
        return {
            "sport": "cricket",
            "video_name": video_name,
            "annotated_video": _find_video(output_dir, video_name),
            "highlights_video": _find_file(output_dir, f"{video_name}_highlights.mp4"),
            "pitch_map": _find_file(output_dir, f"{video_name}_pitch_map.jpg"),
            "ground_radar": _find_file(output_dir, f"{video_name}_ground_radar.jpg"),
            "wagon_wheel": _find_file(output_dir, f"{video_name}_wagon_wheel.jpg"),
            "analytics": analytics_data
        }
    
    return {"sport": "unknown", "video_name": video_name, "analytics": None}
```

### 1.5 Cricket Pipeline Stage Parsing

Add cricket-specific stage detection to `PipelineManager._parse_line()`:

```python
# Cricket pipeline stages (7 stages)
if "[1/7]" in line:
    self.status["stage"] = 1
    self.status["total_stages"] = 7
    self.status["stage_name"] = "Initializing Hardware & CUDA"
elif "[2/7]" in line:
    self.status["stage"] = 2
    self.status["stage_name"] = "Loading Cricket AI Models"
elif "[3/7]" in line:
    self.status["stage"] = 3
    self.status["stage_name"] = "Opening Video & Audio Stream"
elif "[4/7]" in line:
    self.status["stage"] = 4
    self.status["stage_name"] = "Ball Tracking & Trajectory Analysis"
elif "[5/7]" in line:
    self.status["stage"] = 5
    self.status["stage_name"] = "DRS & Delivery Analysis"
elif "[6/7]" in line:
    self.status["stage"] = 6
    self.status["stage_name"] = "Audio-Visual Highlight Extraction"
elif "[7/7]" in line:
    self.status["stage"] = 7
    self.status["stage_name"] = "Generating Analytics & Match Report"
```

---

## Phase 2: Frontend Rebuild (`index.html`, `style.css`, `app.js`)

### 2.1 Rebrand from CourtVision → SportsVision

| Element | Before | After |
|:---|:---|:---|
| Title | `CourtVision 🏀 Spatial Analytics Engine` | `SportsVision 🏟️ Multi-Sport Analytics Engine` |
| Brand Title | `CourtVision` | `SportsVision` |
| Brand Subtitle | `Spatial Analytics Engine v2.0` | `AI-Powered Multi-Sport Analytics v3.0` |
| Brand Icon | `🏀` | `🏟️` (or a custom logo) |
| Upload Button | `📁 Upload New Basketball Video` | `📁 Upload Sports Video` |
| All hardcoded "basketball" text | Various | Dynamic based on detected sport |

### 2.2 New Navigation Structure

```
┌─────────────────────────────────────────────────────────────┐
│  🏟️ SportsVision    AI-Powered Multi-Sport Analytics v3.0  │
├─────────────────────────────────────────────────────────────┤
│  [🚀 Pipeline Studio] [🎬 Video Theater] [📊 Analytics]    │
│  [🌡️ Heatmaps & Visuals] [🌟 Highlights]                  │
└─────────────────────────────────────────────────────────────┘
```

> Tabs remain the same, but content within each adapts based on detected sport.

### 2.3 Upload & Pipeline Studio (Tab 1) — Redesign

#### A. Enhanced Video Upload Area

Replace the basic file input with a premium drag-and-drop zone:

```
┌──────────────────────────────────────────┐
│                                          │
│    ┌──────────────────────────────────┐  │
│    │                                  │  │
│    │     📁 Drag & Drop Video Here    │  │
│    │     or click to browse           │  │
│    │                                  │  │
│    │     Supports: MP4, MOV, AVI,     │  │
│    │     MKV, M4V (max 2GB)           │  │
│    │                                  │  │
│    │     ┌────────────────────────┐    │  │
│    │     │  ██████████░░ 67%     │    │  │  ← Upload progress bar
│    │     └────────────────────────┘    │  │
│    │                                  │  │
│    └──────────────────────────────────┘  │
│                                          │
│    ─── OR select from library ───        │
│    ┌──────────────────────────────────┐  │
│    │ [▼ Select Video               ] │  │
│    └──────────────────────────────────┘  │
│                                          │
│    ── Sport Detection ──                 │
│    ┌──────────────────────────────────┐  │
│    │ ○ Auto-Detect  ○ Basketball     │  │
│    │ ○ Cricket                       │  │
│    └──────────────────────────────────┘  │
│                                          │
│    [▶ Run Analytics Pipeline]            │
└──────────────────────────────────────────┘
```

**Implementation details:**

- Drag-and-drop zone with hover animation (border glow + scale pulse)
- Upload progress bar (real-time % from `XMLHttpRequest.upload.onprogress`)
- File type and size validation client-side before upload
- Sport override radio buttons (`auto` / `basketball` / `cricket`)
- Pass `--sport` flag to pipeline when user overrides

#### B. Dynamic Stage Stepper

The stage stepper should adapt to the detected sport:

- **Basketball:** 8 stages (existing)
- **Cricket:** 7 stages

```javascript
function updateStageStepperForSport(sport) {
    const basketballStages = [
        { num: 1, label: "Hardware" },
        { num: 2, label: "AI Models" },
        { num: 3, label: "Video" },
        { num: 4, label: "Tracking" },
        { num: 5, label: "Minimap" },
        { num: 6, label: "Shot Arc" },
        { num: 7, label: "Highlights" },
        { num: 8, label: "Report" }
    ];
    
    const cricketStages = [
        { num: 1, label: "Hardware" },
        { num: 2, label: "AI Models" },
        { num: 3, label: "Video/Audio" },
        { num: 4, label: "Ball Track" },
        { num: 5, label: "DRS/Events" },
        { num: 6, label: "Highlights" },
        { num: 7, label: "Report" }
    ];
    
    const stages = sport === "cricket" ? cricketStages : basketballStages;
    renderStageSteps(stages);
}
```

### 2.4 Video Theater (Tab 2) — Sport-Adaptive

The side-by-side comparison stays the same (raw input vs. AI annotated), but the feature guide cards at the bottom adapt:

**Basketball features (existing):**
- Player Tracking & SAM2
- Ball Path & Glowing Tail
- Dynamic Jersey Clustering
- 2D Tactical Minimap

**Cricket features (new):**
- Ball Flight & Kalman Tracking
- Hawk-Eye DRS & LBW System
- Pitch Map & Line/Length
- Speed Gun & Shot Classification

### 2.5 Game Analytics Dashboard (Tab 3) — Full Redesign

This is the biggest change. The dashboard must render completely different content based on sport.

#### A. Basketball Analytics (Existing, with minor polish)

```
┌─────────────────────────────────────────────────────┐
│ Summary Cards: [Shots] [Duration] [Frames] [Court]  │
│                                                     │
│ Team 1 vs Team 2:                                   │
│   Possession Bar: ███████░░░░ 65% vs 35%            │
│   Passes: 12 vs 8                                   │
│   Interceptions: 3 vs 5                             │
│                                                     │
│ Player Speed & Distance Leaderboard Table           │
└─────────────────────────────────────────────────────┘
```

#### B. Cricket Analytics (NEW — Complete Build)

```
┌─────────────────────────────────────────────────────────────┐
│ Match Summary Cards:                                        │
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐  │
│ │ Deliveries│ │  Wickets  │ │Boundaries │ │ Strike Rate │  │
│ │    24     │ │    3      │ │  4×4, 2×6 │ │   125.0     │  │
│ │ 4.0 Overs │ │           │ │           │ │             │  │
│ └───────────┘ └───────────┘ └───────────┘ └─────────────┘  │
│                                                             │
│ Speed Analysis:                                             │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ 🔫 Speed Gun Distribution                            │    │
│ │ Fastest: 142.3 km/h  Avg: 131.8 km/h                │    │
│ │ ████████████████░░░░  ← histogram bars               │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                             │
│ Tactical Distributions (Side-by-Side):                      │
│ ┌──────────────────┐  ┌──────────────────┐                  │
│ │ Line Distribution│  │Length Distribution│                  │
│ │ Off Stump: 42%   │  │Good Length: 54%  │                  │
│ │ On Stumps: 33%   │  │Short: 21%        │                  │
│ │ Leg Stump: 25%   │  │Yorker: 8%        │                  │
│ └──────────────────┘  └──────────────────┘                  │
│                                                             │
│ Shot Type Distribution (Pie/Donut Chart):                   │
│ ┌──────────────────────────────────────────────────────┐    │
│ │  Cover Drive: 25%  Defense: 33%  Pull: 17%  ...     │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                             │
│ Ball-by-Ball Commentary Log:                                │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ 0.1 • Short ball outside off, 138.2 km/h, PULL      │    │
│ │       → FOUR through mid-wicket! xD: 12%             │    │
│ │ 0.2 • Good length on stumps, 135.1 km/h, DEFENSE    │    │
│ │       → Dot ball. xD: 45%                            │    │
│ │ 0.3 • Full outside off, 141.0 km/h, COVER DRIVE     │    │
│ │       → BOUNDARY! xD: 8%                             │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                             │
│ DRS Verdict (if triggered):                                 │
│ ┌──────────────────────────────────────────────────────┐    │
│ │  PITCHING    │   IMPACT    │  WICKETS                │    │
│ │  ✅ In-Line  │  ✅ In-Line │  ✅ HITTING             │    │
│ │              │             │                         │    │
│ │         DECISION: OUT — LBW                          │    │
│ └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.6 Heatmaps & Spatial Visuals (Tab 4) — Sport-Adaptive

#### Basketball (existing):
- Team 1 / Team 2 court coverage heatmaps

#### Cricket (new):
- **Pitch Map** — bounce landing positions on 22-yard pitch template
- **Ground Radar / Wagon Wheel** — shot direction visualization
- **Delivery Speed Timeline** — line chart of speed per delivery

### 2.7 Highlight Reel (Tab 5) — Universal

Same for both sports — just plays the generated highlight video. Update the description text based on sport:

- **Basketball:** "Parabolic shot arc detection, hoop intersection analysis"
- **Cricket:** "Audio-visual fusion — wickets, boundaries, crowd energy spikes"

---

## Phase 3: Frontend JavaScript Logic (`app.js`)

### 3.1 Sport-Aware State Management

```javascript
// New state variable
let detectedSport = "unknown";  // "basketball" | "cricket" | "unknown"

// Updated results handler
async function fetchAndDisplayResults(videoStem) {
    const res = await fetch(`/api/results/${videoStem}`);
    const results = await res.json();
    currentResults = results;
    detectedSport = results.sport || "unknown";
    
    // Render sport-specific UI
    if (detectedSport === "basketball") {
        renderBasketballResults(results);
    } else if (detectedSport === "cricket") {
        renderCricketResults(results);
    }
}
```

### 3.2 Sport Detection from SSE Stream

Listen for sport detection during pipeline run:

```javascript
eventSource.onmessage = (e) => {
    const data = JSON.parse(e.data);
    
    if (data.status && data.status.detected_sport) {
        detectedSport = data.status.detected_sport;
        updateStageStepperForSport(detectedSport);
        updateBrandingForSport(detectedSport);
    }
    // ... existing log/progress handling
};
```

### 3.3 Dynamic Analytics Rendering

```javascript
function renderBasketballResults(results) {
    // Existing logic — possession, passes, interceptions,
    // player speed leaderboard, shot events
    showElement('basketball-analytics-section');
    hideElement('cricket-analytics-section');
    populateBasketballAnalyticsUI(results.analytics);
}

function renderCricketResults(results) {
    showElement('cricket-analytics-section');
    hideElement('basketball-analytics-section');
    populateCricketAnalyticsUI(results.analytics);
}

function populateCricketAnalyticsUI(data) {
    const summary = data.summary;
    
    // Match summary cards
    setText('cricketDeliveries', `${summary.total_deliveries}`);
    setText('cricketOvers', summary.overs);
    setText('cricketWickets', summary.wickets);
    setText('cricketFours', summary.fours);
    setText('cricketSixes', summary.sixes);
    setText('cricketFastestSpeed', `${summary.speed_analysis_kmh.fastest_delivery} km/h`);
    setText('cricketAvgSpeed', `${summary.speed_analysis_kmh.average_delivery} km/h`);
    
    // Line/Length distribution bars
    renderDistributionBars('lineDistContainer', data.tactical_distributions.line);
    renderDistributionBars('lengthDistContainer', data.tactical_distributions.length);
    
    // Shot type donut chart
    renderShotTypeChart(data.tactical_distributions.shot_types);
    
    // Commentary log
    renderCommentaryLog(data.commentary_transcript);
    
    // DRS verdict cards (from deliveries with DRS data)
    renderDRSCards(data.deliveries);
}
```

### 3.4 Charts (Vanilla JS — No External Libraries)

Build lightweight chart components using CSS and canvas:

```javascript
// Horizontal bar chart for distributions
function renderDistributionBars(containerId, data) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    const total = Object.values(data).reduce((a, b) => a + b, 0);
    const sorted = Object.entries(data).sort(([,a], [,b]) => b - a);
    
    sorted.forEach(([label, count]) => {
        const pct = total > 0 ? (count / total * 100) : 0;
        const bar = document.createElement('div');
        bar.className = 'dist-bar-row';
        bar.innerHTML = `
            <span class="dist-label">${label}</span>
            <div class="dist-bar-track">
                <div class="dist-bar-fill" style="width: ${pct}%"></div>
            </div>
            <span class="dist-pct">${pct.toFixed(0)}%</span>
        `;
        container.appendChild(bar);
    });
}
```

### 3.5 Upload with Drag-and-Drop + Progress

```javascript
function initUploadZone() {
    const dropZone = document.getElementById('uploadDropZone');
    const fileInput = document.getElementById('videoUploadInput');
    const progressBar = document.getElementById('uploadProgressFill');
    const progressText = document.getElementById('uploadProgressText');
    
    // Drag events
    ['dragenter', 'dragover'].forEach(e => {
        dropZone.addEventListener(e, (ev) => {
            ev.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });
    
    ['dragleave', 'drop'].forEach(e => {
        dropZone.addEventListener(e, (ev) => {
            ev.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });
    
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });
    
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) uploadFile(e.target.files[0]);
    });
    
    async function uploadFile(file) {
        // Validate
        const validExts = ['.mp4', '.mov', '.avi', '.mkv', '.m4v'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!validExts.includes(ext)) {
            showToast('error', `Unsupported format: ${ext}`);
            return;
        }
        if (file.size > 2 * 1024 * 1024 * 1024) {
            showToast('error', 'File too large (max 2GB)');
            return;
        }
        
        // Upload with progress via XHR
        const formData = new FormData();
        formData.append('file', file);
        
        const xhr = new XMLHttpRequest();
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = pct + '%';
                progressText.textContent = `Uploading... ${pct}%`;
            }
        };
        
        xhr.onload = () => {
            if (xhr.status === 200) {
                showToast('success', `"${file.name}" uploaded!`);
                loadVideos();
            }
            // Reset progress
            progressBar.style.width = '0%';
            progressText.textContent = '';
        };
        
        xhr.open('POST', '/api/upload');
        xhr.send(formData);
    }
}
```

---

## Phase 4: CSS Enhancements (`style.css`)

### 4.1 New Component Styles

| Component | Purpose |
|:---|:---|
| `.upload-drop-zone` | Drag-and-drop upload area with dashed border, hover glow |
| `.upload-drop-zone.drag-over` | Active drag state with pulsing border |
| `.upload-progress` | Upload progress bar (within drop zone) |
| `.sport-badge` | Pill badge showing detected sport (🏀 / 🏏) |
| `.sport-radio-group` | Sport override radio buttons |
| `.cricket-match-card` | Summary cards for cricket stats |
| `.dist-bar-row` | Horizontal distribution bar row |
| `.dist-bar-track` / `.dist-bar-fill` | Distribution percentage bar |
| `.commentary-log` | Scrollable commentary list with delivery markers |
| `.commentary-entry` | Individual delivery commentary item |
| `.drs-verdict-card` | ICC-style 3-box DRS verdict display |
| `.drs-box` | Individual DRS check box (Pitching/Impact/Wickets) |
| `.drs-box.hitting` | Green glow for "Hitting" verdict |
| `.drs-box.missing` | Red for "Missing" |
| `.drs-box.umpires-call` | Orange for "Umpire's Call" |
| `.speed-gauge` | Visual speed indicator for delivery speeds |
| `.toast-notification` | Toast notification system replacing `alert()` |

### 4.2 Upload Drop Zone Styles

```css
.upload-drop-zone {
    border: 2px dashed rgba(255, 255, 255, 0.15);
    border-radius: var(--radius-md);
    padding: 40px 24px;
    text-align: center;
    cursor: pointer;
    transition: var(--transition);
    background: rgba(7, 10, 18, 0.4);
    position: relative;
}

.upload-drop-zone:hover {
    border-color: var(--accent-cyan);
    background: rgba(0, 229, 255, 0.03);
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.1);
}

.upload-drop-zone.drag-over {
    border-color: var(--accent-orange);
    background: rgba(255, 107, 53, 0.06);
    box-shadow: 0 0 30px rgba(255, 107, 53, 0.2);
    transform: scale(1.01);
}

.upload-drop-zone .upload-icon {
    font-size: 3rem;
    margin-bottom: 12px;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
```

### 4.3 DRS Card Styles (Cricket)

```css
.drs-verdict-card {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 20px 0;
    padding: 20px;
    background: rgba(7, 10, 18, 0.6);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
}

.drs-box {
    text-align: center;
    padding: 16px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-color);
    position: relative;
    overflow: hidden;
}

.drs-box-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.drs-box-verdict {
    font-family: var(--font-heading);
    font-size: 1.1rem;
    font-weight: 700;
}

.drs-box.hitting {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.08);
}
.drs-box.hitting .drs-box-verdict { color: var(--accent-green); }

.drs-box.missing {
    border-color: rgba(239, 68, 68, 0.4);
    background: rgba(239, 68, 68, 0.08);
}
.drs-box.missing .drs-box-verdict { color: var(--accent-red); }

.drs-box.umpires-call {
    border-color: rgba(255, 107, 53, 0.4);
    background: rgba(255, 107, 53, 0.08);
}
.drs-box.umpires-call .drs-box-verdict { color: var(--accent-orange); }
```

---

## Phase 5: HTML Structure (`index.html`)

### 5.1 Analytics Pane — Dual Sport Sections

```html
<!-- Analytics Pane -->
<section class="tab-pane" id="analytics-pane">
    
    <!-- Sport Detection Badge -->
    <div class="sport-indicator">
        <span class="sport-badge" id="sportBadge">🏟️ Auto-Detect</span>
    </div>
    
    <!-- Basketball Analytics (shown when sport === "basketball") -->
    <div id="basketball-analytics-section" style="display: none;">
        <!-- Existing basketball UI — possession, passes, player table -->
        ...
    </div>
    
    <!-- Cricket Analytics (shown when sport === "cricket") -->
    <div id="cricket-analytics-section" style="display: none;">
        
        <!-- Match Summary Cards -->
        <div class="analytics-summary-cards">
            <div class="stat-box orange">
                <div class="stat-box-title">Deliveries Bowled</div>
                <div class="stat-box-val" id="cricketDeliveries">0</div>
                <div class="stat-box-sub" id="cricketOvers">0.0 Overs</div>
            </div>
            <div class="stat-box cyan">
                <div class="stat-box-title">Wickets Taken</div>
                <div class="stat-box-val" id="cricketWickets">0</div>
            </div>
            <div class="stat-box green">
                <div class="stat-box-title">Boundaries</div>
                <div class="stat-box-val" id="cricketFours">0</div>
                <div class="stat-box-sub">Fours</div>
            </div>
            <div class="stat-box purple">
                <div class="stat-box-title">Sixes</div>
                <div class="stat-box-val" id="cricketSixes">0</div>
            </div>
        </div>
        
        <!-- Speed Gun Analysis -->
        <div class="glass-panel">
            <div class="panel-header">
                <h2 class="panel-title">
                    <span class="panel-title-icon">🔫</span> Speed Gun Analysis
                </h2>
            </div>
            <div class="speed-cards">
                <div class="metric-card">
                    <div class="metric-title">Fastest Delivery</div>
                    <div class="metric-val cyan" id="cricketFastestSpeed">0.0</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted);">km/h</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Average Speed</div>
                    <div class="metric-val orange" id="cricketAvgSpeed">0.0</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted);">km/h</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Slowest Delivery</div>
                    <div class="metric-val" id="cricketSlowestSpeed">0.0</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted);">km/h</div>
                </div>
            </div>
        </div>
        
        <!-- Line & Length Distribution -->
        <div class="distributions-grid">
            <div class="glass-panel">
                <h3 class="panel-title">📐 Line Distribution</h3>
                <div id="lineDistContainer"></div>
            </div>
            <div class="glass-panel">
                <h3 class="panel-title">📏 Length Distribution</h3>
                <div id="lengthDistContainer"></div>
            </div>
        </div>
        
        <!-- Shot Type Distribution -->
        <div class="glass-panel">
            <h3 class="panel-title">🏏 Shot Type Distribution</h3>
            <div id="shotTypeContainer"></div>
        </div>
        
        <!-- DRS Verdict (conditional) -->
        <div class="glass-panel" id="drsSection" style="display: none;">
            <h3 class="panel-title">⚖️ DRS LBW Verdict</h3>
            <div id="drsVerdictContainer"></div>
        </div>
        
        <!-- Ball-by-Ball Commentary -->
        <div class="glass-panel">
            <div class="panel-header">
                <h3 class="panel-title">
                    <span class="panel-title-icon">📝</span> Ball-by-Ball Commentary
                </h3>
            </div>
            <div class="commentary-log" id="commentaryLog"></div>
        </div>
        
    </div>
</section>
```

### 5.2 Heatmaps Pane — Sport-Adaptive

```html
<section class="tab-pane" id="heatmaps-pane">
    
    <!-- Basketball Heatmaps (existing) -->
    <div id="basketball-heatmaps-section" style="display: none;">
        <!-- Team 1 / Team 2 court heatmaps -->
        ...
    </div>
    
    <!-- Cricket Visuals (new) -->
    <div id="cricket-visuals-section" style="display: none;">
        <div class="heatmaps-grid">
            <!-- Pitch Map -->
            <div class="heatmap-card">
                <div class="heatmap-card-header">
                    <span style="font-weight: 700; color: var(--accent-cyan);">
                        22-Yard Pitch Map
                    </span>
                    <span class="video-badge ai">Bounce Points</span>
                </div>
                <div class="heatmap-image-wrapper">
                    <img id="cricketPitchMapImg" alt="Pitch Map" src="">
                </div>
            </div>
            
            <!-- Ground Radar / Wagon Wheel -->
            <div class="heatmap-card">
                <div class="heatmap-card-header">
                    <span style="font-weight: 700; color: var(--accent-orange);">
                        Ground Radar
                    </span>
                    <span class="video-badge ai">Shot Direction</span>
                </div>
                <div class="heatmap-image-wrapper">
                    <img id="cricketGroundRadarImg" alt="Ground Radar" src="">
                </div>
            </div>
        </div>
    </div>
</section>
```

---

## Phase 6: Implementation Order (Task Checklist)

> [!TIP]
> Each task is designed to be independently testable. Complete them in order for minimal conflicts.

### Backend Tasks

| # | Task | Files Modified | Complexity |
|:---|:---|:---|:---|
| B1 | Add sport detection to `PipelineManager` status | `app.py` | Low |
| B2 | Add cricket stage parsing (`[1/7]` through `[7/7]`) | `app.py` | Low |
| B3 | Build sport-aware `load_results_for_video()` | `app.py` | Medium |
| B4 | Add sport-scoped image/video serving endpoints | `app.py` | Medium |
| B5 | Enhance upload endpoint with validation + progress | `app.py` | Low |
| B6 | Add `--sport` pass-through from `/api/run` to pipeline | `app.py` | Low |
| B7 | Update FastAPI title/metadata to SportsVision | `app.py` | Trivial |

### Frontend Tasks

| # | Task | Files Modified | Complexity |
|:---|:---|:---|:---|
| F1 | Rebrand all CourtVision → SportsVision | `index.html`, `app.js` | Low |
| F2 | Build drag-and-drop upload zone | `index.html`, `style.css`, `app.js` | Medium |
| F3 | Add sport override radio buttons | `index.html`, `app.js` | Low |
| F4 | Build dynamic stage stepper (sport-adaptive) | `app.js`, `index.html` | Medium |
| F5 | Build cricket analytics section HTML | `index.html` | Medium |
| F6 | Build cricket analytics JS rendering | `app.js` | High |
| F7 | Build distribution bar chart component | `app.js`, `style.css` | Medium |
| F8 | Build DRS verdict card component | `index.html`, `style.css`, `app.js` | Medium |
| F9 | Build commentary log component | `index.html`, `style.css`, `app.js` | Low |
| F10 | Build cricket heatmaps/visuals section | `index.html`, `app.js` | Medium |
| F11 | Add sport-adaptive video theater feature cards | `index.html`, `app.js` | Low |
| F12 | Replace `alert()` calls with toast notifications | `style.css`, `app.js` | Medium |
| F13 | Add upload progress bar with XHR | `app.js`, `style.css` | Medium |
| F14 | Polish responsive design for all new sections | `style.css` | Medium |

### Integration Tasks

| # | Task | Complexity |
|:---|:---|:---|
| I1 | End-to-end test: upload basketball video → run → view results | Medium |
| I2 | End-to-end test: upload cricket video → run → view results | Medium |
| I3 | Test sport auto-detection UI feedback during pipeline | Low |
| I4 | Test pre-computed results loading for both sports | Low |

---

## Summary of File Changes

| File | Action | Scope of Change |
|:---|:---|:---|
| [`app.py`](file:///c:/important%20files/main%20files/projects/SportsVision/app.py) | **Modify** | Sport detection, cricket stage parsing, sport-aware results loader, enhanced upload, sport-scoped serving |
| [`web/static/index.html`](file:///c:/important%20files/main%20files/projects/SportsVision/web/static/index.html) | **Major Rewrite** | Rebrand, drag-drop upload, sport radio, cricket analytics HTML, cricket heatmaps HTML, DRS section |
| [`web/static/css/style.css`](file:///c:/important%20files/main%20files/projects/SportsVision/web/static/css/style.css) | **Extend** | Upload zone, DRS cards, distribution bars, commentary log, sport badges, toast notifications |
| [`web/static/js/app.js`](file:///c:/important%20files/main%20files/projects/SportsVision/web/static/js/app.js) | **Major Rewrite** | Sport state management, dynamic rendering, cricket analytics, upload with progress, chart components |

> [!NOTE]
> The backend pipeline code (`main_pipeline.py`, `core/basketball/`, `core/cricket/`) does **not** need changes — it's already fully functional. This plan only touches the web layer.

---

## Tech Stack Decision

| Concern | Decision | Rationale |
|:---|:---|:---|
| Framework | **No framework** — vanilla HTML/CSS/JS | Already using this pattern; adding React/Vite for a single-page dashboard is overkill |
| Charts | **Vanilla CSS + Canvas** | No Chart.js dependency needed; distribution bars and donut charts are simple enough |
| Backend | **FastAPI (existing)** | Already proven, SSE works, no changes to infrastructure |
| Styling | **Vanilla CSS with CSS custom properties** | Existing design system is excellent; just extend it |
| State Management | **Module-level JS variables** | Simple enough for this scope; no Redux/Zustand needed |
