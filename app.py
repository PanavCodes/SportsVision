import os
import sys
import json
import time
import re
import asyncio
import subprocess
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = None

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web" / "static"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(config.OUTPUT_DIR)
BASKETBALL_OUTPUT_DIR = Path(config.BASKETBALL_OUTPUT_DIR)
CRICKET_OUTPUT_DIR = Path(config.CRICKET_OUTPUT_DIR)
CRICKET_STATS_DIR = Path(config.CRICKET_STATS_DIR)

app = FastAPI(title="SportsVision Multi-Sport Spatial Analytics Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active pipeline management
class PipelineManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.current_video = ""
        self.start_time = 0.0
        self.log_history: List[str] = []
        self.listeners: List[asyncio.Queue] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.detected_sport = "unknown"
        self.status = {
            "is_running": False,
            "stage": 0,
            "total_stages": 8,
            "stage_name": "Idle",
            "progress_pct": 0,
            "current_frame": 0,
            "total_frames": 0,
            "fps": "0.0",
            "eta": "--:--",
            "vram": "0MB",
            "video_name": "",
            "detected_sport": "unknown",
            "error": None,
            "completed": False,
            "results": None
        }

    def reset_status(self, video_name: str, total_frames_hint: int = 0):
        self.log_history = []
        self.detected_sport = "unknown"
        self.status = {
            "is_running": True,
            "stage": 1,
            "total_stages": 8,
            "stage_name": "Initializing Hardware",
            "progress_pct": 0,
            "current_frame": 0,
            "total_frames": total_frames_hint,
            "fps": "0.0",
            "eta": "Calculating...",
            "vram": "N/A",
            "video_name": video_name,
            "detected_sport": "unknown",
            "error": None,
            "completed": False,
            "results": None
        }

    def broadcast(self, message: dict):
        if not self.loop:
            return
        for q in list(self.listeners):
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, message)
            except Exception:
                pass

    def append_log(self, line: str):
        line = line.rstrip("\r\n")
        if not line:
            return
        self.log_history.append(line)
        if len(self.log_history) > 2000:
            self.log_history.pop(0)

        # Parse stages & progress
        self._parse_line(line)
        self.broadcast({"type": "log", "line": line, "status": self.status})

    def _parse_line(self, line: str):
        # ---- Sport Detection ----
        if "Auto-detected sport: BASKETBALL" in line or "Defaulting to BASKETBALL" in line:
            self.detected_sport = "basketball"
            self.status["detected_sport"] = "basketball"
            self.status["total_stages"] = 8
        elif "Auto-detected sport: CRICKET" in line:
            self.detected_sport = "cricket"
            self.status["detected_sport"] = "cricket"
            self.status["total_stages"] = 7

        # ---- Basketball Pipeline Stages (8 stages) ----
        if "[1/8]" in line:
            self.status["stage"] = 1
            self.status["total_stages"] = 8
            self.status["stage_name"] = "Initializing Hardware & CUDA"
        elif "[2/8]" in line:
            self.status["stage"] = 2
            self.status["stage_name"] = "Loading AI Models (YOLO, SAM2, EasyOCR)"
        elif "[3/8]" in line:
            self.status["stage"] = 3
            self.status["stage_name"] = "Opening Video Stream"
            m = re.search(r"Total Frames:\s*(\d+)", line)
            if m:
                self.status["total_frames"] = int(m.group(1))
        elif "[4/8]" in line:
            self.status["stage"] = 4
            self.status["stage_name"] = "Spatial Tracking, Homography & Minimap"
        elif "[5/8]" in line:
            self.status["stage"] = 5
            self.status["stage_name"] = "Video Processing Complete"
            self.status["progress_pct"] = 100
        elif "[6/8]" in line:
            self.status["stage"] = 6
            self.status["stage_name"] = "Parabolic Shot Arc Analysis"
        elif "[7/8]" in line:
            self.status["stage"] = 7
            self.status["stage_name"] = "Extracting Key Highlight Clips"
        elif "[8/8]" in line:
            self.status["stage"] = 8
            self.status["stage_name"] = "Generating Analytics & Thermal Heatmaps"

        # ---- Cricket Pipeline Stages (7 stages) ----
        elif "[1/7]" in line:
            self.status["stage"] = 1
            self.status["total_stages"] = 7
            self.status["stage_name"] = "Initializing Hardware & CUDA"
        elif "[2/7]" in line:
            self.status["stage"] = 2
            self.status["stage_name"] = "Loading Cricket AI Models"
        elif "[3/7]" in line:
            self.status["stage"] = 3
            self.status["stage_name"] = "Opening Video & Audio Stream"
            m = re.search(r"Total Frames:\s*(\d+)", line)
            if m:
                self.status["total_frames"] = int(m.group(1))
        elif "[4/7]" in line:
            self.status["stage"] = 4
            self.status["stage_name"] = "Ball Tracking & Trajectory Analysis"
        elif "[5/7]" in line:
            self.status["stage"] = 5
            self.status["stage_name"] = "Video Processing Complete"
            self.status["progress_pct"] = 100
        elif "[6/7]" in line:
            self.status["stage"] = 6
            self.status["stage_name"] = "Audio-Visual Highlight Extraction"
        elif "[7/7]" in line:
            self.status["stage"] = 7
            self.status["stage_name"] = "Generating Cricket Analytics & Commentary Report"

        # ---- Completion ----
        elif "Pipeline execution complete!" in line or "Analytics Report Saved" in line:
            self.status["completed"] = True
            self.status["is_running"] = False
            self.status["stage_name"] = "Pipeline Finished Successfully"

        # Check for tqdm line: Processing:  50%|..| 45/90 [00:10<00:10, 4.5 frames/s, FPS=4.5, ETA=10s]
        m_prog = re.search(r"(\d+)%\|.*\|\s*(\d+)/(\d+)", line)
        if m_prog:
            pct = int(m_prog.group(1))
            cur = int(m_prog.group(2))
            tot = int(m_prog.group(3))
            self.status["progress_pct"] = pct
            self.status["current_frame"] = cur
            self.status["total_frames"] = tot

        m_fps = re.search(r"FPS=([0-9.]+)", line)
        if m_fps:
            self.status["fps"] = m_fps.group(1)

        m_eta = re.search(r"ETA=([0-9a-zA-Z\s]+),", line)
        if m_eta:
            self.status["eta"] = m_eta.group(1).strip()

        m_vram = re.search(r"VRAM=([0-9a-zA-Z]+)", line)
        if m_vram:
            self.status["vram"] = m_vram.group(1).strip()

    def run_pipeline(
        self,
        video_path: str,
        sport_override: str = "auto",
        max_frames: Optional[int] = None,
        frame_skip: Optional[int] = None,
        batch_size: Optional[int] = None,
        resume: bool = False
    ):
        if self.is_running:
            raise RuntimeError("A pipeline run is already in progress.")

        video_name = Path(video_path).stem
        self.reset_status(video_name)
        self.is_running = True
        self.current_video = video_path
        self.start_time = time.time()

        cmd = [
            sys.executable,
            "-u",
            str(BASE_DIR / "main_pipeline.py"),
            video_path
        ]
        if sport_override and sport_override != "auto":
            cmd.extend(["--sport", sport_override])
        if max_frames:
            cmd.extend(["--max-frames", str(max_frames)])
        if frame_skip:
            cmd.extend(["--frame-skip", str(frame_skip)])
        if batch_size:
            cmd.extend(["--batch-size", str(batch_size)])
        if resume:
            cmd.append("--resume")

        def worker():
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(BASE_DIR),
                    env=env
                )

                for line in iter(self.process.stdout.readline, ''):
                    if not line:
                        break
                    self.append_log(line)

                self.process.stdout.close()
                return_code = self.process.wait()

                if return_code == 0:
                    self.status["completed"] = True
                    self.status["stage_name"] = "Ready"
                    # Transcode output to web mp4 if needed
                    self._ensure_web_video(video_name)
                    # Load results
                    results = load_results_for_video(video_name)
                    self.status["results"] = results
                    self.broadcast({"type": "complete", "video_name": video_name, "results": results})
                else:
                    self.status["error"] = f"Pipeline exited with error code {return_code}"
                    self.broadcast({"type": "error", "error": self.status["error"]})

            except Exception as e:
                self.status["error"] = str(e)
                self.broadcast({"type": "error", "error": str(e)})
            finally:
                self.is_running = False
                self.status["is_running"] = False
                self.process = None

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _ensure_web_video(self, video_name: str):
        """Transcode OpenCV mp4v video to H.264 web compatible video if not already converted."""
        # Check both basketball and cricket output dirs
        for out_dir in [BASKETBALL_OUTPUT_DIR, CRICKET_OUTPUT_DIR]:
            raw_out = out_dir / f"{video_name}_annotated.mp4"
            web_out = out_dir / f"{video_name}_annotated_web.mp4"

            if raw_out.exists() and FFMPEG_EXE:
                if not web_out.exists() or (web_out.stat().st_mtime < raw_out.stat().st_mtime):
                    try:
                        self.append_log(f"[Web Transcoder] Optimizing video for browser playback (H.264)...")
                        subprocess.run(
                            [
                                FFMPEG_EXE, "-y",
                                "-i", str(raw_out),
                                "-c:v", "libx264",
                                "-preset", "veryfast",
                                "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart",
                                str(web_out)
                            ],
                            capture_output=True,
                            check=True
                        )
                        self.append_log(f"[Web Transcoder] Web playback optimization complete: {web_out.name}")
                    except Exception as e:
                        self.append_log(f"[Web Transcoder Warning] {e}")

    def stop_pipeline(self):
        if self.process and self.is_running:
            self.append_log("\n[!] User requested pipeline abort. Terminating process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.is_running = False
            self.status["is_running"] = False
            self.status["stage_name"] = "Aborted by User"
            self.broadcast({"type": "aborted", "status": self.status})
            return True
        return False

pipeline_mgr = PipelineManager()

def find_available_videos():
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    videos = []

    search_dirs = [DATA_DIR, DATA_DIR / "videos", DATA_DIR / "videos" / "basketball", DATA_DIR / "videos" / "cricket"]
    for sdir in search_dirs:
        if sdir.exists():
            for f in sdir.iterdir():
                if f.is_file() and f.suffix.lower() in video_exts and not f.name.startswith("."):
                    # Avoid duplicates by filename
                    if any(v["name"] == f.name for v in videos):
                        continue
                    size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                    name = f.stem

                    # Check basketball outputs
                    has_bb_annotated = (
                        (BASKETBALL_OUTPUT_DIR / f"{name}_annotated.mp4").exists()
                        or (BASKETBALL_OUTPUT_DIR / f"{name}_annotated_web.mp4").exists()
                        or (OUTPUT_DIR / f"{name}_annotated.mp4").exists()
                        or (OUTPUT_DIR / f"{name}_annotated_web.mp4").exists()
                    )
                    has_bb_analytics = (
                        (BASKETBALL_OUTPUT_DIR / f"{name}_analytics.json").exists()
                        or (OUTPUT_DIR / f"{name}_analytics.json").exists()
                    )

                    # Check cricket outputs
                    has_ck_annotated = (
                        (CRICKET_OUTPUT_DIR / f"{name}_annotated.mp4").exists()
                        or (CRICKET_OUTPUT_DIR / f"{name}_annotated_web.mp4").exists()
                        or (OUTPUT_DIR / f"{name}_annotated.mp4").exists()
                    )
                    has_ck_analytics = (
                        (CRICKET_STATS_DIR / f"{name}.json").exists()
                        or (CRICKET_OUTPUT_DIR / f"{name}.json").exists()
                    )

                    has_results = (has_bb_annotated and has_bb_analytics) or (has_ck_annotated and has_ck_analytics)
                    detected_sport = "unknown"
                    if has_ck_analytics:
                        detected_sport = "cricket"
                    elif has_bb_analytics:
                        # Check inside json for true sport tag if available
                        json_path = (BASKETBALL_OUTPUT_DIR / f"{name}_analytics.json")
                        if not json_path.exists():
                            json_path = (OUTPUT_DIR / f"{name}_analytics.json")
                        try:
                            with open(json_path, "r") as jf:
                                jdata = json.load(jf)
                                if jdata.get("metadata", {}).get("sport") == "cricket" or "deliveries" in jdata:
                                    detected_sport = "cricket"
                                else:
                                    detected_sport = "basketball"
                        except Exception:
                            detected_sport = "basketball"

                    videos.append({
                        "name": f.name,
                        "stem": name,
                        "path": str(f.resolve()),
                        "size_mb": size_mb,
                        "has_results": has_results,
                        "detected_sport": detected_sport,
                    })
    return videos


def _find_file(directory: Path, filename: str):
    """Check if a file exists in the directory and return its name, else None."""
    path = directory / filename
    return filename if path.exists() else None


def _find_file_multidir(directories: list, filename: str):
    """Check if a file exists across candidate directories and return its name, else None."""
    for d in directories:
        if (d / filename).exists():
            return filename
    return None


def ensure_cricket_visuals(video_name: str, analytics_data: Optional[dict] = None) -> tuple:
    """Ensure pitch map and ground radar images exist for a cricket video, generating them if needed."""
    pitch_map_name = f"{video_name}_pitch_map.jpg"
    ground_radar_name = f"{video_name}_ground_radar.jpg"
    pitch_map_path = CRICKET_OUTPUT_DIR / pitch_map_name
    ground_radar_path = CRICKET_OUTPUT_DIR / ground_radar_name

    if pitch_map_path.exists() and ground_radar_path.exists():
        return pitch_map_name, ground_radar_name

    CRICKET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np
        from core.cricket.pitch_mapper import CricketPitchMapper

        mapper = CricketPitchMapper()
        deliveries = analytics_data.get("deliveries", []) if analytics_data else []

        length_y_map = {
            "Yorker": 19.1,
            "Full": 17.5,
            "Good Length": 14.8,
            "Short of Length": 12.5,
            "Bouncer / Short": 10.0,
            "Full Toss": 19.5,
        }
        line_x_map = {
            "Wide Outside Off": -0.75,
            "Outside Off": -0.38,
            "On the Stumps": 0.0,
            "Middle": 0.0,
            "Leg Stump / Pad Line": 0.35,
            "Down Leg Side": 0.75,
        }

        # 1. Pitch Map Generation
        if not pitch_map_path.exists():
            template_path = Path(config.CRICKET_PITCH_IMAGE_PATH)
            if template_path.exists():
                img = cv2.imread(str(template_path))
            else:
                img = np.zeros((320, 600, 3), dtype=np.uint8)
                img[:] = (26, 40, 26)

            if img is not None:
                for idx, deliv in enumerate(deliveries):
                    coord = deliv.get("bounce_coord_pitch_metric")
                    if coord and len(coord) == 2 and coord[0] is not None:
                        mx, my = coord[0], coord[1]
                    else:
                        l_str = deliv.get("length", "Good Length")
                        ln_str = deliv.get("line", "On the Stumps")
                        mx = length_y_map.get(l_str, 15.0)
                        my = line_x_map.get(ln_str, 0.0)

                    pt = mapper.metric_to_minimap(mx, my)
                    if pt:
                        px, py = pt
                        is_wkt = deliv.get("is_wicket", False) or "wicket" in str(deliv.get("outcome", "")).lower()
                        runs = deliv.get("runs", 0)
                        if is_wkt:
                            color = (0, 0, 255)       # Red for wicket
                        elif runs >= 4:
                            color = (0, 215, 255)     # Gold for boundary
                        elif runs > 0:
                            color = (255, 229, 0)     # Cyan for runs
                        else:
                            color = (53, 107, 255)    # Orange for dot

                        cv2.circle(img, (px, py), 11, color, 1, cv2.LINE_AA)
                        cv2.circle(img, (px, py), 7, color, -1, cv2.LINE_AA)
                        cv2.circle(img, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

                        spd = deliv.get("release_speed_kmh", 0.0)
                        lbl = f"{int(spd)}k" if spd > 0 else f"#{idx+1}"
                        cv2.putText(img, lbl, (px + 9, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(img, f"SportsVision Pitch Map: {video_name}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(pitch_map_path), img)

        # 2. Ground Radar / Wagon Wheel Generation
        if not ground_radar_path.exists():
            template_path = Path(config.CRICKET_GROUND_IMAGE_PATH)
            if template_path.exists():
                img = cv2.imread(str(template_path))
            else:
                img = np.zeros((530, 538, 3), dtype=np.uint8)
                img[:] = (20, 35, 20)

            if img is not None:
                cx, cy = 265, 263
                shot_angle_map = {
                    "cover_drive": -50,
                    "straight_drive": 0,
                    "pull": 75,
                    "hook": 125,
                    "late_cut": -130,
                    "square_cut": -90,
                    "flick": 50,
                    "sweep": 135,
                    "defense": None,
                    "lofted": -30
                }

                for idx, deliv in enumerate(deliveries):
                    shot = str(deliv.get("shot_played", "")).lower()
                    runs = deliv.get("runs", 0)
                    is_wkt = deliv.get("is_wicket", False)
                    angle_deg = shot_angle_map.get(shot)

                    if angle_deg is not None:
                        rad = np.radians(angle_deg)
                        dist = 70 + (runs * 40) if runs > 0 else 60
                        dist = min(dist, 235)
                        ex = int(cx + dist * np.sin(rad))
                        ey = int(cy - dist * np.cos(rad))

                        color = (0, 0, 255) if is_wkt else ((0, 215, 255) if runs >= 4 else (0, 229, 255))
                        cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.15, line_type=cv2.LINE_AA)
                        cv2.circle(img, (ex, ey), 5, color, -1, cv2.LINE_AA)

                        label = f"{shot.replace('_', ' ')} ({runs}r)"
                        cv2.putText(img, label, (ex + 6, ey + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        offset = 20 + idx * 8
                        cv2.circle(img, (cx + (offset if idx % 2 == 0 else -offset), cy + offset // 2), 4, (100, 180, 255), -1, cv2.LINE_AA)

                cv2.putText(img, f"SportsVision Ground Radar: {video_name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(ground_radar_path), img)

    except Exception as e:
        print(f"[Cricket Visuals Generator Warning] {e}")

    return (
        pitch_map_name if pitch_map_path.exists() else None,
        ground_radar_name if ground_radar_path.exists() else None
    )


def load_results_for_video(video_name: str):
    """Load results with auto sport detection from output directories."""

    # Check cricket outputs first
    ck_analytics_path = CRICKET_STATS_DIR / f"{video_name}.json"
    if not ck_analytics_path.exists():
        ck_analytics_path = CRICKET_OUTPUT_DIR / f"{video_name}.json"

    # Check basketball outputs
    bb_analytics_path = BASKETBALL_OUTPUT_DIR / f"{video_name}_analytics.json"
    if not bb_analytics_path.exists():
        bb_analytics_path = OUTPUT_DIR / f"{video_name}_analytics.json"

    sport = "unknown"
    analytics_data = None

    if ck_analytics_path.exists():
        sport = "cricket"
        try:
            with open(ck_analytics_path, "r") as f:
                analytics_data = json.load(f)
        except Exception:
            pass
    elif bb_analytics_path.exists():
        try:
            with open(bb_analytics_path, "r") as f:
                analytics_data = json.load(f)
            if analytics_data and (analytics_data.get("metadata", {}).get("sport") == "cricket" or "deliveries" in analytics_data):
                sport = "cricket"
            else:
                sport = "basketball"
        except Exception:
            sport = "basketball"

    # If still unknown, check if either video exists
    if sport == "unknown":
        if (CRICKET_OUTPUT_DIR / f"{video_name}_annotated.mp4").exists():
            sport = "cricket"
        elif (BASKETBALL_OUTPUT_DIR / f"{video_name}_annotated.mp4").exists() or (OUTPUT_DIR / f"{video_name}_annotated.mp4").exists():
            sport = "basketball"

    if sport == "cricket":
        out_dirs = [CRICKET_OUTPUT_DIR, OUTPUT_DIR]
        annotated_web = _find_file_multidir(out_dirs, f"{video_name}_annotated_web.mp4")
        annotated_raw = _find_file_multidir(out_dirs, f"{video_name}_annotated.mp4")

        # Ensure visuals exist
        pitch_map, ground_radar = ensure_cricket_visuals(video_name, analytics_data)

        return {
            "sport": "cricket",
            "video_name": video_name,
            "annotated_video": annotated_web or annotated_raw,
            "is_web_optimized": annotated_web is not None,
            "highlights_video": _find_file_multidir(out_dirs, f"{video_name}_highlights.mp4"),
            "pitch_map": pitch_map,
            "ground_radar": ground_radar,
            "wagon_wheel": ground_radar,
            "analytics": analytics_data
        }

    elif sport == "basketball":
        out_dirs = [BASKETBALL_OUTPUT_DIR, OUTPUT_DIR]
        annotated_web = _find_file_multidir(out_dirs, f"{video_name}_annotated_web.mp4")
        annotated_raw = _find_file_multidir(out_dirs, f"{video_name}_annotated.mp4")

        return {
            "sport": "basketball",
            "video_name": video_name,
            "annotated_video": annotated_web or annotated_raw,
            "is_web_optimized": annotated_web is not None,
            "highlights_video": _find_file_multidir(out_dirs, f"{video_name}_highlights.mp4"),
            "team1_heatmap": _find_file_multidir(out_dirs, f"{video_name}_team1_heatmap.jpg"),
            "team2_heatmap": _find_file_multidir(out_dirs, f"{video_name}_team2_heatmap.jpg"),
            "analytics": analytics_data
        }

    # Fallback for unknown
    all_dirs = [BASKETBALL_OUTPUT_DIR, CRICKET_OUTPUT_DIR, OUTPUT_DIR]
    annotated_web = _find_file_multidir(all_dirs, f"{video_name}_annotated_web.mp4")
    annotated_raw = _find_file_multidir(all_dirs, f"{video_name}_annotated.mp4")
    return {
        "sport": "unknown",
        "video_name": video_name,
        "annotated_video": annotated_web or annotated_raw,
        "is_web_optimized": annotated_web is not None,
        "highlights_video": _find_file_multidir(all_dirs, f"{video_name}_highlights.mp4"),
        "team1_heatmap": _find_file_multidir(all_dirs, f"{video_name}_team1_heatmap.jpg"),
        "team2_heatmap": _find_file_multidir(all_dirs, f"{video_name}_team2_heatmap.jpg"),
        "pitch_map": _find_file_multidir(all_dirs, f"{video_name}_pitch_map.jpg"),
        "ground_radar": _find_file_multidir(all_dirs, f"{video_name}_ground_radar.jpg"),
        "analytics": analytics_data
    }


@app.on_event("startup")
async def startup_event():
    pipeline_mgr.loop = asyncio.get_event_loop()

# Range-supporting video streaming endpoint
def stream_video(file_path: Path, request: Request):
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        byte1, byte2 = 0, None
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            groups = match.groups()
            byte1 = int(groups[0])
            if groups[1]:
                byte2 = int(groups[1])

        if byte2 is None:
            byte2 = min(byte1 + 1024 * 1024 * 4, file_size - 1) # 4MB chunk
        else:
            byte2 = min(byte2, file_size - 1)

        length = byte2 - byte1 + 1

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(byte1)
                remaining = length
                chunk_size = 1024 * 256
                while remaining > 0:
                    read_len = min(chunk_size, remaining)
                    data = f.read(read_len)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)
    else:
        def iterfile_full():
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 512):
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iterfile_full(), status_code=200, headers=headers)


# ---------------- API ROUTES ----------------

@app.get("/api/system")
async def get_system_info():
    import torch
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0

    return {
        "cuda_available": cuda_avail,
        "device": config.DEVICE,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "batch_size": config.BATCH_SIZE,
        "frame_skip": config.FRAME_SKIP,
        "use_sam2": getattr(config, "USE_SAM2", True),
        "use_ocr": getattr(config, "USE_OCR", True),
        "use_rf_detr": getattr(config, "USE_RF_DETR", True),
        "ffmpeg_available": FFMPEG_EXE is not None,
        "supported_sports": ["basketball", "cricket"]
    }

@app.get("/api/videos")
async def get_videos():
    return {"videos": find_available_videos()}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # Validate file type
    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: {', '.join(ALLOWED_EXTENSIONS)}")

    # Save to data/videos/
    videos_dir = DATA_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    dest_path = videos_dir / file.filename

    MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    total_written = 0

    with open(dest_path, "wb") as buffer:
        while content := await file.read(1024 * 1024 * 5): # 5MB chunks
            total_written += len(content)
            if total_written > MAX_SIZE:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 2GB)")
            buffer.write(content)

    return {
        "status": "success",
        "filename": file.filename,
        "path": str(dest_path),
        "size_mb": round(total_written / (1024 * 1024), 1)
    }

@app.get("/api/status")
async def get_pipeline_status():
    return {
        **pipeline_mgr.status,
        "elapsed_seconds": round(time.time() - pipeline_mgr.start_time, 1) if pipeline_mgr.is_running else 0
    }

@app.post("/api/run")
async def start_pipeline(
    video_path: str = Form(...),
    sport: str = Form("auto"),
    max_frames: Optional[int] = Form(None),
    frame_skip: Optional[int] = Form(None),
    batch_size: Optional[int] = Form(None),
    resume: bool = Form(False)
):
    if pipeline_mgr.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is already running.")

    if not Path(video_path).exists():
        # Check in DATA_DIR and subdirs
        for candidate_dir in [DATA_DIR, DATA_DIR / "videos", DATA_DIR / "videos" / "basketball", DATA_DIR / "videos" / "cricket"]:
            candidate = candidate_dir / video_path
            if candidate.exists():
                video_path = str(candidate)
                break
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

    try:
        pipeline_mgr.run_pipeline(
            video_path=video_path,
            sport_override=sport,
            max_frames=max_frames,
            frame_skip=frame_skip,
            batch_size=batch_size,
            resume=resume
        )
        return {"status": "started", "video_path": video_path, "sport": sport}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop")
async def stop_pipeline():
    stopped = pipeline_mgr.stop_pipeline()
    return {"status": "stopped" if stopped else "not_running"}

@app.get("/api/stream-logs")
async def stream_logs():
    async def event_generator():
        q = asyncio.Queue()
        pipeline_mgr.listeners.append(q)

        # Send initial batch of logs
        initial_payload = {
            "type": "init",
            "logs": pipeline_mgr.log_history[-150:],
            "status": pipeline_mgr.status
        }
        yield f"data: {json.dumps(initial_payload)}\n\n"

        try:
            while True:
                msg = await q.get()
                yield f"data: {json.dumps(msg)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in pipeline_mgr.listeners:
                pipeline_mgr.listeners.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/results/{video_name}")
async def get_results(video_name: str, background_tasks: BackgroundTasks):
    results = load_results_for_video(video_name)
    # Check if raw exists but web does not — trigger background transcode
    sport = results.get("sport", "unknown")
    if sport == "basketball":
        raw_out = BASKETBALL_OUTPUT_DIR / f"{video_name}_annotated.mp4"
        web_out = BASKETBALL_OUTPUT_DIR / f"{video_name}_annotated_web.mp4"
    elif sport == "cricket":
        raw_out = CRICKET_OUTPUT_DIR / f"{video_name}_annotated.mp4"
        web_out = CRICKET_OUTPUT_DIR / f"{video_name}_annotated_web.mp4"
    else:
        raw_out = OUTPUT_DIR / f"{video_name}_annotated.mp4"
        web_out = OUTPUT_DIR / f"{video_name}_annotated_web.mp4"

    if raw_out.exists() and not web_out.exists():
        background_tasks.add_task(pipeline_mgr._ensure_web_video, video_name)

    return results

@app.get("/api/video/raw/{filename}")
async def get_raw_video(filename: str, request: Request):
    # Search in data dirs
    for search_dir in [DATA_DIR, DATA_DIR / "videos", DATA_DIR / "videos" / "basketball", DATA_DIR / "videos" / "cricket"]:
        path = search_dir / filename
        if path.exists():
            return stream_video(path, request)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/available-sports")
async def get_available_sports():
    return {"sports": ["basketball", "cricket"]}

@app.get("/api/video/output/{filename}")
async def get_output_video(filename: str, request: Request):
    # Search sport-specific output dirs, then fallback
    for out_dir in [BASKETBALL_OUTPUT_DIR, CRICKET_OUTPUT_DIR, OUTPUT_DIR]:
        path = out_dir / filename
        if path.exists():
            return stream_video(path, request)
    # Check if requested web version but only raw exists
    if "_annotated_web.mp4" in filename:
        raw_name = filename.replace("_annotated_web.mp4", "_annotated.mp4")
        for out_dir in [BASKETBALL_OUTPUT_DIR, CRICKET_OUTPUT_DIR, OUTPUT_DIR]:
            raw_cand = out_dir / raw_name
            if raw_cand.exists():
                return stream_video(raw_cand, request)
    raise HTTPException(status_code=404, detail="Output video not found")

@app.get("/api/video/output/{sport}/{filename}")
async def get_sport_output_video(sport: str, filename: str, request: Request):
    if sport == "basketball":
        search_dirs = [BASKETBALL_OUTPUT_DIR, OUTPUT_DIR]
    elif sport == "cricket":
        search_dirs = [CRICKET_OUTPUT_DIR, OUTPUT_DIR]
    else:
        search_dirs = [BASKETBALL_OUTPUT_DIR, CRICKET_OUTPUT_DIR, OUTPUT_DIR]

    for out_dir in search_dirs:
        path = out_dir / filename
        if path.exists():
            return stream_video(path, request)
    if "_annotated_web.mp4" in filename:
        raw_name = filename.replace("_annotated_web.mp4", "_annotated.mp4")
        for out_dir in search_dirs:
            raw_cand = out_dir / raw_name
            if raw_cand.exists():
                return stream_video(raw_cand, request)
    raise HTTPException(status_code=404, detail="Output video not found")

@app.get("/api/image/{filename}")
async def get_image(filename: str):
    # Search sport-specific dirs then fallback
    for search_dir in [BASKETBALL_OUTPUT_DIR, CRICKET_OUTPUT_DIR, OUTPUT_DIR, DATA_DIR]:
        path = search_dir / filename
        if path.exists():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/api/image/{sport}/{filename}")
async def get_sport_image(sport: str, filename: str):
    if sport == "basketball":
        search_dirs = [BASKETBALL_OUTPUT_DIR, OUTPUT_DIR, DATA_DIR]
    elif sport == "cricket":
        search_dirs = [CRICKET_OUTPUT_DIR, OUTPUT_DIR, DATA_DIR]
    else:
        search_dirs = [BASKETBALL_OUTPUT_DIR, CRICKET_OUTPUT_DIR, OUTPUT_DIR, DATA_DIR]

    for search_dir in search_dirs:
        path = search_dir / filename
        if path.exists():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

# Static file serving
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>SportsVision Frontend Loading...</h1>")

if __name__ == "__main__":
    import webbrowser
    port = 8000
    print("=" * 60)
    print(" SportsVision Multi-Sport Spatial Analytics Web Studio")
    print(f" Starting web UI at http://localhost:{port}")
    print("=" * 60)

    # Launch browser automatically
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
