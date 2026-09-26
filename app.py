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

app = FastAPI(title="CourtVision Basketball Spatial Analytics Engine")

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
        self.status = {
            "is_running": False,
            "stage": 0,
            "stage_name": "Idle",
            "progress_pct": 0,
            "current_frame": 0,
            "total_frames": 0,
            "fps": "0.0",
            "eta": "--:--",
            "vram": "0MB",
            "video_name": "",
            "error": None,
            "completed": False,
            "results": None
        }

    def reset_status(self, video_name: str, total_frames_hint: int = 0):
        self.log_history = []
        self.status = {
            "is_running": True,
            "stage": 1,
            "stage_name": "Initializing Hardware",
            "progress_pct": 0,
            "current_frame": 0,
            "total_frames": total_frames_hint,
            "fps": "0.0",
            "eta": "Calculating...",
            "vram": "N/A",
            "video_name": video_name,
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
        # Stage regexes
        if "[1/8]" in line:
            self.status["stage"] = 1
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
        elif "Pipeline execution complete!" in line:
            self.status["completed"] = True
            self.status["is_running"] = False
            self.status["stage_name"] = "Pipeline Finished Successfully"

        # Check for tqdm line: Processing:  50%|...| 45/90 [00:10<00:10, 4.5 frames/s, FPS=4.5, ETA=10s, Batch=1.1s, VRAM=512MB]
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
                    self.status["stage"] = 8
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
        raw_out = OUTPUT_DIR / f"{video_name}_annotated.mp4"
        web_out = OUTPUT_DIR / f"{video_name}_annotated_web.mp4"

        if raw_out.exists() and FFMPEG_EXE:
            # Check if web_out doesn't exist or is older
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

    search_dirs = [DATA_DIR, DATA_DIR / "videos"]
    for sdir in search_dirs:
        if sdir.exists():
            for f in sdir.iterdir():
                if f.is_file() and f.suffix.lower() in video_exts and not f.name.startswith("."):
                    size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                    name = f.stem
                    # Check if processed results exist
                    has_annotated = (OUTPUT_DIR / f"{name}_annotated.mp4").exists() or (OUTPUT_DIR / f"{name}_annotated_web.mp4").exists()
                    has_analytics = (OUTPUT_DIR / f"{name}_analytics.json").exists()
                    has_heatmaps = (OUTPUT_DIR / f"{name}_team1_heatmap.jpg").exists()

                    videos.append({
                        "name": f.name,
                        "stem": name,
                        "path": str(f.resolve()),
                        "size_mb": size_mb,
                        "has_results": has_annotated and has_analytics,
                        "has_annotated": has_annotated,
                        "has_analytics": has_analytics,
                        "has_heatmaps": has_heatmaps
                    })
    return videos

def load_results_for_video(video_name: str):
    annotated_web = OUTPUT_DIR / f"{video_name}_annotated_web.mp4"
    annotated_raw = OUTPUT_DIR / f"{video_name}_annotated.mp4"
    highlight_path = OUTPUT_DIR / f"{video_name}_highlights.mp4"
    analytics_path = OUTPUT_DIR / f"{video_name}_analytics.json"
    t1_heatmap = OUTPUT_DIR / f"{video_name}_team1_heatmap.jpg"
    t2_heatmap = OUTPUT_DIR / f"{video_name}_team2_heatmap.jpg"

    results = {
        "video_name": video_name,
        "annotated_video": annotated_web.name if annotated_web.exists() else (annotated_raw.name if annotated_raw.exists() else None),
        "is_web_optimized": annotated_web.exists(),
        "highlights_video": highlight_path.name if highlight_path.exists() else None,
        "team1_heatmap": t1_heatmap.name if t1_heatmap.exists() else None,
        "team2_heatmap": t2_heatmap.name if t2_heatmap.exists() else None,
        "analytics": None
    }

    if analytics_path.exists():
        try:
            with open(analytics_path, "r") as f:
                results["analytics"] = json.load(f)
        except Exception:
            pass

    return results

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
        # e.g., "bytes=0-1024"
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
        "ffmpeg_available": FFMPEG_EXE is not None
    }

@app.get("/api/videos")
async def get_videos():
    return {"videos": find_available_videos()}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = DATA_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        while content := await file.read(1024 * 1024 * 5): # 5MB chunks
            buffer.write(content)
    return {"status": "success", "filename": file.filename, "path": str(dest_path)}

@app.get("/api/status")
async def get_pipeline_status():
    return {
        **pipeline_mgr.status,
        "elapsed_seconds": round(time.time() - pipeline_mgr.start_time, 1) if pipeline_mgr.is_running else 0
    }

@app.post("/api/run")
async def start_pipeline(
    video_path: str = Form(...),
    max_frames: Optional[int] = Form(None),
    frame_skip: Optional[int] = Form(None),
    batch_size: Optional[int] = Form(None),
    resume: bool = Form(False)
):
    if pipeline_mgr.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is already running.")

    if not Path(video_path).exists():
        # Check in DATA_DIR
        candidate = DATA_DIR / video_path
        if candidate.exists():
            video_path = str(candidate)
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

    try:
        pipeline_mgr.run_pipeline(
            video_path=video_path,
            max_frames=max_frames,
            frame_skip=frame_skip,
            batch_size=batch_size,
            resume=resume
        )
        return {"status": "started", "video_path": video_path}
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
    # Check if raw exists but web does not
    raw_out = OUTPUT_DIR / f"{video_name}_annotated.mp4"
    web_out = OUTPUT_DIR / f"{video_name}_annotated_web.mp4"
    if raw_out.exists() and not web_out.exists():
        background_tasks.add_task(pipeline_mgr._ensure_web_video, video_name)
    return results

@app.get("/api/video/raw/{filename}")
async def get_raw_video(filename: str, request: Request):
    # Check DATA_DIR or DATA_DIR/videos
    path = DATA_DIR / filename
    if not path.exists():
        path = DATA_DIR / "videos" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return stream_video(path, request)

@app.get("/api/video/output/{filename}")
async def get_output_video(filename: str, request: Request):
    path = OUTPUT_DIR / filename
    if not path.exists():
        # Check if requested web version but only raw exists
        if "_annotated_web.mp4" in filename:
            raw_cand = OUTPUT_DIR / filename.replace("_annotated_web.mp4", "_annotated.mp4")
            if raw_cand.exists():
                return stream_video(raw_cand, request)
        raise HTTPException(status_code=404, detail="Output video not found")
    return stream_video(path, request)

@app.get("/api/image/{filename}")
async def get_image(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        # Fallback check in data/
        path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

# Static file serving
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>CourtVision Frontend Loading...</h1>")

if __name__ == "__main__":
    import webbrowser
    port = 8000
    print("=" * 60)
    print(" CourtVision Spatial Analytics Web Studio")
    print(f" Starting web UI at http://localhost:{port}")
    print("=" * 60)

    # Launch browser automatically
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
