import os
# [Option 1: Global Windows CUDA Fix]
# Forces Windows to load the correct global TensorRT/CUDA 12.4 DLLs first!
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
except Exception:
    pass

import argparse
import sys
import time
import json
import cv2
import torch
import numpy as np
import threading
import queue
import concurrent.futures

# Import configuration
import config


# Import helpers
from helpers.gpu_diagnostics import verify_cuda

class VideoWriterThread(threading.Thread):
    def __init__(self, out_video):
        super().__init__()
        self.out_video = out_video
        self.q = queue.Queue(maxsize=2000)
        self.running = True
        
    def run(self):
        while self.running or not self.q.empty():
            try:
                frame = self.q.get(timeout=0.1)
                self.out_video.write(frame)
            except queue.Empty:
                pass
                
    def write(self, frame):
        self.q.put(frame)
        
    def release(self):
        self.running = False
        self.join()
        self.out_video.release()


def format_time(seconds):
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    # Import our core modules locally to avoid multiprocessing spawn conflicts
    from core.detector import CourtVisionDetector
    from core.team_classifier import CourtVisionTeamClassifier
    from core.court_mapper import CourtVisionMapper
    from core.shot_detector import CourtVisionShotDetector
    from core.highlight_generator import CourtVisionHighlightGenerator
    from core.visualizer import CourtVisionVisualizer
    from core.ball_possession import CourtVisionBallPossession
    from core.pass_detector import CourtVisionPassDetector
    from core.speed_calculator import CourtVisionSpeedCalculator
    from core.analytics_report import CourtVisionAnalyticsReport

    parser = argparse.ArgumentParser(description="CourtVision Basketball Analytics Pipeline")
    parser.add_argument("video_path", type=str, help="Path to input basketball video")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    parser.add_argument("--frame-skip", type=int, default=None, help="Process every Nth frame (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Frames per batch (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print(" CourtVision Spatial Analytics Engine v2.0")
    print(" Full-Game Processing Pipeline")
    print("=" * 60)

    # 1. Verify Hardware
    print("\n[1/8] Initializing Hardware...")
    verify_cuda()

    # Configuration
    BATCH_SIZE = args.batch_size or config.BATCH_SIZE
    FRAME_SKIP = args.frame_skip or config.FRAME_SKIP

    print("\n[2/8] Initializing AI Engines...")
    detector = CourtVisionDetector()
    
    if getattr(config, 'USE_SAM2', False):
        from core.segmentation import PlayerSegmenter
        segmenter = PlayerSegmenter()
    else:
        segmenter = None
        
    if getattr(config, 'USE_OCR', False):
        from core.player_id import CourtVisionPlayerID
        player_id_model = CourtVisionPlayerID()
    else:
        player_id_model = None
        
    team_classifier = CourtVisionTeamClassifier()
    mapper = CourtVisionMapper()
    visualizer = CourtVisionVisualizer()
    shot_detector = CourtVisionShotDetector(fps=30)
    highlight_generator = CourtVisionHighlightGenerator()
    ball_possession_detector = CourtVisionBallPossession()
    pass_detector = CourtVisionPassDetector()
    speed_calculator = CourtVisionSpeedCalculator()
    analytics = CourtVisionAnalyticsReport()

    # 3. Open Video
    print("\n[3/8] Opening Video...")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {args.video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    duration_sec = total_frames / fps if fps > 0 else 0
    analytics.fps = fps

    print(f"  Video: {os.path.basename(args.video_path)}")
    print(f"  Resolution: {width}x{height} @ {fps} FPS")
    print(f"  Total Frames: {total_frames} ({duration_sec:.0f}s / {duration_sec/60:.1f} min)")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Frame Skip: {FRAME_SKIP}")
    print(f"  Court Keypoint Stride: {config.COURT_KEYPOINT_STRIDE}")

    # Setup output
    out_name = os.path.basename(args.video_path).split('.')[0]
    out_video_path = os.path.join(config.OUTPUT_DIR, f"{out_name}_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
    out_width, out_height = config.OUTPUT_RESOLUTION
    out_video = VideoWriterThread(cv2.VideoWriter(out_video_path, fourcc, fps, (out_width, out_height)))
    out_video.start()

    visualizer.set_total_frames(total_frames)

    # Checkpoint support
    checkpoint_path = os.path.join(config.OUTPUT_DIR, f"{out_name}_checkpoint.json")
    start_frame = 0

    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        start_frame = ckpt.get('last_frame', 0)
        print(f"\n  Resuming from frame {start_frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 4. Process Video in Streaming Batches
    print(f"\n[4/8] Processing Video (Streaming Batches)...")
    import tqdm

    frame_count = start_frame
    batch_count = 0
    pipeline_start = time.time()

    pbar = tqdm.tqdm(total=total_frames - start_frame, desc="Processing", unit="frames",
                     initial=0)

    while True:
        # --- Read a batch of frames ---
        video_frames = []
        for _ in range(BATCH_SIZE):
            if args.max_frames and frame_count >= args.max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            # Frame skipping
            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                frame_count += 1
                continue

            # Upscale to OUTPUT_RESOLUTION
            frame = cv2.resize(frame, config.OUTPUT_RESOLUTION, interpolation=cv2.INTER_CUBIC)

            video_frames.append(frame)
            frame_count += 1

        if not video_frames:
            break

        batch_count += 1
        batch_start = time.time()

        # --- [Stage A] Object Detection & Tracking ---
        player_tracks, ball_tracks = detector.track(video_frames)

        # --- [Stage A.1] Player Segmentation (SAM2) ---
        if segmenter is not None:
            for idx, frame in enumerate(video_frames):
                frame_tracks = player_tracks[idx]
                if frame_tracks:
                    pids = list(frame_tracks.keys())
                    bboxes = [frame_tracks[pid]['bbox'] for pid in pids]
                    masks = segmenter.segment_players(frame, bboxes)
                    for pid, mask in zip(pids, masks):
                        frame_tracks[pid]['mask'] = mask

        # --- [Stage B] Team Classification ---
        player_tracks = team_classifier.classify_players(
            video_frames, player_tracks, referee_ids=detector.referee_ids
        )
        
        # Inject dynamic colors into visualizer and minimap drawer if initialized
        if team_classifier.is_initialized and team_classifier.team_1_bgr is not None:
            visualizer.update_colors(team_classifier.team_1_bgr, team_classifier.team_2_bgr)
            # OpenCV colors to tuple for minimap drawer
            mapper.drawer.team_1_color = (int(team_classifier.team_1_bgr[0]), int(team_classifier.team_1_bgr[1]), int(team_classifier.team_1_bgr[2]))
            mapper.drawer.team_2_color = (int(team_classifier.team_2_bgr[0]), int(team_classifier.team_2_bgr[1]), int(team_classifier.team_2_bgr[2]))

        # --- [Stage B.1] Player Identification (OCR) ---
        if player_id_model is not None:
            player_tracks = player_id_model.identify_players(video_frames, player_tracks)

        # --- [Stage C] Court Mapping (Homography with stride optimization) ---
        player_tracks, ball_tracks = mapper.transform_tracks(video_frames, player_tracks, ball_tracks)

        # --- [Stage D] Ball Possession Detection ---
        ball_possession = ball_possession_detector.detect(player_tracks, ball_tracks)

        # --- [Stage E] Pass & Interception Detection ---
        passes = pass_detector.detect_passes(ball_possession, player_tracks)
        interceptions = pass_detector.detect_interceptions(ball_possession, player_tracks)

        # --- [Stage F] Speed & Distance Calculation ---
        speeds_data = None
        distances_data = None
        if mapper.tactical_player_positions is not None:
            distances_data = speed_calculator.calculate_distances(mapper.tactical_player_positions)
            speeds_data = speed_calculator.calculate_speeds(distances_data, fps=fps)

        # --- [Stage G] Draw Broadcast Graphics ---
        def draw_single_frame(idx):
            frame_speeds = speeds_data[idx] if speeds_data else None
            frame_holder = ball_possession[idx] if idx < len(ball_possession) else -1
            frame_pass = passes[idx] if idx < len(passes) else 0
            frame_intercept = interceptions[idx] if idx < len(interceptions) else 0

            return visualizer.draw_frame(
                video_frames[idx],
                player_tracks[idx],
                ball_tracks[idx],
                ball_holder=frame_holder,
                passes_frame=frame_pass,
                interceptions_frame=frame_intercept,
                speeds=frame_speeds
            )
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            video_frames = list(executor.map(draw_single_frame, range(len(video_frames))))

        # --- [Stage H] Tactical Minimap Overlay ---
        output_frames = mapper.render_tactical_view(video_frames, player_tracks, ball_tracks)

        # --- [Stage I] Shot Detection & Write Output ---
        shot_detector.update_batch(output_frames, ball_tracks, player_tracks)
        
        for frame in output_frames:
            out_video.write(frame)

        # --- [Stage J] Update Analytics ---
        analytics.update_possession(ball_possession, player_tracks)
        analytics.update_passes(passes, interceptions)
        analytics.update_heatmaps(mapper.tactical_player_positions)
        analytics.update_team_assignments(player_tracks)
        if distances_data:
            analytics.update_distances(distances_data)
        if speeds_data:
            analytics.update_speeds(speeds_data)
        analytics.update_frame_count(len(video_frames))

        # --- Progress & ETA ---
        batch_time = time.time() - batch_start
        elapsed = time.time() - pipeline_start
        frames_done = frame_count - start_frame
        fps_avg = frames_done / elapsed if elapsed > 0 else 0
        remaining_frames = total_frames - frame_count
        eta = remaining_frames / fps_avg if fps_avg > 0 else 0

        pbar.update(len(video_frames))
        pbar.set_postfix({
            'FPS': f'{fps_avg:.1f}',
            'ETA': format_time(eta),
            'Batch': f'{batch_time:.1f}s',
            'VRAM': f'{torch.cuda.memory_allocated() / 1024**2:.0f}MB' if torch.cuda.is_available() else 'N/A'
        })

        # --- Checkpoint (every 10 batches) ---
        if batch_count % 10 == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump({'last_frame': frame_count, 'batch': batch_count}, f)

        # Check max frames
        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    out_video.release()
    pbar.close()

    total_time = time.time() - pipeline_start
    print(f"\n[5/8] Video processing complete!")
    print(f"  Annotated video saved: {out_video_path}")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Average FPS: {(frame_count - start_frame) / total_time:.1f}")

    # 6. Shot detection summary
    print("\n[6/8] Shot Detection Summary...")
    analytics.update_shots(shot_detector.frame_confidences)

    # 7. Highlights
    print("\n[7/8] Compiling Highlights...")
    highlight_path = os.path.join(config.OUTPUT_DIR, f"{out_name}_highlights.mp4")
    highlight_generator.extract_highlights(
        args.video_path, shot_detector.frame_confidences, highlight_path, fps=fps
    )

    # 8. Analytics Report
    print("\n[8/8] Generating Analytics Report...")
    analytics.save_report(out_name)

    # Cleanup checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nPipeline execution complete! Total wall-clock time: {format_time(total_time)}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
