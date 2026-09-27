import os
import sys

# Windows CUDA DLL directory initialization (dynamic discovery for cross-PC portability)
if sys.platform == "win32":
    cuda_candidates = []
    if "CUDA_PATH" in os.environ:
        cuda_candidates.append(os.path.join(os.environ["CUDA_PATH"], "bin"))
    toolkit_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(toolkit_base):
        for ver_dir in sorted(os.listdir(toolkit_base), reverse=True):
            cuda_candidates.append(os.path.join(toolkit_base, ver_dir, "bin"))
    for cand in cuda_candidates:
        if os.path.isdir(cand):
            try:
                os.add_dll_directory(cand)
                break
            except Exception:
                pass

import argparse
import config

def main():
    parser = argparse.ArgumentParser(
        description="SportsVision Multi-Sport Spatial Analytics Engine (Basketball & Cricket)"
    )
    parser.add_argument("video_path", type=str, help="Path to input video (basketball or cricket)")
    parser.add_argument(
        "--sport", type=str, choices=["auto", "basketball", "cricket"], default="auto",
        help="Sport pipeline to execute: 'auto' (default), 'basketball', or 'cricket'"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    parser.add_argument("--frame-skip", type=int, default=None, help="Process every Nth frame (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Frames per batch (overrides config)")
    parser.add_argument("--no-sam2", action="store_true", help="Disable SAM2 instance segmentation to conserve VRAM (recommended for <=4GB GPUs like MX450)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Input video not found: {args.video_path}")

    if args.no_sam2:
        config.USE_SAM2 = False
        print("[Main Pipeline] Hardware Profile: SAM2 disabled to conserve VRAM.")

    # Determine Sport Category
    chosen_sport = args.sport
    if chosen_sport == "auto":
        from core.sport_router.router import detect_sport
        print(f"[Main Pipeline] Auto-detecting sport for: {args.video_path}")
        detected = detect_sport(args.video_path)
        if detected in ["basketball", "cricket"]:
            chosen_sport = detected
            print(f"[Main Pipeline] Auto-detected sport: {chosen_sport.upper()}")
        else:
            print("[Main Pipeline] Sport could not be conclusively determined. Defaulting to BASKETBALL.")
            chosen_sport = "basketball"

    # Dispatch to appropriate pipeline
    if chosen_sport == "cricket":
        from core.cricket.pipeline import run_cricket_pipeline
        return run_cricket_pipeline(
            video_path=args.video_path,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
            batch_size=args.batch_size,
            resume=args.resume
        )
    else:
        from core.basketball.pipeline import run_basketball_pipeline
        return run_basketball_pipeline(
            video_path=args.video_path,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
            batch_size=args.batch_size,
            resume=args.resume
        )


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
