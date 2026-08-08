import cv2
import numpy as np
from typing import Tuple, List

def read_video(video_path: str, max_frames: int = None) -> Tuple[List[np.ndarray], int]:
    """
    Reads a video into a list of frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_count = 0
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        
    cap.release()
    return frames, fps

def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Saves a list of frames to a video file.
    """
    if not frames:
        return
        
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
        
    out.release()
