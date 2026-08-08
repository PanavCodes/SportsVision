"""
video_utils.py

Utility functions for reading and saving videos using OpenCV.
"""

import cv2
import os
from typing import List


def read_video(video_path: str) -> List:
    """
    Reads a video file and returns its frames as a list of numpy arrays.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def save_video(output_frames, output_path):
    dir_name = os.path.dirname(output_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # Use 'mp4v' codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Ensure the output_path ends with .mp4
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'

    height, width = output_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))

    for frame in output_frames:
        out.write(frame)
    out.release()

    print(f"Video saved to {output_path}")
