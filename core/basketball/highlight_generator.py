import sys
import os
import cv2
import numpy as np
from scipy.signal import find_peaks
from moviepy import VideoFileClip, concatenate_videoclips

import config

class CourtVisionHighlightGenerator:
    """
    Compiles highlight clips based on shot detector confidences.
    """
    def __init__(self):
        self.threshold = 0.5

    def extract_highlights(self, video_path: str, confidences: list, output_path: str, fps: int = 30):
        print("Extracting highlights based on ResNet shot confidences...")
        if not confidences or max(confidences) < self.threshold:
            print("No high confidence shots detected. Skipping highlights.")
            return
            
        clips = []
        video = VideoFileClip(video_path)
        
        # Simple local maxima finding
        in_highlight = False
        start_frame = 0
        
        for i, conf in enumerate(confidences):
            if conf >= self.threshold and not in_highlight:
                in_highlight = True
                start_frame = max(0, i - (config.HIGHLIGHT_CLIP_BEFORE_SEC * fps))
            elif conf < self.threshold and in_highlight:
                in_highlight = False
                end_frame = min(len(confidences), i + (config.HIGHLIGHT_CLIP_AFTER_SEC * fps))
                
                # Extract clip
                start_time = max(0.0, start_frame / fps)
                end_time = min(end_frame / fps, video.duration - 0.001)
                
                # Make sure clip is at least 0.1s long
                if end_time > start_time + 0.1:
                    clips.append(video.subclipped(start_time, end_time))
                
        if clips:
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False)
        else:
            print("No complete highlight clips formed.")
