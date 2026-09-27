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
            
        intervals = []
        in_highlight = False
        start_frame = 0
        total_frames = len(confidences)
        
        for i, conf in enumerate(confidences):
            if conf >= self.threshold and not in_highlight:
                in_highlight = True
                start_frame = max(0, i - int(config.HIGHLIGHT_CLIP_BEFORE_SEC * fps))
            elif conf < self.threshold and in_highlight:
                in_highlight = False
                end_frame = min(total_frames, i + int(config.HIGHLIGHT_CLIP_AFTER_SEC * fps))
                intervals.append((start_frame, end_frame))
                
        # If video ended while still in a highlight
        if in_highlight:
            intervals.append((start_frame, total_frames))
            
        if not intervals:
            print("No complete highlight clips formed.")
            return

        # Merge overlapping intervals
        merged_intervals = []
        for s, e in sorted(intervals):
            if merged_intervals and s <= merged_intervals[-1][1]:
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], e))
            else:
                merged_intervals.append((s, e))

        video = None
        final_clip = None
        try:
            video = VideoFileClip(video_path)
            clips = []
            for s_frame, e_frame in merged_intervals:
                start_time = max(0.0, s_frame / fps)
                end_time = min(e_frame / fps, video.duration - 0.001)
                if end_time > start_time + 0.1:
                    clips.append(video.subclipped(start_time, end_time))

            if clips:
                final_clip = concatenate_videoclips(clips)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                final_clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False)
                print(f"[BasketballHighlightGenerator] Highlights saved: {output_path}")
            else:
                print("No complete highlight clips formed.")
        except Exception as e:
            print(f"[BasketballHighlightGenerator] MoviePy error ({e}), falling back to OpenCV...")
            self._extract_with_opencv(video_path, merged_intervals, output_path, fps)
        finally:
            if final_clip is not None:
                try:
                    final_clip.close()
                except Exception:
                    pass
            if video is not None:
                try:
                    video.close()
                except Exception:
                    pass

    def _extract_with_opencv(self, video_path: str, intervals: list, output_path: str, fps: int = 30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for s, e in intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, s))
            for curr in range(s, min(total_f, e)):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

        cap.release()
        out.release()
        print(f"[BasketballHighlightGenerator] Highlights written via OpenCV: {output_path}")
