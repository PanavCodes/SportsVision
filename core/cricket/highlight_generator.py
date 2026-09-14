import os
import cv2
from typing import List, Optional
import config
from core.cricket.audio_processor import CricketAudioProcessor

class CricketHighlightGenerator:
    """
    Multi-Modal Audio-Visual Highlight Generator for Cricket.
    
    Inspired by vijay-2012/High-Spot--Cricket-Highlights-Generator.
    Fuses visual event detections (wickets, boundaries, high-xD deliveries)
    with acoustic energy peaks (crowd roar, umpire appeals, bat cracks).
    """
    def __init__(self, clip_before_sec: float = 3.0, clip_after_sec: float = 3.0):
        self.clip_before_sec = clip_before_sec
        self.clip_after_sec = clip_after_sec
        self.audio_processor = CricketAudioProcessor()

    def fuse_highlight_moments(
        self,
        visual_frames: List[int],
        video_path: str,
        fps: float = 30.0,
        temporal_tolerance_sec: float = 2.5
    ) -> List[int]:
        """
        Fuses visual candidate frames with acoustic peaks from the audio track.
        Prioritizes moments that have BOTH visual events AND acoustic cheering.
        """
        audio_info = self.audio_processor.extract_audio_peaks(video_path, fps=fps)
        audio_frames = audio_info.get('peak_frames', [])

        if not audio_frames:
            # Fall back to visual frames alone
            return sorted(list(set(visual_frames)))

        combined = []
        tol_frames = int(temporal_tolerance_sec * fps)

        # 1. Co-occurring moments (highest highlight priority)
        co_occurring = []
        for vf in visual_frames:
            if any(abs(vf - af) <= tol_frames for af in audio_frames):
                co_occurring.append(vf)

        # 2. Add remaining visual events (guaranteed boundaries/wickets)
        combined.extend(co_occurring)
        for vf in visual_frames:
            if vf not in combined:
                combined.append(vf)

        # 3. Add prominent standalone audio roar moments
        for af in audio_frames:
            if not any(abs(af - c) <= tol_frames for c in combined):
                combined.append(af)

        return sorted(combined)

    def extract_highlights(
        self,
        video_path: str,
        highlighted_frames: List[int],
        output_path: str,
        fps: int = 30
    ) -> List[int]:
        """
        Extract subclips around key event moments and concatenate them into a highlights reel.
        """
        # Multi-modal fusion of visual + acoustic moments
        fused_frames = self.fuse_highlight_moments(highlighted_frames, video_path, fps=fps)

        if not fused_frames:
            print("[CricketHighlightGenerator] No major highlights detected.")
            return []

        print(f"[CricketHighlightGenerator] Generating fused highlights for {len(fused_frames)} moments...")

        video = None
        final_clip = None
        try:
            from moviepy import VideoFileClip, concatenate_videoclips
            video = VideoFileClip(video_path)
            clips = []

            for frame_idx in fused_frames:
                start_sec = max(0.0, (frame_idx / fps) - self.clip_before_sec)
                end_sec = min(video.duration - 0.05, (frame_idx / fps) + self.clip_after_sec)

                if end_sec > start_sec + 0.5:
                    clips.append(video.subclipped(start_sec, end_sec))

            if clips:
                final_clip = concatenate_videoclips(clips)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                final_clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False)
                print(f"[CricketHighlightGenerator] Fused highlights saved: {output_path}")
        except Exception as e:
            print(f"[CricketHighlightGenerator] Fallback to OpenCV highlight extraction ({e})...")
            self._extract_with_opencv(video_path, fused_frames, output_path, fps)
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

        return fused_frames

    def _extract_with_opencv(self, video_path, highlighted_frames, output_path, fps):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        intervals = []
        for f in highlighted_frames:
            s = max(0, f - int(self.clip_before_sec * fps))
            e = min(total_f, f + int(self.clip_after_sec * fps))
            intervals.append((s, e))

        # Merge overlapping intervals
        merged = []
        for s, e in sorted(intervals):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for s, e in merged:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            for curr in range(s, e):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

        cap.release()
        out.release()
        print(f"[CricketHighlightGenerator] Fused highlights written via OpenCV: {output_path}")
