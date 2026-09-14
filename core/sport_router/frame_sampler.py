import cv2
import numpy as np

class FrameSampler:
    """
    Extracts representative sample frames across the duration of a video
    to evaluate visual sport cues without reading the entire video file.
    """
    def __init__(self, num_samples=5):
        self.num_samples = num_samples

    def sample_frames(self, video_path: str):
        """
        Pulls N evenly spaced frames across the video.
        Returns: list of (timestamp_sec, frame_bgr)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if total_frames <= 0:
            cap.release()
            return []

        # Avoid 0% and 100% exact boundaries which often contain black/fade frames
        # Use 10%, 30%, 50%, 70%, 90%
        percentages = np.linspace(0.10, 0.90, self.num_samples)
        sample_indices = [int(p * total_frames) for p in percentages]

        sampled_frames = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                timestamp = f_idx / fps
                sampled_frames.append((timestamp, frame))

        cap.release()
        return sampled_frames
