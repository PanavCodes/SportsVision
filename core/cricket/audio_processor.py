import os
import numpy as np
from typing import List, Dict, Optional, Tuple

class CricketAudioProcessor:
    """
    Audio Energy & Acoustic Peak Detector for Cricket Highlights.
    
    Inspired by vijay-2012/High-Spot--Cricket-Highlights-Generator.
    Extracts the audio track from broadcast video, computes short-time energy (RMS),
    and detects crowd cheers, umpire appeals, and bat-crack acoustic spikes.
    """
    def __init__(self, window_sec: float = 0.25, threshold_std: float = 2.0):
        self.window_sec = window_sec
        self.threshold_std = threshold_std

    def extract_audio_peaks(self, video_path: str, fps: float = 30.0) -> Dict:
        """
        Analyzes audio track from video file and returns detected high-energy event frames.
        Returns:
            dict with:
                'has_audio': bool,
                'peak_frames': list of frame indices,
                'peak_times': list of timestamps in seconds,
                'energy_envelope': list of sampled energy values
        """
        if not os.path.exists(video_path):
            return {'has_audio': False, 'peak_frames': [], 'peak_times': []}

        try:
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)

            if video.audio is None:
                video.close()
                return {'has_audio': False, 'peak_frames': [], 'peak_times': []}

            # Sample audio array at 22050 Hz (sufficient for energy envelope)
            sample_rate = 22050
            audio_array = video.audio.to_soundarray(fps=sample_rate)
            video.close()

            if audio_array is None or len(audio_array) == 0:
                return {'has_audio': False, 'peak_frames': [], 'peak_times': []}

            # Convert stereo to mono if needed
            if audio_array.ndim > 1:
                mono_audio = np.mean(audio_array, axis=1)
            else:
                mono_audio = audio_array

            # Window size in audio samples
            hop_samples = int(self.window_sec * sample_rate)
            n_windows = len(mono_audio) // hop_samples

            if n_windows < 4:
                return {'has_audio': True, 'peak_frames': [], 'peak_times': []}

            # Compute RMS short-time energy
            rms_energy = []
            timestamps = []
            for i in range(n_windows):
                segment = mono_audio[i * hop_samples : (i + 1) * hop_samples]
                energy = float(np.sqrt(np.mean(segment ** 2)))
                rms_energy.append(energy)
                timestamps.append(i * self.window_sec)

            rms_energy = np.array(rms_energy, dtype=np.float32)

            # Baseline statistics
            mean_energy = float(np.mean(rms_energy))
            std_energy = float(np.std(rms_energy))
            threshold = mean_energy + self.threshold_std * std_energy

            # Detect peaks above threshold
            peak_indices = np.where(rms_energy > threshold)[0]
            peak_times = []
            peak_frames = []

            # Gating: enforce minimum 2.0s separation between highlight peaks
            last_peak_time = -999.0
            for p_idx in peak_indices:
                t = timestamps[p_idx]
                if t - last_peak_time >= 2.0:
                    peak_times.append(round(t, 2))
                    peak_frames.append(int(round(t * fps)))
                    last_peak_time = t

            return {
                'has_audio': True,
                'mean_energy': round(mean_energy, 4),
                'peak_threshold': round(threshold, 4),
                'peak_frames': peak_frames,
                'peak_times': peak_times
            }

        except Exception as e:
            # Graceful fallback if video has no valid audio codec
            return {
                'has_audio': False,
                'error': str(e),
                'peak_frames': [],
                'peak_times': []
            }
