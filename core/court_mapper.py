import sys
import os
import torch
import numpy as np

import config

if config.REPO1_DIR not in sys.path:
    sys.path.insert(0, config.REPO1_DIR)

from tactical_view.tactical_view import TacticalViewConverter
from Court_keypoint_detection.court_keypoint_detection import CourtKeypointDetector
from drawers.tactical_view_drawer import TacticalViewDrawer

class CourtVisionMapper:
    """
    Wraps Repo 1's court keypoint detection, homography transformation, and tactical drawing.
    Optimized with keypoint stride: runs the expensive court keypoint model every N frames
    and interpolates keypoints for intermediate frames. This is the single biggest 
    performance improvement for long video processing.
    """
    def __init__(self):
        self.converter = TacticalViewConverter(court_image_path=config.COURT_IMAGE_PATH)
        self.keypoint_detector = None
        if os.path.exists(config.COURT_MODEL_PATH):
            self.keypoint_detector = CourtKeypointDetector(config.COURT_MODEL_PATH)
            if config.USE_FP16:
                try:
                    self.keypoint_detector.model.to(config.DEVICE)
                    self.keypoint_detector.model.half()
                except:
                    pass
                
        self.drawer = TacticalViewDrawer()
        
        self.keypoints_per_frame = None
        self.tactical_player_positions = None
        self.ema_tactical_positions = {}
        self.stride = getattr(config, 'COURT_KEYPOINT_STRIDE', 5)

    def _get_keypoints_with_stride(self, video_frames):
        """
        Run court keypoint detection with stride optimization.
        Only detects keypoints on every Nth frame and copies to intermediate frames.
        This avoids running the expensive court keypoint CNN on every single frame,
        since court lines barely move between adjacent frames.
        """
        if self.keypoint_detector is None:
            return None
            
        stride = self.stride
        
        # Select keyframes to process
        keyframe_indices = list(range(0, len(video_frames), stride))
        keyframes = [video_frames[i] for i in keyframe_indices]
        
        # Run the expensive model only on keyframes
        with torch.no_grad():
            keyframe_keypoints = self.keypoint_detector.get_court_keypoints(keyframes, read_from_stub=False)
        
        # Expand to all frames by repeating the nearest keyframe result
        all_keypoints = []
        kf_idx = 0
        for frame_idx in range(len(video_frames)):
            # Find the closest keyframe
            if kf_idx + 1 < len(keyframe_indices) and frame_idx >= keyframe_indices[kf_idx + 1]:
                kf_idx += 1
            all_keypoints.append(keyframe_keypoints[min(kf_idx, len(keyframe_keypoints) - 1)])
        
        # Free VRAM after keypoint detection
        if "cuda" in config.DEVICE:
            torch.cuda.empty_cache()
            
        return all_keypoints

    def transform_tracks(self, video_frames: list, player_tracks: list, ball_tracks: list):
        """
        Detects court keypoints (with stride optimization), validates them, 
        and transforms player positions to tactical view.
        Applies EMA smoothing to tactical positions.
        """
        if self.keypoint_detector is None:
            return player_tracks, ball_tracks
        
        # Use strided keypoint detection for performance
        self.keypoints_per_frame = self._get_keypoints_with_stride(video_frames)
        if self.keypoints_per_frame is None:
            return player_tracks, ball_tracks
            
        self.keypoints_per_frame = self.converter.validate_keypoints(self.keypoints_per_frame)
        
        raw_tactical_positions = self.converter.transform_players_to_tactical_view(self.keypoints_per_frame, player_tracks)
        
        # Apply EMA smoothing to tactical positions
        smoothed_tactical_positions = []
        for i, frame_positions in enumerate(raw_tactical_positions):
            smoothed_frame = {}
            for player_id, pos in frame_positions.items():
                pos_array = np.array(pos)
                if player_id not in self.ema_tactical_positions:
                    self.ema_tactical_positions[player_id] = pos_array
                else:
                    self.ema_tactical_positions[player_id] = 0.6 * pos_array + 0.4 * self.ema_tactical_positions[player_id]
                
                smoothed_pos = self.ema_tactical_positions[player_id].tolist()
                smoothed_frame[player_id] = smoothed_pos
                
                # Sideline filter: ignore people way outside the 2800x1500 court bounds
                # Basketball court is 28m x 15m. Allow a small 2-meter margin (200px)
                sx, sy = smoothed_pos
                if sx < -200 or sx > 3000 or sy < -200 or sy > 1700:
                    if player_id in player_tracks[i]:
                        player_tracks[i][player_id]['team'] = 0  # Ignore out-of-bounds entity
                        
            smoothed_tactical_positions.append(smoothed_frame)
            
        self.tactical_player_positions = smoothed_tactical_positions
        
        return player_tracks, ball_tracks

    def render_tactical_view(self, output_video_frames: list, player_tracks: list, ball_tracks: list):
        if self.keypoint_detector is None or self.keypoints_per_frame is None:
            return output_video_frames
            
        player_assignment = []
        for frame_tracks in player_tracks:
            assignment_dict = {}
            for player_id, data in frame_tracks.items():
                if 'team' in data:
                    assignment_dict[player_id] = data['team']
            player_assignment.append(assignment_dict)
            
        ball_aquisition = [None] * len(output_video_frames)
        
        return self.drawer.draw(
            output_video_frames,
            self.converter.court_image_path,
            self.converter.width,
            self.converter.height,
            self.converter.key_points,
            self.tactical_player_positions,
            player_assignment,
            ball_aquisition
        )
