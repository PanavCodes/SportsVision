import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography
from .bbox_utils import get_foot_position, measure_distance
from .camera_motion import CameraMotionCompensator

class TacticalViewConverter:
    """
    Converts player positions from video frames to a tactical (top-down) court view using homography.
    Enhanced with Lucas-Kanade Optical Flow Camera Motion Compensation (CMC) for persistent
    minimap registration during broadcast zooms and keypoint occlusions.
    """

    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 200
        self.height = 107

        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15 

        # Optical Flow Camera Motion Compensation engine
        self.cmc = CameraMotionCompensator(max_fallback_frames=90)
        self.prev_gray = None
        self.prev_H = None
        self.consecutive_fallback_frames = 0
        self.telemetry = {
            'total_frames': 0,
            'direct_frames': 0,
            'cmc_frames': 0,
            'lost_frames': 0
        }

        # Predefined key points representing specific tactical locations on the court
        self.key_points = [
            # Left edge points
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10 / self.actual_height_in_meters) * self.height)),
            (0, int((14.1 / self.actual_height_in_meters) * self.height)),
            (0, int(self.height)),

            # Middle line points (bottom to top)
            (int(self.width / 2), self.height),
            (int(self.width / 2), 0),
            
            # Left Free throw line points
            (int((5.79 / self.actual_width_in_meters) * self.width), int((5.18 / self.actual_height_in_meters) * self.height)),
            (int((5.79 / self.actual_width_in_meters) * self.width), int((10 / self.actual_height_in_meters) * self.height)),

            # Right edge points (bottom to top)
            (self.width, int(self.height)),
            (self.width, int((14.1 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10 / self.actual_height_in_meters) * self.height)),
            (self.width, int((5.18 / self.actual_height_in_meters) * self.height)),
            (self.width, int((0.91 / self.actual_height_in_meters) * self.height)),
            (self.width, 0),

            # Right Free throw line points
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width), int((5.18 / self.actual_height_in_meters) * self.height)),
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width), int((10 / self.actual_height_in_meters) * self.height)),
        ]

    def validate_keypoints(self, keypoints_list):
        """
        Validate detected keypoints by comparing distances between points
        against expected court proportions to filter out invalid detections.
        """
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            # Convert keypoints to list of coordinate pairs for easier processing
            if hasattr(frame_keypoints, 'xy'):
                frame_keypoints = frame_keypoints.xy.tolist()[0]
            elif hasattr(frame_keypoints, 'tolist'):
                frame_keypoints = frame_keypoints.tolist()

            # Indices of keypoints that have positive coordinates (likely detected)
            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0 and i < len(self.key_points)]

            # Skip validation if less than 3 valid keypoints detected
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            for i in detected_indices:
                # Skip if current keypoint is zeroed (already invalidated)
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Select two other valid keypoints for ratio comparison
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[0], other_indices[1]

                # Measured distances between keypoints in detected frame
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                # Expected distances between corresponding tactical keypoints
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    # Relative error between detected and tactical ratios
                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    # If error is too high, mark the keypoint as invalid
                    if error > 0.8:  # 80% margin
                        if hasattr(keypoints_list[frame_idx], 'xy'):
                            keypoints_list[frame_idx].xy[0][i] *= 0
                            keypoints_list[frame_idx].xyn[0][i] *= 0
                        else:
                            keypoints_list[frame_idx][i] *= 0
                        invalid_keypoints.append(i)
        
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks, video_frames=None):
        """
        Transform player positions from camera view to tactical (top-down) court coordinates
        by computing homography from detected keypoints, backed by Lucas-Kanade Optical Flow
        Camera Motion Compensation (CMC) fallback when keypoints are occluded or drop out.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}
            self.telemetry['total_frames'] += 1

            curr_frame = video_frames[frame_idx] if (video_frames is not None and frame_idx < len(video_frames)) else None
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame is not None else None

            # 1. Estimate inter-frame camera motion M_{t-1 -> t} via optical flow
            M_inter = None
            if self.prev_gray is not None and curr_gray is not None:
                exclude_boxes = [d['bbox'] for d in frame_tracks.values() if 'bbox' in d]
                M_inter, _ = self.cmc.estimate_interframe_motion(self.prev_gray, curr_gray, exclude_bboxes=exclude_boxes)

            # 2. Check for detected keypoints
            if hasattr(frame_keypoints, 'xy'):
                frame_keypoints = frame_keypoints.xy.tolist()[0]
            elif hasattr(frame_keypoints, 'tolist'):
                frame_keypoints = frame_keypoints.tolist()

            valid_indices = []
            if frame_keypoints is not None and len(frame_keypoints) > 0:
                valid_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0 and i < len(self.key_points)]

            H_current = None

            # Mode A: Sufficient court keypoints detected (>= 4 points)
            if len(valid_indices) >= 4:
                source_points = np.array([frame_keypoints[i] for i in valid_indices], dtype=np.float32)
                target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
                try:
                    homography = Homography(
                        source_points,
                        target_points,
                        prev_H=self.prev_H,
                        min_inliers=4,
                        reprojection_thresh=45.0
                    )
                    H_direct = homography.homography_matrix

                    # If previous homography and inter-frame motion exist, blend with predicted H to reduce jitter
                    if self.prev_H is not None and M_inter is not None:
                        H_pred = self.cmc.cascade_homography(self.prev_H, M_inter)
                        if H_pred is not None:
                            H_current = 0.85 * H_direct + 0.15 * H_pred
                            if abs(H_current[2, 2]) > 1e-6:
                                H_current = H_current / H_current[2, 2]
                        else:
                            H_current = H_direct
                    else:
                        H_current = H_direct

                    self.prev_H = H_current
                    self.consecutive_fallback_frames = 0
                    self.telemetry['direct_frames'] += 1
                except Exception:
                    H_current = None

            # Mode B: Keypoints occluded or failed — engage Optical Flow CMC Fallback
            if H_current is None:
                if self.prev_H is not None and M_inter is not None and self.consecutive_fallback_frames < self.cmc.max_fallback_frames:
                    H_cmc = self.cmc.cascade_homography(self.prev_H, M_inter)
                    if H_cmc is not None:
                        H_current = H_cmc
                        self.prev_H = H_current
                        self.consecutive_fallback_frames += 1
                        self.telemetry['cmc_frames'] += 1
                elif self.prev_H is not None and self.consecutive_fallback_frames < self.cmc.max_fallback_frames:
                    # Static fallback if optical flow had insufficient features
                    H_current = self.prev_H
                    self.consecutive_fallback_frames += 1
                    self.telemetry['cmc_frames'] += 1
                else:
                    self.telemetry['lost_frames'] += 1

            # 3. Project player foot positions onto metric minimap
            if H_current is not None:
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([[get_foot_position(bbox)]], dtype=np.float32)
                    try:
                        tactical_pos = cv2.perspectiveTransform(player_position, H_current).reshape(-1, 2)
                        x, y = float(tactical_pos[0][0]), float(tactical_pos[0][1])

                        # Allow slight 15% out-of-bounds cushion and clamp to minimap bounds [0, width] x [0, height]
                        if -self.width * 0.15 <= x <= self.width * 1.15 and -self.height * 0.15 <= y <= self.height * 1.15:
                            x_clamped = max(0.0, min(float(self.width), x))
                            y_clamped = max(0.0, min(float(self.height), y))
                            tactical_positions[player_id] = [x_clamped, y_clamped]
                    except Exception:
                        continue

            tactical_player_positions.append(tactical_positions)
            if curr_gray is not None:
                self.prev_gray = curr_gray
        
        return tactical_player_positions

    def get_telemetry_summary(self):
        """Returns empirical tracking continuity metrics for paper ablation tables."""
        total = max(1, self.telemetry['total_frames'])
        return {
            'total_frames': self.telemetry['total_frames'],
            'direct_keypoint_pct': round((self.telemetry['direct_frames'] / total) * 100, 2),
            'optical_flow_cmc_pct': round((self.telemetry['cmc_frames'] / total) * 100, 2),
            'overall_continuity_pct': round(((self.telemetry['direct_frames'] + self.telemetry['cmc_frames']) / total) * 100, 2),
            'lost_frames_pct': round((self.telemetry['lost_frames'] / total) * 100, 2)
        }

