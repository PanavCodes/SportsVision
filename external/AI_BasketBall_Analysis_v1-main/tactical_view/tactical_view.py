import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils import get_foot_position, measure_distance

class TacticalViewConverter:
    """
    Converts player positions from video frames to a tactical (top-down) court view using homography.
    """

    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 200
        self.height = 107

        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15 

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

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transform player positions from camera view to tactical (top-down) court coordinates
        by computing homography from detected keypoints.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}

            # Convert keypoints to list format
            if hasattr(frame_keypoints, 'xy'):
                frame_keypoints = frame_keypoints.xy.tolist()[0]
            elif hasattr(frame_keypoints, 'tolist'):
                frame_keypoints = frame_keypoints.tolist()

            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Identify valid keypoints (with positive coordinates) for homography calculation
            valid_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0 and i < len(self.key_points)]

            # Need at least 4 points for a reliable homography estimation
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Prepare source and destination points for homography
            source_points = np.array([frame_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
            
            try:
                # Compute homography matrix mapping camera view to tactical view
                homography = Homography(source_points, target_points)
                
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    # Extract player's foot position from bounding box
                    player_position = np.array([get_foot_position(bbox)])
                    # Apply homography transformation to player position
                    tactical_position = homography.transform_points(player_position)

                    x, y = tactical_position[0][0], tactical_position[0][1]
                    # Ignore players outside the tactical court boundaries
                    if x < 0 or x > self.width or y < 0 or y > self.height:
                        continue

                    tactical_positions[player_id] = [x, y]
                    
            except (ValueError, cv2.error):
                # In case homography computation fails, return empty positions for this frame
                tactical_player_positions.append(tactical_positions)
                continue
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions
