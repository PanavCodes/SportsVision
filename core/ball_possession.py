import numpy as np
from collections import deque, Counter


class CourtVisionBallPossession:
    """
    Detects which player has possession of the ball based on spatial proximity
    and bounding box containment. Adapted from Repo 1's BallAquisitionDetector,
    rewritten to be self-contained (no sys.path hacks).
    
    Uses:
    - Ball-to-player distance via key body points
    - Ball containment ratio (what % of ball bbox is inside player bbox)
    - Temporal smoothing with a sliding window majority vote
    - Grace period to maintain possession during brief ball-loss frames
    """
    def __init__(self):
        self.possession_threshold = 120  # Increased for 720p upscale
        self.smoothing_window = 5        # Frames for majority vote
        self.containment_threshold = 0.8 # Min ball containment ratio to prefer containment
        
    def _get_center(self, bbox):
        """Get center point of a bounding box [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _measure_distance(self, p1, p2):
        """Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
    
    def _get_key_points(self, player_bbox, ball_center):
        """
        Compute key points around a player's bounding box for proximity checks.
        Includes body alignment points and standard bbox anchor points.
        """
        bx, by = ball_center
        x1, y1, x2, y2 = player_bbox
        w = x2 - x1
        h = y2 - y1
        
        points = []
        
        # Vertical alignment: if ball is between top and bottom of player
        if y1 < by < y2:
            points.append((x1, by))
            points.append((x2, by))
        
        # Horizontal alignment: if ball is between left and right of player
        if x1 < bx < x2:
            points.append((bx, y1))
            points.append((bx, y2))
        
        # Standard anchor points around the bbox
        points += [
            (x1 + w / 2, y1),             # top center
            (x2, y1),                       # top right
            (x1, y1),                       # top left
            (x2, y1 + h / 2),             # center right
            (x1, y1 + h / 2),             # center left
            (x1 + w / 2, y1 + h / 2),     # center
            (x2, y2),                       # bottom right
            (x1, y2),                       # bottom left
            (x1 + w / 2, y2),             # bottom center
            (x1 + w / 2, y1 + h / 3),     # upper-mid center
        ]
        return points
    
    def _ball_containment_ratio(self, player_bbox, ball_bbox):
        """Calculate what fraction of the ball bbox is inside the player bbox."""
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox
        
        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)
        
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        ball_area = (bx2 - bx1) * (by2 - by1)
        return intersection / ball_area if ball_area > 0 else 0.0
    
    def _find_best_candidate(self, ball_center, ball_bbox, player_tracks_frame):
        """Find the player most likely holding the ball."""
        high_containment = []
        regular_candidates = []
        
        for player_id, player_data in player_tracks_frame.items():
            bbox = player_data.get('bbox', [])
            if not bbox:
                continue
                
            containment = self._ball_containment_ratio(bbox, ball_bbox)
            key_points = self._get_key_points(bbox, ball_center)
            min_dist = min(self._measure_distance(ball_center, pt) for pt in key_points)
            
            if containment > self.containment_threshold:
                high_containment.append((player_id, containment))
            else:
                regular_candidates.append((player_id, min_dist))
        
        # Prefer high containment candidates (max containment)
        if high_containment:
            return max(high_containment, key=lambda x: x[1])[0]
        
        # Otherwise pick closest within threshold (min distance)
        if regular_candidates:
            best = min(regular_candidates, key=lambda x: x[1])
            if best[1] < self.possession_threshold:
                return best[0]
        
        return -1
    
    def detect(self, player_tracks: list, ball_tracks: list) -> list:
        """
        Detect ball possession across a batch of frames.
        
        Args:
            player_tracks: List of dicts, one per frame. {player_id: {'bbox': [x1,y1,x2,y2], ...}}
            ball_tracks: List of dicts, one per frame. {1: {'bbox': [x1,y1,x2,y2]}} or empty
            
        Returns:
            List of player IDs (one per frame). -1 = no possession detected.
        """
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        candidate_buffer = deque(maxlen=self.smoothing_window)
        last_valid_player = -1
        grace_period = 3
        grace_counter = 0
        
        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info or not ball_info.get('bbox'):
                if grace_counter < grace_period:
                    possession_list[frame_num] = last_valid_player
                    grace_counter += 1
                continue
            
            ball_bbox = ball_info['bbox']
            ball_center = self._get_center(ball_bbox)
            
            best_player = self._find_best_candidate(
                ball_center, ball_bbox, player_tracks[frame_num]
            )
            
            candidate_buffer.append(best_player)
            
            if len(candidate_buffer) == self.smoothing_window:
                common, count = Counter(candidate_buffer).most_common(1)[0]
                if common != -1 and count >= self.smoothing_window // 2:
                    possession_list[frame_num] = common
                    last_valid_player = common
                    grace_counter = 0
                else:
                    possession_list[frame_num] = last_valid_player
                    grace_counter += 1
            else:
                possession_list[frame_num] = last_valid_player
        
        return possession_list
