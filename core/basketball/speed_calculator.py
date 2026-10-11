import numpy as np


class CourtVisionSpeedCalculator:
    """
    Calculates player speed and distance using tactical view coordinates
    calibrated to standard FIBA real-world dimensions (28m x 15m).
    
    Tactical minimap dimensions: 200px x 107px (from TacticalViewConverter)
    Court physical dimensions: 28.0m x 15.0m
    """
    
    def __init__(self, width_px=200, height_px=107, width_m=28.0, height_m=15.0):
        self.width_px = width_px
        self.height_px = height_px
        self.width_m = width_m
        self.height_m = height_m
        # Scale factors (meters per pixel)
        self.m_per_px_x = self.width_m / self.width_px
        self.m_per_px_y = self.height_m / self.height_px
    
    def _pixel_to_meters(self, px_x, px_y):
        """Convert tactical view pixel coordinates to real-world meters."""
        return px_x * self.m_per_px_x, px_y * self.m_per_px_y
    
    def _distance(self, p1, p2):
        """Euclidean distance in physical meters between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
    
    def calculate_distances(self, tactical_positions: list) -> list:
        """
        Calculate per-frame distance traveled for each player in physical meters.
        Includes stationary jitter deadband and track-swap outlier suppression.
        
        Args:
            tactical_positions: List of dicts per frame. {player_id: [x, y]} in tactical pixels.
            
        Returns:
            List of dicts per frame. {player_id: distance_in_meters}
        """
        previous_positions = {}
        distances = []
        
        for frame_positions in tactical_positions:
            frame_distances = {}
            
            for player_id, pos in frame_positions.items():
                if player_id in previous_positions:
                    prev_pos = previous_positions[player_id]
                    prev_m = self._pixel_to_meters(prev_pos[0], prev_pos[1])
                    curr_m = self._pixel_to_meters(pos[0], pos[1])
                    dist = self._distance(prev_m, curr_m)
                    
                    # Outlier suppression: ignore jumps > 2.5m per frame (likely ID swap)
                    # Deadband filtering: ignore sub-centimeter jitter (< 0.02m)
                    if 0.02 <= dist <= 2.5:
                        frame_distances[player_id] = dist
                    elif dist < 0.02:
                        frame_distances[player_id] = 0.0
                    else:
                        frame_distances[player_id] = 0.0
                else:
                    frame_distances[player_id] = 0.0
                
                previous_positions[player_id] = pos
            
            distances.append(frame_distances)
        
        return distances
    
    def calculate_speeds(self, distances: list, fps: int = 30) -> list:
        """
        Calculate player speeds in km/h using a sliding window over distance data.
        
        Args:
            distances: Output from calculate_distances()
            fps: Video frames per second
            
        Returns:
            List of dicts per frame. {player_id: speed_in_kmh}
        """
        speeds = []
        window_size = 5  # Sliding window across 5 frames
        
        for frame_idx in range(len(distances)):
            frame_speeds = {}
            
            for player_id in distances[frame_idx].keys():
                start = max(0, frame_idx - (window_size * 2) + 1)
                
                total_dist = 0.0
                frames_present = 0
                
                for i in range(start, frame_idx + 1):
                    if player_id in distances[i]:
                        total_dist += distances[i][player_id]
                        frames_present += 1
                
                if frames_present >= window_size and fps > 0:
                    time_sec = frames_present / fps
                    speed_mps = total_dist / time_sec  # meters per second
                    speed_kmh = speed_mps * 3.6        # km/h
                    # Realistic human sprinting cap (38 km/h max sprint speed)
                    frame_speeds[player_id] = round(min(speed_kmh, 38.0), 1)
                else:
                    frame_speeds[player_id] = 0.0
            
            speeds.append(frame_speeds)
        
        return speeds
    
    def calculate_total_distance(self, distances: list) -> dict:
        """
        Calculate total distance covered by each player across all frames in meters.
        
        Returns:
            Dict {player_id: total_distance_in_meters}
        """
        totals = {}
        for frame_dist in distances:
            for player_id, dist in frame_dist.items():
                totals[player_id] = totals.get(player_id, 0.0) + dist
        return {pid: round(d, 2) for pid, d in totals.items()}
