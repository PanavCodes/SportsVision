import numpy as np


class CourtVisionSpeedCalculator:
    """
    Calculates player speed and distance using tactical view coordinates
    converted to real-world meters. Adapted from Repo 1's SpeedAndDistanceCalculator.
    
    Basketball court dimensions: 28m x 15m
    Tactical view dimensions: 300px x 161px (from Repo 1's TacticalViewConverter)
    """
    
    def __init__(self, width_px=300, height_px=161, width_m=28, height_m=15):
        self.width_px = width_px
        self.height_px = height_px
        self.width_m = width_m
        self.height_m = height_m
    
    def _pixel_to_meters(self, px_x, px_y):
        """Convert tactical view pixel coordinates to real-world meters."""
        m_x = px_x * self.width_m / self.width_px
        m_y = px_y * self.height_m / self.height_px
        return m_x, m_y
    
    def _distance(self, p1, p2):
        """Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
    
    def calculate_distances(self, tactical_positions: list) -> list:
        """
        Calculate per-frame distance traveled for each player using tactical view positions.
        
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
                    # Scale factor to account for measurement noise
                    frame_distances[player_id] = dist * 0.4
                
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
        window_size = 5  # Minimum frames needed for speed calculation
        
        for frame_idx in range(len(distances)):
            frame_speeds = {}
            
            for player_id in distances[frame_idx].keys():
                # Look back over a sliding window
                start = max(0, frame_idx - (window_size * 3) + 1)
                
                total_dist = 0
                frames_present = 0
                last_present = None
                
                for i in range(start, frame_idx + 1):
                    if player_id in distances[i]:
                        if last_present is not None:
                            total_dist += distances[i][player_id]
                            frames_present += 1
                        last_present = i
                
                if frames_present >= window_size:
                    time_sec = frames_present / fps
                    time_hrs = time_sec / 3600
                    if time_hrs > 0:
                        speed_kmh = (total_dist / 1000) / time_hrs
                        frame_speeds[player_id] = round(speed_kmh, 1)
                    else:
                        frame_speeds[player_id] = 0
                else:
                    frame_speeds[player_id] = 0
            
            speeds.append(frame_speeds)
        
        return speeds
    
    def calculate_total_distance(self, distances: list) -> dict:
        """
        Calculate total distance covered by each player across all frames.
        
        Returns:
            Dict {player_id: total_distance_in_meters}
        """
        totals = {}
        for frame_dist in distances:
            for player_id, dist in frame_dist.items():
                totals[player_id] = totals.get(player_id, 0) + dist
        return totals
