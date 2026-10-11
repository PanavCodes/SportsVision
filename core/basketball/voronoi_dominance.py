import numpy as np
import cv2
from scipy.spatial import KDTree, ConvexHull

class VoronoiDominanceAnalyzer:
    """
    Computes real-time spatial dominance, bounded Voronoi tessellations,
    and team convex hull dynamics directly on the 2D tactical court plane.
    """
    def __init__(self, width: int = 200, height: int = 107, grid_step: int = 2):
        self.width = width
        self.height = height
        self.grid_step = grid_step
        
        # Precompute court grid points for vectorized KD-Tree nearest-neighbor queries
        x_coords = np.arange(0, self.width, self.grid_step)
        y_coords = np.arange(0, self.height, self.grid_step)
        self.grid_shape = (len(y_coords), len(x_coords))

        gx, gy = np.meshgrid(x_coords, y_coords)
        self.grid_pts = np.vstack([gx.ravel(), gy.ravel()]).T # Shape (N_cells, 2)
        self.total_cells = len(self.grid_pts)

        # Cumulative game-wide dominance statistics
        self.team1_dominance_samples = []
        self.team2_dominance_samples = []

    def compute_dominance(self, player_positions: dict, player_teams: dict):
        """
        Computes metric court spatial dominance (%) for Team 1 and Team 2.
        
        Args:
            player_positions: {player_id: [x, y]} in tactical coordinates.
            player_teams: {player_id: team_id} where 1=Team 1, 2=Team 2, 0=Referee.
            
        Returns:
            dict containing dominance percentages, team convex hull areas, and control grid.
        """
        valid_pids = []
        valid_coords = []
        valid_teams = []

        for pid, pos in player_positions.items():
            team = player_teams.get(pid, 0)
            if team in [1, 2] and 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                valid_pids.append(pid)
                valid_coords.append(pos)
                valid_teams.append(team)

        if len(valid_coords) < 2:
            return {
                'team1_dominance_pct': 50.0,
                'team2_dominance_pct': 50.0,
                'team1_hull_area_m2': 0.0,
                'team2_hull_area_m2': 0.0,
                'dominant_team': 0,
                'control_mask': None
            }

        coords_arr = np.array(valid_coords, dtype=np.float32)
        teams_arr = np.array(valid_teams, dtype=np.int32)

        # 1. Query KDTree to partition the court into Voronoi cells
        tree = KDTree(coords_arr)
        _, nearest_idx = tree.query(self.grid_pts)
        cell_teams = teams_arr[nearest_idx]

        t1_count = np.sum(cell_teams == 1)
        t2_count = np.sum(cell_teams == 2)
        total_valid_cells = max(1, t1_count + t2_count)

        t1_pct = round((t1_count / total_valid_cells) * 100.0, 1)
        t2_pct = round((t2_count / total_valid_cells) * 100.0, 1)

        self.team1_dominance_samples.append(t1_pct)
        self.team2_dominance_samples.append(t2_pct)

        # 2. Compute convex hull area for each team (defensive/offensive spread in physical m^2)
        # 1 px in x = 28/200 m = 0.14m; 1 px in y = 15/107 m = 0.14m -> 1 px^2 = 0.0196 m^2
        px_to_m2 = (28.0 / self.width) * (15.0 / self.height)

        def get_hull_area(team_num):
            pts = coords_arr[teams_arr == team_num]
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    return round(hull.volume * px_to_m2, 1) # In 2D, hull.volume is area
                except Exception:
                    return 0.0
            return 0.0

        t1_hull = get_hull_area(1)
        t2_hull = get_hull_area(2)

        dominant_team = 1 if t1_pct > t2_pct else (2 if t2_pct > t1_pct else 0)

        # Build 2D control mask for visual rendering
        mask_2d = cell_teams.reshape(self.grid_shape)

        return {
            'team1_dominance_pct': t1_pct,
            'team2_dominance_pct': t2_pct,
            'team1_hull_area_m2': t1_hull,
            'team2_hull_area_m2': t2_hull,
            'dominant_team': dominant_team,
            'control_mask': mask_2d
        }

    def render_voronoi_overlay(self, court_img: np.ndarray, control_mask: np.ndarray,
                               team1_color: tuple, team2_color: tuple, alpha: float = 0.30):
        """
        Blends semi-transparent Voronoi dominance colors over the court minimap.
        """
        if court_img is None or control_mask is None:
            return court_img

        h, w = court_img.shape[:2]
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Upsample control mask to court dimensions
        resized_mask = cv2.resize(control_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply team colors
        colored_mask[resized_mask == 1] = team1_color
        colored_mask[resized_mask == 2] = team2_color

        # Alpha blend with court image
        has_color = (resized_mask == 1) | (resized_mask == 2)
        blended = court_img.copy()
        blended[has_color] = cv2.addWeighted(court_img, 1.0 - alpha, colored_mask, alpha, 0)[has_color]

        return blended

    def get_match_summary(self):
        """Returns match-wide average spatial dominance for analytics report."""
        if not self.team1_dominance_samples:
            return {'team1_avg_dominance_pct': 50.0, 'team2_avg_dominance_pct': 50.0}
        return {
            'team1_avg_dominance_pct': round(float(np.mean(self.team1_dominance_samples)), 1),
            'team2_avg_dominance_pct': round(float(np.mean(self.team2_dominance_samples)), 1)
        }
