import json
import os
import numpy as np
from datetime import datetime

import config


class CourtVisionAnalyticsReport:
    """
    Generates a comprehensive post-game analytics report as JSON + text summary.
    Accumulates statistics across all pipeline batches and writes the final report.
    """
    
    def __init__(self):
        # Possession
        self.team1_possession_frames = 0
        self.team2_possession_frames = 0
        
        # Passes & Interceptions
        self.team1_passes = 0
        self.team2_passes = 0
        self.team1_interceptions = 0
        self.team2_interceptions = 0
        
        # Per-player cumulative distance
        self.player_distances = {}  # {player_id: total_meters}
        
        # Per-player speed samples (for averaging)
        self.player_speed_samples = {}  # {player_id: [speeds]}
        
        # Per-player team assignment (majority vote)
        self.player_team_votes = {}  # {player_id: {team: count}}
        
        # Shot events
        self.shot_confidences = []
        
        # Frame count
        self.total_frames = 0
        self.fps = 30
        
        # Heatmap (x, y) spatial coordinates
        self.heatmap_points = []
    
    def update_possession(self, ball_possession: list, player_tracks: list):
        """Update possession stats from a batch of frames."""
        for frame_idx, holder in enumerate(ball_possession):
            if holder == -1:
                continue
            data = player_tracks[frame_idx].get(holder, {})
            team = data.get('team', 0)
            if team == 1:
                self.team1_possession_frames += 1
            elif team == 2:
                self.team2_possession_frames += 1
    
    def update_passes(self, passes: list, interceptions: list):
        """Update pass/interception counts from a batch."""
        for p in passes:
            if p == 1:
                self.team1_passes += 1
            elif p == 2:
                self.team2_passes += 1
        for i in interceptions:
            if i == 1:
                self.team1_interceptions += 1
            elif i == 2:
                self.team2_interceptions += 1
    
    def update_distances(self, distances: list):
        """Update per-player distance from a batch."""
        for frame_dist in distances:
            for pid, dist in frame_dist.items():
                self.player_distances[pid] = self.player_distances.get(pid, 0) + dist
    
    def update_speeds(self, speeds: list):
        """Update per-player speed samples from a batch."""
        for frame_speeds in speeds:
            for pid, speed in frame_speeds.items():
                if speed > 0:
                    if pid not in self.player_speed_samples:
                        self.player_speed_samples[pid] = []
                    self.player_speed_samples[pid].append(speed)
    
    def update_team_assignments(self, player_tracks: list):
        """Track team assignments for majority voting."""
        for frame_tracks in player_tracks:
            for pid, data in frame_tracks.items():
                team = data.get('team', 0)
                if pid not in self.player_team_votes:
                    self.player_team_votes[pid] = {}
                self.player_team_votes[pid][team] = self.player_team_votes[pid].get(team, 0) + 1
    
    def update_shots(self, confidences: list):
        """Accumulate shot confidence values."""
        self.shot_confidences.extend(confidences)
        
    def update_heatmaps(self, tactical_positions: list):
        """Accumulate (x, y) coordinates for heatmaps."""
        if not tactical_positions:
            return
        for frame_pos in tactical_positions:
            for pid, (x, y) in frame_pos.items():
                self.heatmap_points.append((pid, int(x), int(y)))
    
    def update_frame_count(self, n_frames: int):
        self.total_frames += n_frames
    
    def _get_player_team(self, pid):
        """Get the majority-voted team for a player."""
        if pid not in self.player_team_votes:
            return 0
        votes = self.player_team_votes[pid]
        return max(votes.items(), key=lambda x: x[1])[0]
    
    def generate_report(self, video_name: str) -> dict:
        """Generate the final analytics report."""
        
        # Export thermal heatmaps
        self._export_heatmaps(video_name)
        
        total_possession = self.team1_possession_frames + self.team2_possession_frames
        
        # Per-player stats
        player_stats = {}
        all_player_ids = set(self.player_team_votes.keys()) | set(self.player_distances.keys()) | set(self.player_speed_samples.keys())
        
        for pid in all_player_ids:
            team = self._get_player_team(pid)
            dist = self.player_distances.get(pid, 0)
            speeds = self.player_speed_samples.get(pid, [])
            avg_speed = round(np.mean(speeds), 1) if speeds else 0
            max_speed = round(max(speeds), 1) if speeds else 0
            
            player_stats[str(pid)] = {
                "team": team,
                "total_distance_m": round(dist, 1),
                "avg_speed_kmh": avg_speed,
                "max_speed_kmh": max_speed,
            }
        
        # Count shot events (confidence > 0.5)
        shot_count = sum(1 for c in self.shot_confidences if c >= 0.5)
        
        report = {
            "video_name": video_name,
            "generated_at": datetime.now().isoformat(),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.total_frames / self.fps, 1),
            "fps": self.fps,
            "team_stats": {
                "team_1": {
                    "possession_pct": round(self.team1_possession_frames / total_possession * 100, 1) if total_possession > 0 else 0,
                    "total_passes": self.team1_passes,
                    "total_interceptions": self.team1_interceptions,
                },
                "team_2": {
                    "possession_pct": round(self.team2_possession_frames / total_possession * 100, 1) if total_possession > 0 else 0,
                    "total_passes": self.team2_passes,
                    "total_interceptions": self.team2_interceptions,
                }
            },
            "shot_events_detected": shot_count,
            "player_stats": player_stats,
        }
        
        return report
    
    def save_report(self, video_name: str, output_dir: str = None):
        """Generate and save the report to JSON."""
        if output_dir is None:
            output_dir = config.ANALYTICS_DIR
            
        report = self.generate_report(video_name)
        
        # Save JSON
        json_path = os.path.join(output_dir, f"{video_name}_analytics.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f" POST-GAME ANALYTICS REPORT")
        print(f"{'='*60}")
        print(f" Video: {video_name}")
        print(f" Duration: {report['duration_seconds']:.0f}s ({report['duration_seconds']/60:.1f} min)")
        print(f" Frames: {report['total_frames']}")
        print(f"\n Team 1 — Possession: {report['team_stats']['team_1']['possession_pct']}%")
        print(f"          Passes: {report['team_stats']['team_1']['total_passes']}")
        print(f"          Interceptions: {report['team_stats']['team_1']['total_interceptions']}")
        print(f"\n Team 2 — Possession: {report['team_stats']['team_2']['possession_pct']}%")
        print(f"          Passes: {report['team_stats']['team_2']['total_passes']}")
        print(f"          Interceptions: {report['team_stats']['team_2']['total_interceptions']}")
        print(f"\n Shots detected: {report['shot_events_detected']}")
        print(f"\n Saved to: {json_path}")
        print(f"{'='*60}")
        
        return json_path

    def _export_heatmaps(self, video_name: str):
        import cv2
        out_name = os.path.basename(video_name).split('.')[0]
        
        # 28m x 15m tactical court scaled to pixels (e.g. 2800 x 1500)
        WIDTH, HEIGHT = 2800, 1500
        
        grid_team1 = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        grid_team2 = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        
        for pid, x, y in self.heatmap_points:
            # Scale coordinates from 200x107 to 2800x1500
            x_scaled = int((x / 200.0) * WIDTH)
            y_scaled = int((y / 107.0) * HEIGHT)
            
            if 0 <= x_scaled < WIDTH and 0 <= y_scaled < HEIGHT:
                team = self._get_player_team(pid)
                if team == 1:
                    grid_team1[y_scaled, x_scaled] += 1
                elif team == 2:
                    grid_team2[y_scaled, x_scaled] += 1
                    
        def render_and_save(grid, out_path):
            # Apply heavy Gaussian blur to create thermal blob effect
            blurred = cv2.GaussianBlur(grid, (0, 0), sigmaX=35, sigmaY=35)
            
            # Normalize 0-255
            max_val = np.max(blurred)
            if max_val > 0:
                blurred = (blurred / max_val * 255).astype(np.uint8)
            else:
                blurred = blurred.astype(np.uint8)
                
            heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
            
            court_bg = cv2.imread(config.COURT_IMAGE_PATH)
            if court_bg is not None:
                court_bg = cv2.resize(court_bg, (WIDTH, HEIGHT))
                # Add thermal heatmap onto the court
                blended = cv2.addWeighted(court_bg, 0.5, heatmap, 0.5, 0)
                cv2.imwrite(out_path, blended)
            else:
                cv2.imwrite(out_path, heatmap)
                
        render_and_save(grid_team1, os.path.join(config.OUTPUT_DIR, f"{out_name}_team1_heatmap.jpg"))
        render_and_save(grid_team2, os.path.join(config.OUTPUT_DIR, f"{out_name}_team2_heatmap.jpg"))
        print(f"[+] Exported Heatmaps for Team 1 and Team 2")
