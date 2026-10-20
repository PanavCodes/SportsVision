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

        # Per-player jersey numbers (majority vote)
        self.player_jersey_votes = {}  # {player_id: {jersey_str: count}}

        # Per-player referee and frame tracking
        self.player_referee_votes = {}  # {player_id: count}
        self.player_frame_counts = {}   # {player_id: count}
        self.player_active_frames = {}  # {player_id: set(frame_indices)}
        
        # Shot events
        self.shot_confidences = []
        
        # Frame count
        self.total_frames = 0
        self.fps = 30
        
        # Heatmap (x, y) spatial coordinates
        self.heatmap_points = []

        # Camera Motion Compensation (CMC) telemetry
        self.cmc_telemetry = None

        # Voronoi Spatial Dominance statistics
        self.voronoi_stats = None

    def update_cmc_telemetry(self, cmc_stats: dict):
        """Record camera motion compensation and homography continuity telemetry."""
        self.cmc_telemetry = cmc_stats

    def update_voronoi_stats(self, voronoi_stats: dict):
        """Record match-wide Voronoi spatial dominance statistics."""
        self.voronoi_stats = voronoi_stats

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
                self.player_distances[pid] = self.player_distances.get(pid, 0.0) + dist
    
    def update_speeds(self, speeds: list):
        """Update per-player speed samples from a batch."""
        for frame_speeds in speeds:
            for pid, speed in frame_speeds.items():
                if speed > 0:
                    if pid not in self.player_speed_samples:
                        self.player_speed_samples[pid] = []
                    self.player_speed_samples[pid].append(speed)
    
    def update_team_assignments(self, player_tracks: list):
        """Track team assignments, jersey numbers, and referee flags for players."""
        for frame_offset, frame_tracks in enumerate(player_tracks):
            global_frame = self.total_frames + frame_offset
            for pid, data in frame_tracks.items():
                self.player_frame_counts[pid] = self.player_frame_counts.get(pid, 0) + 1
                
                if pid not in self.player_active_frames:
                    self.player_active_frames[pid] = set()
                self.player_active_frames[pid].add(global_frame)
                
                team = data.get('team', 0)
                if pid not in self.player_team_votes:
                    self.player_team_votes[pid] = {}
                self.player_team_votes[pid][team] = self.player_team_votes[pid].get(team, 0) + 1
                
                if data.get('is_referee', False) or data.get('is_sideline', False) or team == 0:
                    self.player_referee_votes[pid] = self.player_referee_votes.get(pid, 0) + 1
                
                jersey = data.get('jersey_number')
                if jersey is not None:
                    jersey_str = str(jersey).strip()
                    if jersey_str and jersey_str.isdigit():
                        if pid not in self.player_jersey_votes:
                            self.player_jersey_votes[pid] = {}
                        self.player_jersey_votes[pid][jersey_str] = self.player_jersey_votes[pid].get(jersey_str, 0) + 1
    
    def update_shots(self, confidences: list):
        """Accumulate shot confidence values."""
        self.shot_confidences.extend(confidences)
        
    def update_heatmaps(self, tactical_positions: list, player_tracks: list = None):
        """
        Accumulate (x, y) coordinates for heatmaps.
        Filters out points outside the active playing court boundaries to prevent edge artifacts.
        Optionally uses frame-level player tracks to capture real-time team assignments.
        """
        if not tactical_positions:
            return
        for f_idx, frame_pos in enumerate(tactical_positions):
            frame_tracks = player_tracks[f_idx] if (player_tracks and f_idx < len(player_tracks)) else {}
            for pid, (x, y) in frame_pos.items():
                # Strictly filter out coordinates outside the court lines [1, 199] x [1, 106].
                # This eliminates boundary wall accumulation from sideline entities.
                if x < 1.0 or x > 199.0 or y < 1.0 or y > 106.0:
                    continue
                
                # Check frame team tag if available
                team_override = None
                if frame_tracks and pid in frame_tracks:
                    p_data = frame_tracks[pid]
                    if p_data.get('is_sideline', False):
                        continue
                    team_override = p_data.get('team', None)
                    if team_override == 0:
                        continue
                
                self.heatmap_points.append((pid, float(x), float(y), team_override))
    
    def update_frame_count(self, n_frames: int):
        self.total_frames += n_frames
    
    def _get_player_team(self, pid):
        """Get the majority-voted team for a player."""
        if pid not in self.player_team_votes:
            return 0
        votes = self.player_team_votes[pid]
        return max(votes.items(), key=lambda x: x[1])[0]
        
    def _is_referee(self, pid) -> bool:
        """Determines if a track is a referee or match official."""
        team = self._get_player_team(pid)
        if team == 0:
            return True
        ref_votes = self.player_referee_votes.get(pid, 0)
        total = self.player_frame_counts.get(pid, 1)
        return (ref_votes / max(1, total)) >= 0.25

    def _get_player_jersey(self, pid):
        """Get the majority-voted jersey number for a player track."""
        if pid not in self.player_jersey_votes:
            return None
        votes = self.player_jersey_votes[pid]
        if not votes:
            return None
        return max(votes.items(), key=lambda x: x[1])[0]
    
    def generate_report(self, video_name: str) -> dict:
        """Generate the final analytics report."""
        
        total_possession = self.team1_possession_frames + self.team2_possession_frames
        
        # Per-player stats: strictly exclude Match Officials / Referees (team == 0)
        # Consolidate tracks belonging to the same player (matching team and jersey number)
        all_player_ids = set(self.player_team_votes.keys()) | set(self.player_distances.keys()) | set(self.player_speed_samples.keys())
        
        valid_tracks = []
        for pid in all_player_ids:
            # Strictly filter out referees and match officials
            if self._is_referee(pid):
                continue
            team = self._get_player_team(pid)
            if team not in (1, 2):
                continue
                
            dist = self.player_distances.get(pid, 0.0)
            frames_count = self.player_frame_counts.get(pid, 0)
            speeds = self.player_speed_samples.get(pid, [])
            
            # Suppress transient ghost tracks from cuts/false detections (< 10 frames and < 1.0m)
            if frames_count > 0 and frames_count < 10 and dist < 1.0 and len(speeds) < 5:
                continue
                
            jersey = self._get_player_jersey(pid)
            active_frames = self.player_active_frames.get(pid, set())
            
            valid_tracks.append({
                'pid': pid,
                'team': team,
                'jersey': jersey,
                'dist': dist,
                'speeds': speeds,
                'frames_count': frames_count,
                'active_frames': active_frames
            })
            
        # Group by (team, jersey) where jersey is known, resolving any overlapping conflicts
        jersey_tracks = {}
        unassigned_tracks = []
        
        for t in valid_tracks:
            j = t['jersey']
            if j is not None:
                key = (t['team'], j)
                if key not in jersey_tracks:
                    jersey_tracks[key] = []
                jersey_tracks[key].append(t)
            else:
                unassigned_tracks.append(t)
                
        consolidated_groups = []
        for (team, j), t_list in jersey_tracks.items():
            if len(t_list) == 1:
                consolidated_groups.append(t_list)
            else:
                # Sort descending by votes for this jersey
                t_list.sort(key=lambda tr: self.player_jersey_votes.get(tr['pid'], {}).get(j, 0), reverse=True)
                primary_group = [t_list[0]]
                for other in t_list[1:]:
                    # Check temporal overlap: two distinct players on the court simultaneously
                    has_overlap = any(len(other['active_frames'] & p['active_frames']) > 10 for p in primary_group)
                    if not has_overlap:
                        primary_group.append(other)
                    else:
                        other['jersey'] = None
                        unassigned_tracks.append(other)
                consolidated_groups.append(primary_group)
                
        player_stats = {}
        all_groups = consolidated_groups + [[t] for t in unassigned_tracks]
        for grp in all_groups:
            # Sort by frame count so primary pid is the longest active track segment
            grp.sort(key=lambda tr: tr['frames_count'], reverse=True)
            primary_pid = grp[0]['pid']
            team = grp[0]['team']
            jersey = grp[0]['jersey']
            
            track_ids = [tr['pid'] for tr in grp]
            total_dist = sum(tr['dist'] for tr in grp)
            combined_speeds = []
            for tr in grp:
                combined_speeds.extend(tr['speeds'])
                
            avg_speed = round(float(np.mean(combined_speeds)), 1) if combined_speeds else 0.0
            max_speed = round(float(max(combined_speeds)), 1) if combined_speeds else 0.0
            player_label = f"#{jersey}" if jersey else f"Track #{primary_pid}"
            
            player_stats[str(primary_pid)] = {
                "team": team,
                "jersey_number": jersey,
                "track_ids": track_ids,
                "display_label": player_label,
                "total_distance_m": round(float(total_dist), 1),
                "avg_speed_kmh": avg_speed,
                "max_speed_kmh": max_speed,
            }
        
        # Export thermal heatmaps with player_stats context for stationary entity filtering
        self._export_heatmaps(video_name, player_stats)
        
        # Count shot events (confidence >= 0.5)
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
            "spatial_dominance_stats": self.voronoi_stats or {
                "team1_avg_dominance_pct": 50.0,
                "team2_avg_dominance_pct": 50.0
            },
            "shot_events_detected": shot_count,
            "player_stats": player_stats,
            "tracking_continuity_stats": self.cmc_telemetry or {
                "direct_keypoint_pct": 100.0,
                "optical_flow_cmc_pct": 0.0,
                "overall_continuity_pct": 100.0,
                "lost_frames_pct": 0.0
            }
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
        print(f" SportsVision Basketball Post-Game Analytics Report")
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
        if self.voronoi_stats:
            print(f"\n Spatial Voronoi Dominance:")
            print(f"   Team 1 Court Dominance: {self.voronoi_stats.get('team1_avg_dominance_pct', 50.0)}%")
            print(f"   Team 2 Court Dominance: {self.voronoi_stats.get('team2_avg_dominance_pct', 50.0)}%")
        if self.cmc_telemetry:
            print(f"\n Camera Motion Tracking Continuity (CMC):")
            print(f"   Direct Keypoints: {self.cmc_telemetry.get('direct_keypoint_pct', 0)}%")
            print(f"   Optical Flow Fallback: {self.cmc_telemetry.get('optical_flow_cmc_pct', 0)}%")
            print(f"   Overall Continuity: {self.cmc_telemetry.get('overall_continuity_pct', 0)}%")
        print(f"\n Shots detected: {report['shot_events_detected']}")
        print(f"\n Saved to: {json_path}")
        print(f"{'='*60}")
        
        return json_path

    def _export_heatmaps(self, video_name: str, player_stats: dict = None):
        import cv2
        out_name = os.path.basename(video_name).split('.')[0]
        
        WIDTH, HEIGHT = 2800, 1500
        
        grid_team1 = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        grid_team2 = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        grid_match = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        
        # Stationary entity filter:
        # In matches with >= 90 frames, filter out tracks that never move (benches, coaches, static refs).
        # A real on-court player covers distance and moves faster than 0.8 km/h.
        stationary_pids = set()
        if player_stats and self.total_frames >= 90:
            for pid_str, stats in player_stats.items():
                pids = stats.get('track_ids', [int(pid_str) if pid_str.isdigit() else pid_str])
                if stats.get('total_distance_m', 0.0) < 3.0 or stats.get('avg_speed_kmh', 0.0) < 0.8:
                    for p in pids:
                        stationary_pids.add(p)

        for pt in self.heatmap_points:
            if len(pt) == 4:
                pid, x, y, team_override = pt
            else:
                pid, x, y = pt
                team_override = None
                
            if pid in stationary_pids:
                continue

            x_scaled = int((x / 200.0) * WIDTH)
            y_scaled = int((y / 107.0) * HEIGHT)
            
            # Additional boundary safeguard: ignore points within 15px of outer canvas edges
            if 15 <= x_scaled < (WIDTH - 15) and 15 <= y_scaled < (HEIGHT - 15):
                team = team_override if (team_override in [1, 2]) else self._get_player_team(pid)
                if team == 1:
                    grid_team1[y_scaled, x_scaled] += 1
                    grid_match[y_scaled, x_scaled] += 1
                elif team == 2:
                    grid_team2[y_scaled, x_scaled] += 1
                    grid_match[y_scaled, x_scaled] += 1
                    
        def render_and_save(grid, out_path):
            if np.sum(grid) == 0:
                court_bg = cv2.imread(config.COURT_IMAGE_PATH)
                if court_bg is not None:
                    court_bg = cv2.resize(court_bg, (WIDTH, HEIGHT))
                    cv2.imwrite(out_path, court_bg)
                return

            # Broadcast-standard Gaussian KDE smoothing (~0.65m radius at 2800x1500)
            blurred = cv2.GaussianBlur(grid, (0, 0), sigmaX=65, sigmaY=65)
            
            # Robust 98th percentile normalization prevents a single static cluster from washing out dynamic player heat
            pos_vals = blurred[blurred > 0]
            if len(pos_vals) > 0:
                vmax = np.percentile(pos_vals, 98)
                if vmax <= 0:
                    vmax = np.max(blurred)
                norm = np.clip(blurred / vmax, 0.0, 1.0)
                norm_8u = (norm * 255).astype(np.uint8)
            else:
                norm_8u = np.zeros_like(blurred, dtype=np.uint8)
                norm = np.zeros_like(blurred, dtype=np.float32)

            heatmap = cv2.applyColorMap(norm_8u, cv2.COLORMAP_JET)
            court_bg = cv2.imread(config.COURT_IMAGE_PATH)
            if court_bg is None:
                cv2.imwrite(out_path, heatmap)
                return
                
            court_bg = cv2.resize(court_bg, (WIDTH, HEIGHT))
            
            # Alpha compositing: 0 alpha where heat is below 0.05 threshold (crystal-clear hardwood court!)
            # Smooth alpha ramp up to 0.70 for professional broadcast thermal visual
            thresh = 0.05
            alpha = np.zeros_like(norm, dtype=np.float32)
            mask = norm >= thresh
            alpha[mask] = 0.20 + 0.50 * ((norm[mask] - thresh) / (1.0 - thresh))
            alpha = np.clip(alpha, 0.0, 0.70)
            alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            blended = (court_bg.astype(np.float32) * (1.0 - alpha_3d) + heatmap.astype(np.float32) * alpha_3d).astype(np.uint8)
            cv2.imwrite(out_path, blended)
                
        render_and_save(grid_team1, os.path.join(config.OUTPUT_DIR, f"{out_name}_team1_heatmap.jpg"))
        render_and_save(grid_team2, os.path.join(config.OUTPUT_DIR, f"{out_name}_team2_heatmap.jpg"))
        render_and_save(grid_match, os.path.join(config.OUTPUT_DIR, f"{out_name}_match_heatmap.jpg"))
