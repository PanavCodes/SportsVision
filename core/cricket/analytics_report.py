import json
import os
from datetime import datetime
import numpy as np

import config

class CricketAnalyticsReport:
    """
    Generates structured Cricket match, telemetry, and commentary analytics reports.
    Exports comprehensive JSON summaries to data/output/cricket/stats/<video_name>.json
    and displays broadcast summary stats and commentary in console.
    """
    def __init__(self, fps=30):
        self.fps = fps
        self.total_frames = 0
        self.deliveries = []
        self.all_speeds = []
        self.total_runs = 0
        self.wickets = 0
        self.boundaries_4 = 0
        self.boundaries_6 = 0
        self.dot_balls = 0
        self.shots_counter = {}
        self.line_distribution = {}
        self.length_distribution = {}
        self.pitch_map_points = []
        self.commentary_log = []
        self.fused_highlights = []

    def update_delivery(self, delivery_data, commentary_text: str = ""):
        if commentary_text:
            delivery_data['commentary'] = commentary_text
            self.commentary_log.append(commentary_text)

        self.deliveries.append(delivery_data)

        spd = delivery_data.get('release_speed_kmh', 0.0)
        if spd > 0:
            self.all_speeds.append(spd)

        runs = delivery_data.get('runs', 0)
        self.total_runs += runs

        outcome = delivery_data.get('outcome', 'dot_ball')
        if outcome in ['wicket', 'lbw_wicket'] or delivery_data.get('is_wicket', False):
            self.wickets += 1
        elif outcome == 'four':
            self.boundaries_4 += 1
        elif outcome == 'six':
            self.boundaries_6 += 1
        elif outcome in ['dot_ball', 'lbw_not_out']:
            self.dot_balls += 1

        shot = delivery_data.get('shot_played')
        if shot:
            self.shots_counter[shot] = self.shots_counter.get(shot, 0) + 1

        line = delivery_data.get('line')
        if line:
            self.line_distribution[line] = self.line_distribution.get(line, 0) + 1

        length = delivery_data.get('length')
        if length:
            self.length_distribution[length] = self.length_distribution.get(length, 0) + 1

        pitch_coord = delivery_data.get('bounce_coord_pitch_metric')
        if pitch_coord:
            self.pitch_map_points.append(pitch_coord)

    def record_highlights(self, highlight_frames: list):
        self.fused_highlights = highlight_frames

    def save_report(self, video_name: str) -> str:
        """
        Write JSON report to data/output/cricket/stats/<video_name>.json
        and print summary.
        """
        stats_dir = getattr(
            config, 'CRICKET_STATS_DIR',
            os.path.join(config.WORKSPACE_ROOT, "data", "output", "cricket", "stats")
        )
        os.makedirs(stats_dir, exist_ok=True)
        report_file = os.path.join(stats_dir, f"{video_name}.json")

        total_dels = len(self.deliveries)
        avg_speed = float(np.mean(self.all_speeds)) if self.all_speeds else 0.0
        max_speed = float(np.max(self.all_speeds)) if self.all_speeds else 0.0
        min_speed = float(np.min(self.all_speeds)) if self.all_speeds else 0.0

        boundary_runs = (self.boundaries_4 * 4) + (self.boundaries_6 * 6)
        estimated_runs = max(self.total_runs, boundary_runs)

        report = {
            'metadata': {
                'sport': 'cricket',
                'engine': 'SportsVision Cricket Spatial Analytics Engine v2.2',
                'video_name': video_name,
                'timestamp': datetime.now().isoformat(),
                'fps': self.fps,
                'total_frames': self.total_frames
            },
            'summary': {
                'total_deliveries': total_dels,
                'overs': f"{total_dels // 6}.{total_dels % 6}",
                'estimated_runs': estimated_runs,
                'wickets': self.wickets,
                'fours': self.boundaries_4,
                'sixes': self.boundaries_6,
                'dot_balls': self.dot_balls,
                'dot_ball_pct': round((self.dot_balls / max(1, total_dels)) * 100.0, 1),
                'audio_visual_highlights_count': len(self.fused_highlights),
                'speed_analysis_kmh': {
                    'fastest_delivery': round(max_speed, 1),
                    'slowest_delivery': round(min_speed, 1),
                    'average_delivery': round(avg_speed, 1)
                }
            },
            'tactical_distributions': {
                'line': self.line_distribution,
                'length': self.length_distribution,
                'shot_types': self.shots_counter
            },
            'commentary_transcript': self.commentary_log,
            'deliveries': self.deliveries
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 60)
        print(" SportsVision Cricket Match Analytics & Commentary Summary")
        print("=" * 60)
        print(f"  Total Deliveries: {total_dels} ({total_dels // 6}.{total_dels % 6} Overs)")
        print(f"  Wickets: {self.wickets} | Boundaries: {self.boundaries_4}x4, {self.boundaries_6}x6")
        print(f"  Speed Gun: Avg {avg_speed:.1f} km/h | Max {max_speed:.1f} km/h")
        if self.commentary_log:
            print("\n  [Ball-by-Ball Commentary]")
            for line in self.commentary_log[-5:]:
                print(f"    • {line}")
        print(f"\n  Analytics Report Saved: {report_file}")
        print("=" * 60)

        return report_file
