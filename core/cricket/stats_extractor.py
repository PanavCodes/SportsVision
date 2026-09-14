import numpy as np
from collections import Counter

# Standard Delivery Metric Schema Keys
KEY_DELIVERY_NUM = 'delivery_number'
KEY_OVER = 'over'
KEY_RELEASE_SPEED = 'release_speed_kmh'
KEY_BOUNCE_SPEED = 'bounce_speed_kmh'
KEY_AVG_SPEED = 'avg_speed_kmh'
KEY_PITCH_DECELERATION = 'pitch_deceleration_pct'
KEY_LINE = 'line'
KEY_LENGTH = 'length'
KEY_BOUNCE_COORD = 'bounce_coord_pitch_metric'
KEY_SHOT = 'shot_played'
KEY_SHOT_CONF = 'shot_confidence'
KEY_OUTCOME = 'outcome'
KEY_RUNS = 'runs'
KEY_IS_WICKET = 'is_wicket'
KEY_COMMENTARY = 'commentary'

class CricketStatsExtractor:
    """
    Cricket Match Statistical Telemetry & Aggregation Extractor:
      Ties together detection, tracking, trajectory analysis, pitch mapping,
      and shot classification into structured ball-by-ball and aggregate match statistics.
    """
    def __init__(self):
        self.deliveries = []
        self.aggregate_stats = {
            'total_deliveries': 0,
            'total_runs_estimated': 0,
            'wickets': 0,
            'boundaries_four': 0,
            'boundaries_six': 0,
            'dot_balls': 0,
            'pitch_bounce_coordinates': [],
            'speeds': [],
            'shots_played': Counter(),
            'line_distribution': {},
            'length_distribution': {}
        }

    def record_delivery(
        self,
        delivery_number,
        speeds_dict,
        bounce_dict,
        line_length_tuple,
        shot_tuple,
        event_dict,
        over_str=None
    ):
        """
        Record and structure full metrics for a single bowled delivery.
        """
        shot_label, shot_conf = shot_tuple if shot_tuple else ("defense", 0.5)
        line_label, length_label = line_length_tuple if line_length_tuple else ("Middle", "Good Length")

        outcome = event_dict.get('outcome', 'dot_ball')
        runs = event_dict.get('runs', 0)
        if outcome == 'four':
            runs = 4
        elif outcome == 'six':
            runs = 6
        elif outcome == 'runs' and runs == 0:
            runs = 1

        delivery_stat = {
            'delivery_number': int(delivery_number),
            'over': over_str or f"0.{delivery_number}",
            'release_speed_kmh': speeds_dict.get('release_speed_kmh', 0.0),
            'bounce_speed_kmh': speeds_dict.get('bounce_speed_kmh', 0.0),
            'avg_speed_kmh': speeds_dict.get('avg_speed_kmh', 0.0),
            'pitch_deceleration_pct': speeds_dict.get('pitch_deceleration_pct', 0.0),
            'line': line_label,
            'length': length_label,
            'bounce_coord_pitch_metric': bounce_dict.get('coord_metric', None),
            'bounce_angle_deg': bounce_dict.get('angle_out', 0.0),
            'bounce_deflection_deg': bounce_dict.get('deflection', 0.0),
            'shot_played': shot_label,
            'shot_confidence': round(float(shot_conf), 2),
            'outcome': outcome,
            'runs': runs,
            'is_wicket': event_dict.get('is_wicket', False)
        }

        self.deliveries.append(delivery_stat)

        # Update Aggregates
        self.aggregate_stats['total_deliveries'] += 1
        self.aggregate_stats['total_runs_estimated'] += runs

        if outcome == 'dot_ball':
            self.aggregate_stats['dot_balls'] += 1
        elif outcome == 'four':
            self.aggregate_stats['boundaries_four'] += 1
        elif outcome == 'six':
            self.aggregate_stats['boundaries_six'] += 1
        elif event_dict.get('is_wicket', False):
            self.aggregate_stats['wickets'] += 1

        rel_speed = speeds_dict.get('release_speed_kmh', 0.0)
        if rel_speed > 0:
            self.aggregate_stats['speeds'].append(rel_speed)

        if shot_label:
            self.aggregate_stats['shots_played'][shot_label] += 1

        if bounce_dict.get('coord_metric'):
            self.aggregate_stats['pitch_bounce_coordinates'].append(bounce_dict['coord_metric'])

        return delivery_stat

    def get_innings_summary(self):
        """
        Compute aggregate analytics summary for the processed video.
        """
        speeds = self.aggregate_stats['speeds']
        avg_spd = float(np.mean(speeds)) if speeds else 0.0
        max_spd = float(np.max(speeds)) if speeds else 0.0
        min_spd = float(np.min(speeds)) if speeds else 0.0

        total_del = max(1, self.aggregate_stats['total_deliveries'])
        dot_pct = (self.aggregate_stats['dot_balls'] / total_del) * 100.0

        return {
            'total_deliveries': self.aggregate_stats['total_deliveries'],
            'overs': f"{self.aggregate_stats['total_deliveries'] // 6}.{self.aggregate_stats['total_deliveries'] % 6}",
            'total_runs': self.aggregate_stats['total_runs_estimated'],
            'wickets': self.aggregate_stats['wickets'],
            'boundaries_four': self.aggregate_stats['boundaries_four'],
            'boundaries_six': self.aggregate_stats['boundaries_six'],
            'dot_ball_percentage': round(dot_pct, 1),
            'speed_stats': {
                'fastest_kmh': round(max_spd, 1),
                'slowest_kmh': round(min_spd, 1),
                'average_kmh': round(avg_spd, 1)
            },
            'shot_distribution': dict(self.aggregate_stats['shots_played']),
            'deliveries': self.deliveries
        }
