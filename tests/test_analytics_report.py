import unittest
from core.basketball.analytics_report import CourtVisionAnalyticsReport

class TestAnalyticsReport(unittest.TestCase):
    def setUp(self):
        self.report_gen = CourtVisionAnalyticsReport()
        # Suppress writing scratch heatmap images to disk during unit tests
        self.report_gen._export_heatmaps = lambda name, *args, **kwargs: None

    def test_possession_and_pass_accumulation(self):
        """Tests that possession percentages and pass stats accumulate correctly."""
        # Simulated batch of 10 frames
        ball_possession = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] # 5 frames Team 1, 5 frames Team 2
        player_tracks = [
            {1: {'team': 1}, 2: {'team': 2}} for _ in range(10)
        ]

        self.report_gen.update_possession(ball_possession, player_tracks)
        self.report_gen.update_passes([1, 0, 1], [0, 2])
        self.report_gen.update_frame_count(10)

        report = self.report_gen.generate_report("test_video")

        self.assertEqual(report['team_stats']['team_1']['possession_pct'], 50.0)
        self.assertEqual(report['team_stats']['team_2']['possession_pct'], 50.0)
        self.assertEqual(report['team_stats']['team_1']['total_passes'], 2)
        self.assertEqual(report['team_stats']['team_2']['total_interceptions'], 1)

    def test_tracking_continuity_stats_included(self):
        """Tests that CMC tracking continuity metrics are exported into final analytics JSON."""
        cmc_stats = {
            'total_frames': 500,
            'direct_keypoint_pct': 72.4,
            'optical_flow_cmc_pct': 26.2,
            'overall_continuity_pct': 98.6,
            'lost_frames_pct': 1.4
        }
        self.report_gen.update_cmc_telemetry(cmc_stats)
        report = self.report_gen.generate_report("game_test")

        self.assertIn('tracking_continuity_stats', report)
        self.assertEqual(report['tracking_continuity_stats']['overall_continuity_pct'], 98.6)
        self.assertEqual(report['tracking_continuity_stats']['optical_flow_cmc_pct'], 26.2)

    def test_heatmap_boundary_and_sideline_filtering(self):
        """Tests that out-of-bounds coordinates and sideline entities are excluded from heatmaps."""
        tactical_positions = [
            {
                1: [100.0, 50.0],  # Valid center court
                2: [0.0, 40.0],    # Exact left boundary edge (should be dropped)
                3: [-10.0, 30.0],  # Out-of-bounds left (should be dropped)
                4: [120.0, 108.0], # Out-of-bounds bottom (should be dropped)
                5: [50.0, 50.0],   # Valid on-court position, but marked as sideline
            }
        ]
        player_tracks = [
            {
                1: {'team': 1, 'is_sideline': False},
                2: {'team': 1, 'is_sideline': False},
                3: {'team': 2, 'is_sideline': True},
                4: {'team': 2, 'is_sideline': False},
                5: {'team': 0, 'is_sideline': True},
            }
        ]

        self.report_gen.update_heatmaps(tactical_positions, player_tracks)
        
        # Only player 1 should be retained
        retained_pids = [pt[0] for pt in self.report_gen.heatmap_points]
        self.assertEqual(retained_pids, [1])

    def test_referee_exclusion_from_player_stats(self):
        """Tests that match officials and referees are strictly excluded from player_stats."""
        player_tracks = [
            {
                1: {'team': 1, 'jersey_number': '23', 'is_referee': False},
                2: {'team': 0, 'is_referee': True},
                3: {'team': 1, 'is_referee': True},
            }
            for _ in range(20)
        ]
        self.report_gen.update_team_assignments(player_tracks)
        self.report_gen.update_distances([{1: 5.0, 2: 12.0, 3: 8.0} for _ in range(20)])
        self.report_gen.update_speeds([{1: 15.0, 2: 18.0, 3: 14.0} for _ in range(20)])
        self.report_gen.update_frame_count(20)

        report = self.report_gen.generate_report('test_game')
        stats = report['player_stats']

        # Only Player 1 should be present; Referee 2 and Official 3 must be excluded
        self.assertIn('1', stats)
        self.assertNotIn('2', stats)
        self.assertNotIn('3', stats)
        self.assertEqual(stats['1']['team'], 1)
        self.assertEqual(stats['1']['jersey_number'], '23')

    def test_jersey_matching_and_track_consolidation(self):
        """Tests that fragmented tracks with the same team and jersey are consolidated into one player."""
        # Track 4 appears in frames 0..19 (Team 1, Jersey 7)
        batch1 = [
            {4: {'team': 1, 'jersey_number': '7', 'is_referee': False}}
            for _ in range(20)
        ]
        self.report_gen.update_team_assignments(batch1)
        self.report_gen.update_distances([{4: 5.0} for _ in range(20)]) # 100.0m
        self.report_gen.update_speeds([{4: 10.0} for _ in range(20)])
        self.report_gen.update_frame_count(20)

        # Track 19 appears in frames 20..34 after a cut (Same Team 1, Same Jersey 7)
        batch2 = [
            {19: {'team': 1, 'jersey_number': '7', 'is_referee': False}}
            for _ in range(15)
        ]
        self.report_gen.update_team_assignments(batch2)
        self.report_gen.update_distances([{19: 4.0} for _ in range(15)]) # 60.0m
        self.report_gen.update_speeds([{19: 16.0} for _ in range(15)])
        self.report_gen.update_frame_count(15)

        report = self.report_gen.generate_report('test_game')
        stats = report['player_stats']

        # Should be consolidated under primary track 4
        self.assertIn('4', stats)
        self.assertNotIn('19', stats)
        player_7 = stats['4']
        self.assertEqual(player_7['jersey_number'], '7')
        self.assertEqual(set(player_7['track_ids']), {4, 19})
        self.assertEqual(player_7['total_distance_m'], 160.0)
        self.assertEqual(player_7['max_speed_kmh'], 16.0)

if __name__ == '__main__':
    unittest.main()
