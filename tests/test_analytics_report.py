import unittest
from core.basketball.analytics_report import CourtVisionAnalyticsReport

class TestAnalyticsReport(unittest.TestCase):
    def setUp(self):
        self.report_gen = CourtVisionAnalyticsReport()
        # Suppress writing scratch heatmap images to disk during unit tests
        self.report_gen._export_heatmaps = lambda name: None

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

if __name__ == '__main__':
    unittest.main()
