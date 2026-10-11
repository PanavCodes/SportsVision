import unittest
import numpy as np
import cv2

from core.basketball.voronoi_dominance import VoronoiDominanceAnalyzer

class TestVoronoiDominance(unittest.TestCase):
    def setUp(self):
        self.analyzer = VoronoiDominanceAnalyzer(width=200, height=107, grid_step=2)

    def test_balanced_half_court_dominance(self):
        """Tests that balanced half-court team spacing yields approximately 50-50 dominance."""
        # Team 1 on left half, Team 2 on right half
        positions = {
            1: [30.0, 30.0],
            2: [30.0, 75.0],
            3: [60.0, 50.0],
            4: [140.0, 50.0],
            5: [170.0, 30.0],
            6: [170.0, 75.0]
        }
        teams = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}

        res = self.analyzer.compute_dominance(positions, teams)

        self.assertAlmostEqual(res['team1_dominance_pct'], 50.0, delta=5.0)
        self.assertAlmostEqual(res['team2_dominance_pct'], 50.0, delta=5.0)
        self.assertIsNotNone(res['control_mask'])

    def test_offensive_overload_dominance_shift(self):
        """Tests that when Team 1 overloads the frontcourt, Team 1 dominance increases."""
        # Team 1 spreads across 75% of the court, compressing Team 2
        positions = {
            1: [40.0, 50.0],
            2: [80.0, 30.0],
            3: [120.0, 50.0],
            4: [150.0, 30.0],
            5: [180.0, 50.0], # Team 2 pushed deep into backcourt corner
            6: [185.0, 60.0]
        }
        teams = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2}

        res = self.analyzer.compute_dominance(positions, teams)

        self.assertGreater(res['team1_dominance_pct'], 60.0)
        self.assertEqual(res['dominant_team'], 1)

    def test_convex_hull_area_computation(self):
        """Tests that team convex hull spatial spreading is computed in physical m^2."""
        # 3 points forming a triangle of known area
        positions = {
            1: [10.0, 10.0],
            2: [10.0, 50.0],
            3: [50.0, 10.0],
            4: [150.0, 20.0] # Team 2 (only 1 player, no hull)
        }
        teams = {1: 1, 2: 1, 3: 1, 4: 2}

        res = self.analyzer.compute_dominance(positions, teams)
        self.assertGreater(res['team1_hull_area_m2'], 5.0)
        self.assertEqual(res['team2_hull_area_m2'], 0.0)

    def test_overlay_rendering(self):
        """Tests that Voronoi partition overlay blends correctly over minimap image."""
        court = np.zeros((107, 200, 3), dtype=np.uint8)
        mask = np.ones((53, 100), dtype=np.uint8)
        mask[:, 50:] = 2

        blended = self.analyzer.render_voronoi_overlay(
            court, mask, team1_color=(255, 0, 0), team2_color=(0, 0, 255), alpha=0.3
        )
        self.assertEqual(blended.shape, (107, 200, 3))
        self.assertGreater(np.max(blended), 0)

if __name__ == '__main__':
    unittest.main()
