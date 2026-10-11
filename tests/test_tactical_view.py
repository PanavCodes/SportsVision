import unittest
import numpy as np
import cv2

import config
from core.basketball.minimap.tactical_view import TacticalViewConverter

class TestTacticalViewConverter(unittest.TestCase):
    def setUp(self):
        self.converter = TacticalViewConverter(config.COURT_IMAGE_PATH)
        self.h, self.w = 720, 1280

        # Define 4 non-collinear keypoint correspondences:
        # keypoint 0 (top-left: 0, 0), 5 (bottom-left: 0, 107), 6 (bottom-mid: 100, 107), 7 (top-mid: 100, 0)
        self.idx_list = [0, 5, 6, 7]
        self.target_pts = np.array([self.converter.key_points[i] for i in self.idx_list], dtype=np.float32)

        # Ground-truth camera homography mapping screen coords -> tactical minimap
        self.src_pts = np.array([
            [150.0, 120.0],
            [180.0, 600.0],
            [640.0, 580.0],
            [600.0, 140.0]
        ], dtype=np.float32)

        # Generate textured frames for optical flow tracking
        np.random.seed(123)
        self.f1_gray = np.random.randint(60, 200, (self.h, self.w), dtype=np.uint8)
        for y in range(80, self.h, 50): cv2.line(self.f1_gray, (0, y), (self.w, y), 255, 3)
        for x in range(80, self.w, 80): cv2.line(self.f1_gray, (x, 0), (x, self.h), 255, 3)

        # Camera pans right by 18px and down by 6px in frame 2
        self.dx, self.dy = 18.0, 6.0
        M_pan = np.float32([[1.0, 0.0, self.dx], [0.0, 1.0, self.dy]])
        self.f2_gray = cv2.warpAffine(self.f1_gray, M_pan, (self.w, self.h))

        self.color_f1 = cv2.cvtColor(self.f1_gray, cv2.COLOR_GRAY2BGR)
        self.color_f2 = cv2.cvtColor(self.f2_gray, cv2.COLOR_GRAY2BGR)

    def test_direct_keypoints_and_cmc_fallback_sequence(self):
        """
        Tests two-frame sequence:
          Frame 0: Direct keypoints present -> Homography computed directly.
          Frame 1: 100% keypoint dropout (0 keypoints) -> Optical Flow CMC fallback engaged.
        Verifies that tactical coordinates remain persistent with near-zero drift.
        """
        # Frame 0 keypoints
        kp_f0 = np.zeros((18, 2), dtype=np.float32)
        for k, i in enumerate(self.idx_list):
            kp_f0[i] = self.src_pts[k]

        # Frame 1 keypoints: complete dropout (0 valid points)
        kp_f1 = np.zeros((18, 2), dtype=np.float32)

        # Player at screen position (400, 350) in frame 0, shifted by camera pan in frame 1
        px, py = 400.0, 350.0
        player_tracks = [
            {10: {'bbox': [px - 20, py - 60, px + 20, py]}},
            {10: {'bbox': [px - 20 + self.dx, py - 60 + self.dy, px + 20 + self.dx, py + self.dy]}}
        ]

        results = self.converter.transform_players_to_tactical_view(
            [kp_f0, kp_f1],
            player_tracks,
            video_frames=[self.color_f1, self.color_f2]
        )

        self.assertEqual(len(results), 2)
        self.assertIn(10, results[0], "Player should be mapped in Frame 0 (Direct Keypoints)")
        self.assertIn(10, results[1], "Player should be mapped in Frame 1 (CMC Fallback)")

        # Verify physical tactical court position consistency
        p0 = np.array(results[0][10])
        p1 = np.array(results[1][10])
        drift = np.linalg.norm(p0 - p1)
        self.assertLess(drift, 1.5, f"Tactical drift under CMC fallback too large: {drift:.2f} px")

        # Verify telemetry statistics
        telemetry = self.converter.get_telemetry_summary()
        self.assertEqual(telemetry['total_frames'], 2)
        self.assertEqual(telemetry['direct_keypoint_pct'], 50.0)
        self.assertEqual(telemetry['optical_flow_cmc_pct'], 50.0)
        self.assertEqual(telemetry['overall_continuity_pct'], 100.0)
        self.assertEqual(telemetry['lost_frames_pct'], 0.0)

    def test_out_of_bounds_clamping(self):
        """Tests that players slightly outside sidelines are clamped to valid minimap range."""
        kp_f0 = np.zeros((18, 2), dtype=np.float32)
        for k, i in enumerate(self.idx_list):
            kp_f0[i] = self.src_pts[k]

        # Place player slightly outside court border
        player_tracks = [
            {99: {'bbox': [130, 100, 150, 110]}}
        ]

        results = self.converter.transform_players_to_tactical_view(
            [kp_f0],
            player_tracks,
            video_frames=[self.color_f1]
        )

        self.assertEqual(len(results), 1)
        if 99 in results[0]:
            x, y = results[0][99]
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, self.converter.width)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, self.converter.height)

if __name__ == '__main__':
    unittest.main()
