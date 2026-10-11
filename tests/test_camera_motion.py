import unittest
import numpy as np
import cv2

from core.basketball.minimap.camera_motion import CameraMotionCompensator

class TestCameraMotionCompensator(unittest.TestCase):
    def setUp(self):
        self.cmc = CameraMotionCompensator(
            max_features=300,
            quality_level=0.01,
            min_distance=8,
            ransac_thresh=3.0,
            max_fallback_frames=90
        )
        self.h, self.w = 720, 1280

        # Create a synthetic high-contrast textured background simulating court & lines
        np.random.seed(42)
        self.base_frame = np.random.randint(60, 200, (self.h, self.w), dtype=np.uint8)
        # Add high-contrast grid lines
        for y in range(80, self.h, 50):
            cv2.line(self.base_frame, (0, y), (self.w, y), 255, 3)
        for x in range(80, self.w, 80):
            cv2.line(self.base_frame, (x, 0), (x, self.h), 255, 3)

    def test_pure_translation_pan(self):
        """Tests that CMC accurately estimates camera pan motion."""
        dx_gt, dy_gt = 20.0, 10.0
        M_gt = np.float32([[1.0, 0.0, dx_gt], [0.0, 1.0, dy_gt]])
        curr_frame = cv2.warpAffine(self.base_frame, M_gt, (self.w, self.h))

        M_est, inliers = self.cmc.estimate_interframe_motion(self.base_frame, curr_frame)
        self.assertIsNotNone(M_est, "CMC should successfully estimate translation")
        self.assertGreater(inliers, 30, f"Expected >30 inliers, got {inliers}")

        dx_est = M_est[0, 2]
        dy_est = M_est[1, 2]
        self.assertAlmostEqual(dx_est, dx_gt, delta=0.5, msg="Estimated dx deviates from ground truth")
        self.assertAlmostEqual(dy_est, dy_gt, delta=0.5, msg="Estimated dy deviates from ground truth")

    def test_zoom_scale(self):
        """Tests that CMC accurately detects uniform zoom scale changes."""
        scale_gt = 1.03
        cx, cy = self.w / 2, self.h / 2
        M_gt_2x3 = cv2.getRotationMatrix2D((cx, cy), 0, scale_gt)
        curr_frame = cv2.warpAffine(self.base_frame, M_gt_2x3, (self.w, self.h))

        M_est, inliers = self.cmc.estimate_interframe_motion(self.base_frame, curr_frame)
        self.assertIsNotNone(M_est, "CMC should successfully estimate zoom")
        
        scale_est = np.sqrt(M_est[0, 0]**2 + M_est[0, 1]**2)
        self.assertAlmostEqual(scale_est, scale_gt, delta=0.03, msg="Estimated scale deviates from ground truth")

    def test_exclude_bboxes_mask(self):
        """Tests that foreground player bounding boxes are successfully excluded from motion estimation."""
        # Create a moving foreground object inside bbox [400, 300, 600, 500]
        f1 = self.base_frame.copy()
        f2 = self.base_frame.copy()
        
        # Add high-contrast block moving oppositely inside player bbox
        cv2.rectangle(f1, (400, 300), (600, 500), 255, -1)
        cv2.rectangle(f2, (450, 350), (650, 550), 255, -1) # moved by (+50, +50)

        # Static background has 0 motion
        exclude_bboxes = [[400, 300, 600, 500]]
        M_est, inliers = self.cmc.estimate_interframe_motion(f1, f2, exclude_bboxes=exclude_bboxes)
        
        self.assertIsNotNone(M_est)
        dx_est = M_est[0, 2]
        dy_est = M_est[1, 2]
        # Should detect static background motion (~0, 0), ignoring the player motion (+50, +50)
        self.assertAlmostEqual(dx_est, 0.0, delta=1.0)
        self.assertAlmostEqual(dy_est, 0.0, delta=1.0)

    def test_cascade_homography_math(self):
        """Tests mathematical correctness of H_t = H_{t-1} * M^{-1}."""
        # Simulated previous homography
        H_prev = np.array([
            [1.5, 0.1, 50.0],
            [-0.1, 1.4, 80.0],
            [0.0002, 0.0001, 1.0]
        ], dtype=np.float64)

        # Motion M: translation (dx=10, dy=5)
        M = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 5.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        H_curr = self.cmc.cascade_homography(H_prev, M)
        self.assertIsNotNone(H_curr)

        # Verify: point p in frame t-1 at (200, 150) moves to (210, 155) in frame t
        p_prev = np.array([[[200.0, 150.0]]], dtype=np.float32)
        p_curr = np.array([[[210.0, 155.0]]], dtype=np.float32)

        court_prev = cv2.perspectiveTransform(p_prev, H_prev).reshape(-1, 2)[0]
        court_curr = cv2.perspectiveTransform(p_curr, H_curr).reshape(-1, 2)[0]

        # Both points represent the exact same spot in real physical court space
        diff = np.linalg.norm(court_prev - court_curr)
        self.assertAlmostEqual(diff, 0.0, delta=0.01, msg="Cascaded homography did not preserve court registration")

if __name__ == '__main__':
    unittest.main()
