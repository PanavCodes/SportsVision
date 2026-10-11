import unittest
import numpy as np
import cv2

from core.basketball.minimap.homography import Homography, fix_homography_flip, reprojection_error

class TestHomography(unittest.TestCase):
    def setUp(self):
        # 4 non-collinear reference court points (corners of half court in minimap pixels)
        self.court_pts = np.array([
            [0.0, 0.0],
            [0.0, 107.0],
            [100.0, 107.0],
            [100.0, 0.0]
        ], dtype=np.float32)

        # Ground-truth perspective transform
        self.H_gt = np.array([
            [1.3, 0.08, 40.0],
            [-0.04, 1.2, 70.0],
            [0.0001, 0.0002, 1.0]
        ], dtype=np.float32)

        # Generate corresponding camera image points
        H_inv = np.linalg.inv(self.H_gt)
        self.img_pts = cv2.perspectiveTransform(
            self.court_pts.reshape(-1, 1, 2), H_inv
        ).reshape(-1, 2)

    def test_four_point_homography(self):
        """Tests that Homography computes accurately with 4 minimum points."""
        hom = Homography(self.img_pts, self.court_pts, min_inliers=4, reprojection_thresh=10.0)
        self.assertIsNotNone(hom.homography_matrix)

        # Test reprojection error
        projected = hom.transform_points(self.img_pts)
        err = np.mean(np.linalg.norm(projected - self.court_pts, axis=1))
        self.assertLess(err, 1.0, f"Reprojection error too high: {err}")

    def test_flip_detection(self):
        """Tests that horizontal determinant flips are detected and corrected."""
        H_flipped = self.H_gt.copy()
        H_flipped[:, 0] *= -1 # Invert X column
        
        R = H_flipped[:2, :2]
        self.assertLess(np.linalg.det(R), 0, "Determinant should be negative for flipped matrix")

        H_fixed = fix_homography_flip(H_flipped)
        R_fixed = H_fixed[:2, :2]
        self.assertGreater(np.linalg.det(R_fixed), 0, "Determinant should be positive after fix")

    def test_fallback_on_degenerate_points(self):
        """Tests that Homography uses prev_H when given collinear degenerate points."""
        collinear_img_pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
        collinear_court_pts = np.array([[0, 0], [0, 20], [0, 40], [0, 60]], dtype=np.float32)

        prev_H = self.H_gt.copy()
        hom = Homography(collinear_img_pts, collinear_court_pts, prev_H=prev_H)
        self.assertTrue(np.allclose(hom.homography_matrix, prev_H))

if __name__ == '__main__':
    unittest.main()
