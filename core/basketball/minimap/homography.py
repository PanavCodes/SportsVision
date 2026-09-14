import numpy as np
import cv2

def fix_homography_flip(H: np.ndarray) -> np.ndarray:
    """
    Detect and correct horizontal flip in the homography matrix by checking
    the determinant of the rotation/scaling part.
    """
    R = H[:2, :2]
    if np.linalg.det(R) < 0:
        H[:, 0] *= -1
    return H

def best_keypoint_order(curr_pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    """
    Compare the keypoints ordering with its reversed version and return the order
    that yields better alignment to the reference points.
    """
    if len(curr_pts) != len(ref_pts):
        return curr_pts
    dist_normal = np.mean(np.linalg.norm(curr_pts - ref_pts, axis=1))
    dist_flipped = np.mean(np.linalg.norm(curr_pts[::-1] - ref_pts, axis=1))
    if dist_flipped < dist_normal:
        return curr_pts[::-1]
    return curr_pts

def reprojection_error(H: np.ndarray, img_pts: np.ndarray, court_pts: np.ndarray) -> float:
    """
    Compute the mean reprojection error (Euclidean distance) between points projected by H
    and the reference court points.
    """
    img_pts = img_pts.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(img_pts, H).reshape(-1, 2)
    return np.mean(np.linalg.norm(projected - court_pts, axis=1))

def blend_homographies(H1: np.ndarray, H2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two homography matrices by weighted averaging and normalization.
    """
    H_blend = alpha * H1 + (1 - alpha) * H2
    return H_blend / H_blend[-1, -1]

class Homography:
    def __init__(self,
                 image_points: np.ndarray,
                 court_points: np.ndarray,
                 prev_H: np.ndarray = None,
                 prev_image_points: np.ndarray = None,
                 min_inliers: int = 6,
                 alpha: float = 0.85,
                 reprojection_thresh: float = 40.0,
                 ransac_reproj_thresh: float = 5.0,
                 ransac_max_iters: int = 2000,
                 verbose: bool = False):
        """
        Estimate robust homography from image_points to court_points with:
         - outlier rejection via RANSAC
         - flip correction
         - reprojection error check
         - optional smoothing with previous homography
        """
        self.verbose = verbose
        self.min_inliers = min_inliers
        self.alpha = alpha
        self.reprojection_thresh = reprojection_thresh

        image_points = np.asarray(image_points, dtype=np.float32)
        court_points = np.asarray(court_points, dtype=np.float32)

        if image_points.shape != court_points.shape or image_points.shape[0] < 4:
            raise ValueError("At least 4 matching points with matching shapes are required.")

        if prev_image_points is not None:
            image_points = best_keypoint_order(image_points, prev_image_points)

        H, mask = cv2.findHomography(
            image_points,
            court_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_thresh,
            maxIters=ransac_max_iters
        )

        if H is None or mask is None:
            if prev_H is not None:
                self.homography_matrix = prev_H
                return
            else:
                raise ValueError("Homography estimation failed and no previous fallback provided.")

        inliers_count = np.sum(mask)
        if inliers_count < min_inliers:
            if prev_H is not None:
                self.homography_matrix = prev_H
                return
            else:
                raise ValueError("Insufficient inliers and no previous fallback provided.")

        H = fix_homography_flip(H)

        inlier_img_pts = image_points[mask.ravel() == 1]
        inlier_court_pts = court_points[mask.ravel() == 1]
        err = reprojection_error(H, inlier_img_pts, inlier_court_pts)

        if err > reprojection_thresh:
            if prev_H is not None:
                self.homography_matrix = prev_H
                return
            else:
                raise ValueError("Reprojection error too high and no previous fallback.")

        if prev_H is not None:
            H = blend_homographies(prev_H, H, alpha)

        self.homography_matrix = H

    def transform_points(self, image_points: np.ndarray) -> np.ndarray:
        """
        Transform points from image coordinates to court coordinates using
        the computed homography.
        """
        points = np.asarray(image_points).reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(points, self.homography_matrix).reshape(-1, 2)
