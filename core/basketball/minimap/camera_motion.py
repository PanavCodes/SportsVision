import cv2
import numpy as np

class CameraMotionCompensator:
    """
    Camera Motion Compensation (CMC) engine using sparse Lucas-Kanade optical flow.
    Estimates the inter-frame transformation M_{t-1 -> t} of the background court,
    allowing cascaded homography projection H_t = H_{t-1} * (M_{t-1 -> t})^{-1}
    when court keypoints are occluded, zoomed in, or partially out of view.
    """
    def __init__(self,
                 max_features: int = 350,
                 quality_level: float = 0.01,
                 min_distance: int = 8,
                 ransac_thresh: float = 3.0,
                 max_fallback_frames: int = 90):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_thresh = ransac_thresh
        self.max_fallback_frames = max_fallback_frames
        
        # Stats tracking for academic evaluation
        self.total_processed_frames = 0
        self.direct_keypoint_frames = 0
        self.optical_flow_fallback_frames = 0
        self.lost_tracking_frames = 0

    def estimate_interframe_motion(self, prev_gray: np.ndarray, curr_gray: np.ndarray, exclude_bboxes: list = None):
        """
        Estimates the 2D affine transformation from prev_gray to curr_gray
        using static background features (excluding players and broadcast graphics).
        
        Returns:
            M: 3x3 homogeneous transformation matrix mapping points in prev_frame to curr_frame,
               or None if motion estimation failed.
            inlier_count: Number of tracked inlier feature points.
        """
        if prev_gray is None or curr_gray is None:
            return None, 0

        h, w = prev_gray.shape[:2]
        
        # 1. Construct static background mask
        # Exclude top 12% of frame (typical broadcast scorebug / HUD banner)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[:int(h * 0.12), :] = 0
        
        # Exclude dynamic foreground objects (players, referees)
        if exclude_bboxes:
            for bbox in exclude_bboxes:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # Expand slightly to cover motion blur and player shadows
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                bx1 = max(0, x1 - pad_x)
                by1 = max(0, y1 - pad_y)
                bx2 = min(w, x2 + pad_x)
                by2 = min(h, y2 + pad_y)
                mask[by1:by2, bx1:bx2] = 0

        # 2. Extract good features to track on previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask
        )

        if prev_pts is None or len(prev_pts) < 12:
            return None, 0

        # 3. Calculate Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if curr_pts is None or status is None:
            return None, 0

        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < 8:
            return None, 0

        # 4. Estimate robust rigid/affine motion with RANSAC
        # Partial affine estimates rotation + uniform scale + translation (4 DOF).
        # This prevents projective singularity / shearing artifacts during zooming.
        M_affine, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=1500
        )

        if M_affine is None or inliers is None:
            return None, 0

        inlier_count = int(np.sum(inliers))
        if inlier_count < 6:
            return None, 0

        # Check for degenerate scale or distortion
        scale = np.sqrt(M_affine[0, 0]**2 + M_affine[0, 1]**2)
        if scale < 0.6 or scale > 1.6:  # Unrealistic camera zoom change within a single frame
            return None, 0

        # Convert 2x3 affine to 3x3 homogeneous matrix
        M_3x3 = np.eye(3, dtype=np.float64)
        M_3x3[:2, :] = M_affine

        return M_3x3, inlier_count

    def cascade_homography(self, prev_H: np.ndarray, M_prev_to_curr: np.ndarray) -> np.ndarray:
        """
        Computes the current homography H_t from H_{t-1} and the inter-frame camera motion M:
            x^{(t)} = M * x^{(t-1)}  ==>  x^{(t-1)} = M^{-1} * x^{(t)}
            X^{court} = H_{t-1} * x^{(t-1)} = H_{t-1} * M^{-1} * x^{(t)}
            => H_t = H_{t-1} * (M)^{-1}
        """
        if prev_H is None or M_prev_to_curr is None:
            return None

        try:
            M_inv = np.linalg.inv(M_prev_to_curr)
            curr_H = prev_H @ M_inv
            # Normalize so bottom-right element is 1.0
            if abs(curr_H[2, 2]) > 1e-6:
                curr_H = curr_H / curr_H[2, 2]
            return curr_H
        except np.linalg.LinAlgError:
            return prev_H.copy()
