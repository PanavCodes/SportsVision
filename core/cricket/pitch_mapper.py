import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
import config

class CricketPitchMapper:
    """
    Cricket Pitch Homography & Spatial Coordinate Transformer:
      1. Maps broadcast camera coordinates to real-world 22-yard pitch meters
         (Length: 20.12m between bowling creases, Width: 3.05m).
      2. Maps points onto the 2D top-down minimap image (data/cricket_pitch.png).
      3. Classifies bowling Line & Length according to standard cricket broadcast analytics.
    """
    def __init__(self, pitch_template_path=None):
        self.pitch_template_path = pitch_template_path or config.CRICKET_PITCH_IMAGE_PATH
        self.pitch_img = None
        if os.path.exists(self.pitch_template_path):
            self.pitch_img = cv2.imread(self.pitch_template_path)

        # Standard 22-yard pitch dimensions (meters)
        self.pitch_length = getattr(config, 'CRICKET_PITCH_LENGTH_METERS', 20.12)
        self.pitch_width = getattr(config, 'CRICKET_PITCH_WIDTH_METERS', 3.05)

        # Minimap template coordinates in data/cricket_pitch.png (600x320)
        # Pitch rect: (80, 120) to (520, 200)
        # Bowler stumps: (120, 160) -> 0m
        # Batsman stumps: (480, 160) -> 20.12m
        self.minimap_bowler_stumps = np.array([120.0, 160.0])
        self.minimap_batsman_stumps = np.array([480.0, 160.0])
        self.minimap_pitch_width_px = 80.0

        # Perspective homography matrix (camera frame -> real pitch meters)
        # Default nominal perspective polygon for standard broadcast high-angle camera
        self.H_cam_to_metric = None
        self.H_cam_to_minimap = None
        self._init_default_homography()

    def _init_default_homography(self, frame_w=1280, frame_h=720):
        """
        Initialize nominal perspective transform for a typical 720p/1080p cricket broadcast.
        Camera source trapezoid: pitch area in broadcast view.
        """
        src_pts = np.array([
            [frame_w * 0.44, frame_h * 0.35],  # bowler crease top-left
            [frame_w * 0.56, frame_h * 0.35],  # bowler crease top-right
            [frame_w * 0.65, frame_h * 0.75],  # batsman crease bottom-right
            [frame_w * 0.35, frame_h * 0.75],  # batsman crease bottom-left
        ], dtype=np.float32)

        # Target real-world metric coordinates:
        # X: 0.0m (bowler stumps) to 20.12m (batsman stumps)
        # Y: -1.52m (left edge) to +1.52m (right edge)
        dst_metric = np.array([
            [0.0, -self.pitch_width / 2.0],
            [0.0, self.pitch_width / 2.0],
            [self.pitch_length, self.pitch_width / 2.0],
            [self.pitch_length, -self.pitch_width / 2.0],
        ], dtype=np.float32)

        # Target minimap pixel coordinates on 600x320 template
        dst_minimap = np.array([
            [120.0, 160.0 - self.minimap_pitch_width_px / 2.0],
            [120.0, 160.0 + self.minimap_pitch_width_px / 2.0],
            [480.0, 160.0 + self.minimap_pitch_width_px / 2.0],
            [480.0, 160.0 - self.minimap_pitch_width_px / 2.0],
        ], dtype=np.float32)

        self.H_cam_to_metric, _ = cv2.findHomography(src_pts, dst_metric)
        self.H_cam_to_minimap, _ = cv2.findHomography(src_pts, dst_minimap)

    def calibrate_pitch(self, src_trapezoid):
        """
        Calibrate homography using 4 detected pitch corner keypoints in image.
        src_trapezoid: 4x2 array of (x, y) coordinates.
        """
        if len(src_trapezoid) == 4:
            src = np.array(src_trapezoid, dtype=np.float32)
            dst_metric = np.array([
                [0.0, -self.pitch_width / 2.0],
                [0.0, self.pitch_width / 2.0],
                [self.pitch_length, self.pitch_width / 2.0],
                [self.pitch_length, -self.pitch_width / 2.0],
            ], dtype=np.float32)

            dst_minimap = np.array([
                [120.0, 160.0 - self.minimap_pitch_width_px / 2.0],
                [120.0, 160.0 + self.minimap_pitch_width_px / 2.0],
                [480.0, 160.0 + self.minimap_pitch_width_px / 2.0],
                [480.0, 160.0 - self.minimap_pitch_width_px / 2.0],
            ], dtype=np.float32)

            self.H_cam_to_metric, _ = cv2.findHomography(src, dst_metric)
            self.H_cam_to_minimap, _ = cv2.findHomography(src, dst_minimap)

    def to_metric(self, px, py):
        """
        Convert pixel coordinate (px, py) in camera frame to real-world metric pitch (X, Y) in meters.
        Returns: (length_from_bowler_m, lateral_offset_m) or (None, None)
        """
        if self.H_cam_to_metric is None:
            return None, None

        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_cam_to_metric)
        mx, my = transformed[0][0]

        # Clamp within reasonable pitch + run-up bounds (-5m to +25m)
        if -5.0 <= mx <= 26.0 and -6.0 <= my <= 6.0:
            return float(mx), float(my)
        return None, None

    def to_minimap(self, px, py):
        """
        Convert pixel coordinate in camera frame to (x, y) coordinates on data/cricket_pitch.png template.
        """
        if self.H_cam_to_minimap is None:
            return None

        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_cam_to_minimap)
        mx, my = transformed[0][0]

        if 0 <= mx <= 600 and 0 <= my <= 320:
            return int(mx), int(my)
        return None

    def to_minimap_multi(self, points):
        """
        Convert a list of (x, y) camera pixel coordinates to minimap coordinates.
        """
        if not points or self.H_cam_to_minimap is None:
            return []

        pts = np.array([[[float(p[0]), float(p[1])]] for p in points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H_cam_to_minimap)

        minimap_pts = []
        for pt in transformed:
            mx, my = pt[0]
            if 0 <= mx <= 600 and 0 <= my <= 320:
                minimap_pts.append((int(mx), int(my)))
        return minimap_pts

    def metric_to_minimap(self, mx: float, my: float):
        """
        Directly maps metric pitch coordinates (mx = 0..20.12m along pitch, my = -1.52..+1.52m lateral)
        to template pixel coordinates on the 600x320 pitch image.
        """
        if mx is None or my is None:
            return None
        # Bowler stumps at (120, 160), Batsman stumps at (480, 160)
        px = 120.0 + (float(mx) / self.pitch_length) * 360.0
        py = 160.0 + (float(my) / self.pitch_width) * self.minimap_pitch_width_px
        if 0 <= px <= 600 and 0 <= py <= 320:
            return int(px), int(py)
        return None

    def metric_to_ground(self, metric_x: float, metric_y: float) -> Optional[Tuple[int, int]]:
        """
        Maps real-world pitch/field metric coordinates (mx in [0, 20.12], my in [-70, +70])
        onto the top-down cricket ground image (538x530, center=(265, 263), boundary radius=254).
        """
        if metric_x is None or metric_y is None:
            return None

        # Inside pitch area (0 <= mx <= 20.12, |my| <= 1.52)
        # In ground graphic: Bowler crease is near (265, 185), Batsman crease is near (265, 340)
        if 0.0 <= metric_x <= self.pitch_length and abs(metric_y) <= (self.pitch_width / 2.0 + 0.5):
            px = 265.0 + metric_y * (24.5 / (self.pitch_width / 2.0))
            py = 185.0 + (metric_x / self.pitch_length) * 155.0
        else:
            # Field / outfield coordinates relative to pitch center (mx=10.06, my=0.0)
            rel_x = metric_y
            rel_y = metric_x - (self.pitch_length / 2.0)
            scale = 254.0 / 70.0  # ~3.63 px/meter
            px = 265.0 + rel_x * scale
            py = 263.0 + rel_y * scale

        # Clamp within ground boundary circle (R=250 px)
        dx = px - 265.0
        dy = py - 263.0
        dist = np.hypot(dx, dy)
        if dist > 250.0:
            px = 265.0 + (dx / dist) * 250.0
            py = 263.0 + (dy / dist) * 250.0

        return int(round(px)), int(round(py))

    def to_ground_map(self, cam_x: float, cam_y: float) -> Optional[Tuple[int, int]]:
        """Maps broadcast camera coordinate directly to cricket ground image coordinate."""
        mx, my = self.to_metric(cam_x, cam_y)
        if mx is not None and my is not None:
            return self.metric_to_ground(mx, my)
        return None

    def to_ground_map_multi(self, points: list) -> list:
        """Batch transforms camera trajectory points to ground image coordinates."""
        ground_pts = []
        for pt in points:
            if pt and pt[0] is not None and pt[1] is not None:
                gpt = self.to_ground_map(pt[0], pt[1])
                if gpt is not None:
                    ground_pts.append(gpt)
        return ground_pts

    def classify_line_and_length(self, metric_x, metric_y):
        """
        Classify bowling delivery line and length given pitch metric coordinates.
        metric_x: length from bowler stumps in meters (0 to 20.12)
        metric_y: lateral offset from pitch center in meters (-1.52 to +1.52)
        Returns:
            line_label (str): e.g. "Outside Off", "On the Stumps", "Leg Stump / Pad Line"
            length_label (str): e.g. "Good Length", "Yorker", "Full", "Short of Length"
        """
        if metric_x is None:
            return "On the Stumps", "Good Length"

        # Distance from batsman stumps:
        dist_from_batsman = max(0.0, self.pitch_length - metric_x)

        # 1. Length Classification
        if dist_from_batsman < 2.0:
            length_label = "Yorker"
        elif dist_from_batsman < 4.0:
            length_label = "Full"
        elif dist_from_batsman < 6.5:
            length_label = "Good Length"
        elif dist_from_batsman < 8.5:
            length_label = "Short of Length"
        else:
            length_label = "Bouncer / Short"

        # 2. Line Classification (assuming right-handed batsman stance)
        # pitch center is 0.0m. Stumps width is ~0.23m (-0.115m to +0.115m)
        if metric_y is None:
            line_label = "Middle"
        elif metric_y < -0.60:
            line_label = "Wide Outside Off"
        elif metric_y < -0.15:
            line_label = "Outside Off"
        elif metric_y <= 0.15:
            line_label = "On the Stumps"
        elif metric_y <= 0.60:
            line_label = "Leg Stump / Pad Line"
        else:
            line_label = "Down Leg Side"

        return line_label, length_label
