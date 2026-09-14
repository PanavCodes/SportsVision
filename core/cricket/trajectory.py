import math
from collections import deque
import numpy as np
from typing import List, Tuple, Optional, Dict

def angle_between_lines(m1, m2=0.0):
    """Calculate the angle between two lines with slopes m1 and m2."""
    try:
        if m1 * m2 == -1:
            return 90.0
        angle = math.degrees(math.atan(abs((m2 - m1) / (1.0 + m1 * m2))))
        return float(angle)
    except ZeroDivisionError:
        return 90.0

def create_bezier_curve(points, smoothness=30):
    """Generate a quadratic Bezier curve through 3 control points."""
    if len(points) < 3:
        return np.array(points, dtype=np.int32)

    t = np.linspace(0, 1, smoothness)
    p0, p1, p2 = points[0], points[1], points[2]

    curve = []
    for step in t:
        x = ((1 - step) ** 2) * p0[0] + 2 * (1 - step) * step * p1[0] + (step ** 2) * p2[0]
        y = ((1 - step) ** 2) * p0[1] + 2 * (1 - step) * step * p1[1] + (step ** 2) * p2[1]
        curve.append([int(x), int(y)])

    return np.array(curve, dtype=np.int32)


class Kalman3DSmoother:
    """
    6-State 3D Kalman Filter [X, Y, Z, vx, vy, vz] for 3D trajectory tracking,
    gravitational physics modeling, and ballistic path synthesis.
    """
    def __init__(self, fps: float = 30.0, gravity: float = 9.81):
        self.dt = 1.0 / max(fps, 1.0)
        self.g = gravity
        self.state_dim = 6
        self.meas_dim = 3

        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)

        # Transition matrix with kinematics
        self.F = np.eye(self.state_dim, dtype=np.float64)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        # Control input for gravity
        self.Bu = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.Bu[2, 0] = -0.5 * self.g * (self.dt ** 2)
        self.Bu[5, 0] = -self.g * self.dt

        # Measurement matrix
        self.H = np.zeros((self.meas_dim, self.state_dim), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.P = np.eye(self.state_dim, dtype=np.float64) * 0.05
        self.Q = np.eye(self.state_dim, dtype=np.float64) * 0.01
        self.R = np.eye(self.meas_dim, dtype=np.float64) * 0.005
        self.initialized = False

    def init_state(self, initial_pos: Tuple[float, float, float], initial_vel: Tuple[float, float, float] = (0.0, 35.0, -2.0)):
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.x[0, 0] = initial_pos[0]
        self.x[1, 0] = initial_pos[1]
        self.x[2, 0] = initial_pos[2]
        self.x[3, 0] = initial_vel[0]
        self.x[4, 0] = initial_vel[1]
        self.x[5, 0] = initial_vel[2]
        self.P = np.eye(self.state_dim, dtype=np.float64) * 0.01
        self.initialized = True

    def predict(self) -> Tuple[float, float, float]:
        if not self.initialized:
            return (0.0, 0.0, 0.0)
        self.x = self.F @ self.x + self.Bu
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))

    def update(self, measurement: Tuple[float, float, float], predict_first: bool = True) -> Tuple[float, float, float]:
        z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float64)
        if not self.initialized:
            self.init_state(measurement)
            return measurement

        if predict_first:
            self.predict()
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        residual = z - (self.H @ self.x)
        self.x = self.x + K @ residual
        I = np.eye(self.state_dim, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))


class CricketTrajectoryAnalyzer:
    """
    Hawk-Eye Grade Cricket Ball Trajectory & Prediction Analyzer:
      - Continuous arc smoothing (Bezier / polynomial)
      - Pitch bounce point detection and incident/rebound angle calculation
      - 3D ballistic trajectory extrapolation through batsman's wickets (DRS Hawk-Eye)
      - Seam/spin deviation angle measurement
      - Projection confidence cone calculation
    """
    def __init__(self, history_len: int = 30, fps: float = 30.0, pitch_length: float = 20.12):
        self.history = deque(maxlen=history_len)
        self.all_centroids = []
        self.bounce_points = []
        self.bounce_angles = []
        self.last_bounce_frame = -999
        self.fps = fps
        self.pitch_length = pitch_length
        self.kalman_3d = Kalman3DSmoother(fps=fps)
        self.delivery_archive = []

    def reset_delivery(self) -> dict:
        """
        Archives current delivery's trajectory data and clears active buffers
        ready for the next delivery in a long multi-over video.
        """
        archived = {
            'centroids': list(self.all_centroids),
            'bounce_points': list(self.bounce_points),
            'bounce_angles': list(self.bounce_angles)
        }
        if self.all_centroids:
            self.delivery_archive.append(archived)

        self.history.clear()
        self.all_centroids = []
        self.bounce_points = []
        self.bounce_angles = []
        self.last_bounce_frame = -999
        self.kalman_3d.initialized = False
        return archived

    def add_point(self, frame_idx: int, centroid: Optional[Tuple[float, float]]):
        """Record 2D ball centroid for frame."""
        if centroid is not None:
            self.history.append((frame_idx, centroid))
            self.all_centroids.append((frame_idx, centroid))
            self._detect_bounce(frame_idx)

    def _detect_bounce(self, current_frame: int):
        """Detect pitch impact (bounce point) by analyzing trajectory inflection."""
        if len(self.history) < 5:
            return

        if current_frame - self.last_bounce_frame < 10:
            return

        recent = list(self.history)[-5:]
        coords = [pt[1] for pt in recent]

        dy1 = coords[1][1] - coords[0][1]
        dy2 = coords[2][1] - coords[1][1]
        dy3 = coords[3][1] - coords[2][1]
        dy4 = coords[4][1] - coords[3][1]

        if dy1 > 0 and dy2 >= 0 and dy4 < dy2:
            bounce_coord = coords[2]
            bounce_frame = recent[2][0]

            dx_in = coords[2][0] - coords[0][0]
            dy_in = coords[2][1] - coords[0][1]
            dx_out = coords[4][0] - coords[2][0]
            dy_out = coords[4][1] - coords[2][1]

            m_in = dy_in / dx_in if dx_in != 0 else 999.0
            m_out = dy_out / dx_out if dx_out != 0 else 999.0

            angle_in = angle_between_lines(m_in, 0.0)
            angle_out = angle_between_lines(m_out, 0.0)
            deflection_angle = abs(angle_in - angle_out)

            self.bounce_points.append({
                'frame': bounce_frame,
                'coord': bounce_coord,
                'angle_in': round(angle_in, 1),
                'angle_out': round(angle_out, 1),
                'deflection': round(deflection_angle, 1)
            })
            self.last_bounce_frame = current_frame

    def get_smoothed_trail(self, max_points: int = 20) -> List[Tuple[int, int]]:
        """Return a smoothly interpolated path of recent ball positions for HUD rendering."""
        if len(self.history) < 3:
            return [pt[1] for pt in self.history]

        recent = [pt[1] for pt in list(self.history)[-max_points:]]
        if len(recent) < 3:
            return recent

        p0 = recent[0]
        mid_idx = len(recent) // 2
        p1 = recent[mid_idx]
        p2 = recent[-1]

        curve = create_bezier_curve([p0, p1, p2], smoothness=min(len(recent) * 2, 40))
        return [tuple(pt) for pt in curve]

    def project_future_path_pixels(self, steps: int = 8) -> List[Tuple[int, int]]:
        """Extrapolate 2D pixel trajectory forward in time for camera HUD overlay."""
        if len(self.history) < 3:
            return []

        recent = [pt[1] for pt in list(self.history)[-4:]]
        x_vals = [p[0] for p in recent]
        y_vals = [p[1] for p in recent]

        dx = (x_vals[-1] - x_vals[0]) / max(1, len(recent) - 1)
        dy = (y_vals[-1] - y_vals[0]) / max(1, len(recent) - 1)

        projected = []
        last_x, last_y = recent[-1]
        gravity_step = 0.5

        for i in range(1, steps + 1):
            next_x = last_x + dx * i
            next_y = last_y + dy * i + 0.5 * gravity_step * (i ** 2)
            projected.append((int(next_x), int(next_y)))

        return projected

    def project_ballistic_to_stumps(
        self,
        impact_point: Tuple[float, float, float],
        velocity_vector: Tuple[float, float, float],
        target_y: float = 20.12,
        gravity: float = 9.81,
        drag_coeff: float = 0.0070
    ) -> Tuple[List[Tuple[float, float, float]], Tuple[float, float, float]]:
        """
        Extrapolates 3D physical ballistic trajectory from pad impact point to batsman's wickets (target_y).
        Returns:
          1. List of intermediate (X, Y, Z) points along the flight
          2. Exact (X, Y, Z) point at target_y (stumps plane)
        """
        x, y, z = impact_point
        vx, vy, vz = velocity_vector
        dt = 1.0 / self.fps

        if vy <= 1.0:
            vy = 30.0  # nominal forward velocity ~108 km/h

        pts = [(float(x), float(y), float(z))]
        max_steps = 100
        step = 0

        while y < target_y and step < max_steps:
            step += 1
            x += vx * dt
            y += vy * dt
            z += vz * dt - 0.5 * gravity * (dt ** 2)

            v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
            drag_x = -drag_coeff * v_mag * vx
            drag_y = -drag_coeff * v_mag * vy
            drag_z = -drag_coeff * v_mag * vz - gravity

            vx += drag_x * dt
            vy += drag_y * dt
            vz += drag_z * dt

            pts.append((float(x), float(min(target_y, y)), float(max(0.0, z))))

        # Interpolate exact stump coordinates at target_y
        last_pt = pts[-1]
        second_last = pts[-2] if len(pts) >= 2 else last_pt

        y1, y2 = second_last[1], last_pt[1]
        span_y = max(1e-5, y2 - y1)
        interp_factor = np.clip((target_y - y1) / span_y, 0.0, 1.0)

        exact_x = second_last[0] + interp_factor * (last_pt[0] - second_last[0])
        exact_z = second_last[2] + interp_factor * (last_pt[2] - second_last[2])
        stump_impact = (float(exact_x), float(target_y), float(max(0.0, exact_z)))

        return pts, stump_impact

    @staticmethod
    def compute_seam_spin_deviation(
        pre_bounce_vel: Tuple[float, float, float],
        post_bounce_vel: Tuple[float, float, float]
    ) -> float:
        """Calculates lateral angular deviation (degrees) off the pitch turf."""
        vx_pre, vy_pre, _ = pre_bounce_vel
        vx_post, vy_post, _ = post_bounce_vel

        angle_pre = np.degrees(np.arctan2(vx_pre, max(0.1, vy_pre)))
        angle_post = np.degrees(np.arctan2(vx_post, max(0.1, vy_post)))
        return float(angle_post - angle_pre)

    @staticmethod
    def compute_confidence_cone(
        distance_projected: float,
        initial_sigma_x: float = 0.008,
        initial_sigma_z: float = 0.012
    ) -> Tuple[float, float]:
        """
        Estimates 95% confidence error cone (sigma_x, sigma_z in meters)
        at stump distance based on projected flight length.
        """
        growth_factor = 1.0 + 0.035 * (max(0.0, distance_projected) ** 1.4)
        sigma_x = initial_sigma_x * growth_factor
        sigma_z = initial_sigma_z * growth_factor
        return (float(sigma_x), float(sigma_z))
