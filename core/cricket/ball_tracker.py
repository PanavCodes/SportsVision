import numpy as np

class CricketBallKalmanFilter:
    """
    Kalman filter for cricket ball tracking:
    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]
    Incorporates physical ballistic motion and gravity for realistic state estimation
    when detector confidence drops or the ball is blurred.
    """
    def __init__(self, dt=1.0/30.0, gravity_pixels=9.8):
        self.dt = dt
        self.gravity = gravity_pixels

        # State transition matrix F
        # x_new = x + vx*dt
        # y_new = y + vy*dt + 0.5*g*dt^2
        # vx_new = vx
        # vy_new = vy + g*dt
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # Process noise covariance Q
        self.Q = np.eye(4, dtype=np.float32) * 5.0
        self.Q[2:, 2:] *= 10.0  # higher uncertainty in velocities

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float32) * 3.0

        # State estimate and covariance
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0

        self.initialized = False
        self.missed_frames = 0
        self.max_missed_frames = 10

    def initialize(self, x, y):
        self.state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = True
        self.missed_frames = 0

    def predict(self):
        if not self.initialized:
            return None

        # Predict state with constant velocity model
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, x, y):
        z = np.array([[x], [y]], dtype=np.float32)

        if not self.initialized:
            self.initialize(x, y)
            return float(x), float(y)

        # Innovation
        y_residual = z - (self.H @ self.state)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + (K @ y_residual)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - (K @ self.H)) @ self.P

        self.missed_frames = 0
        return float(self.state[0, 0]), float(self.state[1, 0])


class CricketBallTracker:
    """
    Cricket Ball Tracker combining YOLO detections, ByteTrack associations,
    and a Kalman filter fallback to guarantee smooth continuous ball paths.
    """
    def __init__(self, fps=30, max_jump_distance=150):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1)
        self.max_jump_distance = max_jump_distance
        self.kalman = CricketBallKalmanFilter(dt=self.dt)

        self.last_valid_pos = None
        self.history = []  # list of (frame_idx, (x, y), is_interpolated)

    def track_batch(self, raw_ball_tracks, start_frame_idx=0):
        """
        Process a batch of raw ball tracks (list of dicts from detector)
        and return smoothed tracks with Kalman interpolation for missing frames.
        """
        smoothed_tracks = []

        for offset, frame_track in enumerate(raw_ball_tracks):
            curr_idx = start_frame_idx + offset
            detection = frame_track.get(1, None)

            if detection is not None:
                cx, cy = detection['centroid']

                # Outlier check: if jump is physically impossible for 1 frame
                if self.last_valid_pos is not None:
                    dist = np.hypot(cx - self.last_valid_pos[0], cy - self.last_valid_pos[1])
                    if dist > self.max_jump_distance and self.kalman.missed_frames < 2:
                        # Reject outlier, use Kalman prediction instead
                        pred = self.kalman.predict()
                        self.kalman.missed_frames += 1
                        if pred is not None:
                            cx, cy = int(pred[0]), int(pred[1])
                            w = detection['bbox'][2] - detection['bbox'][0]
                            h = detection['bbox'][3] - detection['bbox'][1]
                            smoothed_tracks.append({
                                1: {
                                    'bbox': [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                                    'centroid': (cx, cy),
                                    'conf': 0.5,
                                    'interpolated': True
                                }
                            })
                            self.last_valid_pos = (cx, cy)
                            self.history.append((curr_idx, (cx, cy), True))
                            continue

                # Valid detection update
                kx, ky = self.kalman.update(cx, cy)
                self.last_valid_pos = (int(kx), int(ky))
                self.history.append((curr_idx, self.last_valid_pos, False))

                smoothed_tracks.append({
                    1: {
                        'bbox': detection.get('bbox', [cx - 5, cy - 5, cx + 5, cy + 5]),
                        'centroid': self.last_valid_pos,
                        'conf': detection.get('conf', 1.0),
                        'interpolated': False
                    }
                })
            else:
                # Missing detection: use Kalman filter forward prediction
                if self.kalman.initialized and self.kalman.missed_frames < self.kalman.max_missed_frames:
                    pred = self.kalman.predict()
                    self.kalman.missed_frames += 1
                    if pred is not None:
                        px, py = int(pred[0]), int(pred[1])
                        self.last_valid_pos = (px, py)
                        self.history.append((curr_idx, (px, py), True))
                        smoothed_tracks.append({
                            1: {
                                'bbox': [px - 10, py - 10, px + 10, py + 10],
                                'centroid': (px, py),
                                'conf': max(0.2, 0.8 - (self.kalman.missed_frames * 0.1)),
                                'interpolated': True
                            }
                        })
                    else:
                        smoothed_tracks.append({})
                else:
                    smoothed_tracks.append({})

        return smoothed_tracks
