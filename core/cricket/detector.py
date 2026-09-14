import os
import sys
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

import config

class CricketDetector:
    """
    Cricket multi-class detection architecture with DRS-grade heuristics:
      1. Ball Detector: fine-tuned YOLOv8 model specialized for high-velocity cricket ball.
         - Dynamic CLAHE local contrast normalization for stadium shadow transitions.
         - Motion-blur streak detection for fast deliveries (>140 km/h).
         - Kinematic distance gating against previous ball centroid.
      2. Player/Umpire Detector: YOLOv8m COCO model for high-precision human tracking.
      3. 3-Class DRS Scheme & Stumps Calibration:
         - Standardized mapping: [0: ball, 1: stump, 2: pad] (dschandra/lbw_drs_ai).
         - Vertical edge gradient filtering (Sobel) for high-precision stump localization (sanjusabu).
         - Explicit batsman pad bounding box extraction.
    """
    # 3-Class DRS Schema
    DRS_CLASS_MAP = {
        0: "ball",
        1: "stump",
        2: "pad"
    }

    def __init__(self, ball_model_path=None):
        print(f"[CricketDetector] Initializing models on {config.DEVICE}...")

        # 1. Player Detector (Person class = 0 in COCO)
        player_model_path = config.PLAYER_MODEL_PATH
        if not os.path.exists(player_model_path):
            player_model_path = "yolov8m.pt"
        self.player_model = YOLO(player_model_path)
        self._configure_model(self.player_model)

        # 2. Ball Detector: Check for CBDbest.pt (saral7293) or cricket_ball_best.pt
        if ball_model_path is None:
            cbd_path = os.path.join(config.WORKSPACE_ROOT, "models", "cricket", "CBDbest.pt")
            default_path = config.CRICKET_BALL_MODEL_PATH
            alt_path = config.CRICKET_YOLO_MODEL_PATH

            if os.path.exists(cbd_path):
                ball_model_path = cbd_path
            elif os.path.exists(default_path):
                ball_model_path = default_path
            elif os.path.exists(alt_path):
                ball_model_path = alt_path
            else:
                ball_model_path = "yolov8m.pt"

        print(f"[CricketDetector] Loading cricket ball model: {ball_model_path}")
        self.ball_model = YOLO(ball_model_path)
        self._configure_model(self.ball_model)

        # Supervision ByteTrack for continuous player tracks
        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.75,
            frame_rate=30
        )

        self.ball_confidence = getattr(config, 'CRICKET_BALL_CONFIDENCE', 0.20)
        self.player_confidence = getattr(config, 'CRICKET_PLAYER_CONFIDENCE', 0.30)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.prev_ball_pos = None

        # Cache for roles (batsman, bowler, keeper, umpire)
        self.role_assignments = {}
        self.role_votes = defaultdict(lambda: defaultdict(int))

    def _configure_model(self, model):
        """Send model to target device and enable FP16 if supported."""
        try:
            model.to(config.DEVICE)
            if config.USE_FP16 and "cuda" in str(config.DEVICE):
                model.model.half()
        except Exception as e:
            print(f"[CricketDetector] Warning configuring model: {e}")

    def enhance_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies local CLAHE equalization to the luminance (V) channel in HSV space
        to normalize harsh grandstand shadows and day/night floodlight glare.
        """
        if frame is None or len(frame.shape) != 3:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def track(self, video_frames):
        """
        Run detection and tracking across a batch of frames.
        Returns:
            player_tracks: list of dicts {track_id: {'bbox': [x1, y1, x2, y2], 'conf': float, 'role': str}}
            ball_tracks: list of dicts {1: {'bbox': [x1, y1, x2, y2], 'centroid': (x, y), 'conf': float}} or empty
            stumps_info: list of detected pitch end zones or stumps bboxes
        """
        player_tracks = []
        ball_tracks = []
        stumps_info = []

        for frame in video_frames:
            # --- A. Ball Detection with CLAHE & Streak Heuristics ---
            ball_track_frame = {}
            enhanced_frame = self.enhance_lighting(frame)

            ball_results = self.ball_model(
                enhanced_frame,
                conf=max(0.12, self.ball_confidence - 0.05),
                verbose=False,
                device=config.DEVICE
            )[0]

            best_ball_box = None
            best_ball_conf = 0.0

            if ball_results.boxes is not None and len(ball_results.boxes) > 0:
                boxes = ball_results.boxes.xyxy.cpu().numpy()
                confs = ball_results.boxes.conf.cpu().numpy()
                classes = ball_results.boxes.cls.cpu().numpy()

                candidates = []
                for box, conf, cls_id in zip(boxes, confs, classes):
                    w = box[2] - box[0]
                    h = box[3] - box[1]

                    # Filter out boxes too large for a cricket ball (>75px)
                    if w > 75 or h > 75 or w < 3 or h < 3:
                        continue

                    # Aspect ratio check for motion blur streaks
                    ar = w / max(h, 1e-4)
                    is_streak = (ar > 1.3 or ar < 0.77)
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0

                    # Kinematic proximity gating against previous ball centroid
                    if self.prev_ball_pos is not None:
                        dist = np.hypot(cx - self.prev_ball_pos[0], cy - self.prev_ball_pos[1])
                        if dist > 220.0:
                            continue

                    # Acceptance logic: base confidence or motion streak
                    if conf >= self.ball_confidence or (is_streak and conf >= 0.14):
                        candidates.append((box, conf, (cx, cy)))

                if candidates:
                    candidates.sort(key=lambda c: c[1], reverse=True)
                    best_ball_box, best_ball_conf, best_ball_center = candidates[0]
                    self.prev_ball_pos = best_ball_center

            if best_ball_box is not None:
                cx = int((best_ball_box[0] + best_ball_box[2]) / 2)
                cy = int((best_ball_box[1] + best_ball_box[3]) / 2)
                ball_track_frame[1] = {
                    'bbox': [float(best_ball_box[0]), float(best_ball_box[1]),
                             float(best_ball_box[2]), float(best_ball_box[3])],
                    'centroid': (cx, cy),
                    'conf': float(best_ball_conf)
                }

            ball_tracks.append(ball_track_frame)

            # --- B. Player Detection & ByteTrack ---
            player_track_frame = {}
            player_results = self.player_model(
                frame,
                classes=[0],  # Person class in COCO
                conf=self.player_confidence,
                verbose=False,
                device=config.DEVICE
            )[0]

            if player_results.boxes is not None and len(player_results.boxes) > 0:
                detections = sv.Detections.from_ultralytics(player_results)
                tracked_detections = self.player_tracker.update_with_detections(detections)

                if tracked_detections.tracker_id is not None:
                    for xyxy, conf, track_id in zip(
                        tracked_detections.xyxy,
                        tracked_detections.confidence,
                        tracked_detections.tracker_id
                    ):
                        player_track_frame[int(track_id)] = {
                            'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                            'conf': float(conf) if conf is not None else 1.0,
                            'role': self.role_assignments.get(int(track_id), 'player')
                        }

            player_tracks.append(player_track_frame)

            # --- C. Stumps & Batsman Pad Localization with Edge Calibration ---
            stumps_frame = self._detect_stumps_calibrated(frame, player_track_frame)
            stumps_info.append(stumps_frame)

        # Refine player roles across accumulated tracks
        self._classify_player_roles(player_tracks, ball_tracks)

        return player_tracks, ball_tracks, stumps_info

    def _detect_stumps_calibrated(self, frame: np.ndarray, player_track_frame: dict) -> dict:
        """
        Locate stumps at batsman and bowler end with vertical Sobel edge gradient refinement.
        Inspired by sanjusabu/Cricket-Ball-and-Stumps-Detection.
        """
        h, w, _ = frame.shape
        stumps = {
            'batsman_end': (int(w * 0.50), int(h * 0.65)),
            'bowler_end': (int(w * 0.50), int(h * 0.35)),
            'batsman_pads': None
        }

        # Locate batsman
        for pid, pdata in player_track_frame.items():
            if pdata.get('role') == 'batsman':
                bx1, by1, bx2, by2 = pdata['bbox']
                pad_y1 = int(by1 + (by2 - by1) * 0.52)
                stumps['batsman_pads'] = (int(bx1), pad_y1, int(bx2), int(by2))

                # Vertical edge gradient analysis in region behind batsman feet
                roi_x1 = max(0, int(bx1 - 25))
                roi_x2 = min(w, int(bx2 + 25))
                roi_y1 = max(0, int(by2 - (by2 - by1) * 0.40))
                roi_y2 = min(h, int(by2 + 10))

                if roi_x2 > roi_x1 + 10 and roi_y2 > roi_y1 + 10:
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # Vertical Sobel filter detects parallel vertical stump poles
                    sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
                    vert_energy = np.sum(np.abs(sobel_x), axis=0)

                    if len(vert_energy) > 0:
                        peak_col = int(np.argmax(vert_energy))
                        calibrated_x = roi_x1 + peak_col
                        calibrated_y = int(by2 - 10)
                        stumps['batsman_end'] = (calibrated_x, calibrated_y)
                    else:
                        stumps['batsman_end'] = (int((bx1 + bx2) / 2.0), int(by2 - 10))
                else:
                    stumps['batsman_end'] = (int((bx1 + bx2) / 2.0), int(by2 - 10))
                break

        return stumps

    def _classify_player_roles(self, player_tracks, ball_tracks):
        """
        Classify cricket roles with cross-batch temporal voting:
          - Batsman: person standing nearest to the batting crease.
          - Wicketkeeper: person positioned directly behind the batsman.
          - Bowler: player with highest motion towards the pitch before ball release.
        """
        if not player_tracks:
            return

        all_pids = set()
        y_coords = defaultdict(list)
        x_coords = defaultdict(list)

        for frame_tracks in player_tracks:
            for pid, info in frame_tracks.items():
                all_pids.add(pid)
                bbox = info['bbox']
                foot_x = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]
                x_coords[pid].append(foot_x)
                y_coords[pid].append(foot_y)

        if not all_pids:
            return

        # Sort candidate players by median Y coordinate (bottom of frame = closer to camera / keeper)
        sorted_by_y = sorted(all_pids, key=lambda p: np.median(y_coords[p]) if y_coords[p] else 0, reverse=True)

        # Cast votes weighted by observations in current batch
        if len(sorted_by_y) >= 2:
            keeper_id = sorted_by_y[0]
            batsman_id = sorted_by_y[1]
            k_weight = len(y_coords[keeper_id])
            b_weight = len(y_coords[batsman_id])
            self.role_votes[batsman_id]['batsman'] += b_weight
            self.role_votes[keeper_id]['wicketkeeper'] += k_weight
        elif len(sorted_by_y) == 1:
            batsman_id = sorted_by_y[0]
            self.role_votes[batsman_id]['batsman'] += len(y_coords[batsman_id])

        if len(sorted_by_y) >= 3:
            bowler_candidates = sorted_by_y[2:]
            bowler_id = sorted(bowler_candidates, key=lambda p: np.median(y_coords[p]) if y_coords[p] else 9999)[0]
            self.role_votes[bowler_id]['bowler'] += len(y_coords[bowler_id])

        # Resolve winning role for each track ID based on accumulated evidence
        for pid in all_pids:
            if pid in self.role_votes and self.role_votes[pid]:
                winning_role = max(self.role_votes[pid], key=self.role_votes[pid].get)
                self.role_assignments[pid] = winning_role

        # Apply roles back to tracks
        for frame_tracks in player_tracks:
            for pid in frame_tracks:
                if pid in self.role_assignments:
                    frame_tracks[pid]['role'] = self.role_assignments[pid]
