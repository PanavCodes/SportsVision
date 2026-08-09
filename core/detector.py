import sys
import os
import torch
import warnings
from collections import defaultdict
import numpy as np

import config
from ultralytics import YOLO
import supervision as sv

try:
    from rfdetr import RFDETRMedium
except ImportError:
    pass



class CourtVisionDetector:
    """
    Dual-model detection architecture:
      - Player model (COCO yolov8m): Tracks players via native BoT-SORT with stable IDs.
      - Ball model (basketball_best.pt): Detects the basketball and referees.
    
    The COCO model is excellent at detecting people (~14/frame) but cannot see basketballs.
    The basketball_best.pt is fine-tuned to detect balls, players, and referees,
    but only finds ~2 players/frame. By combining them, we get the best of both worlds.
    """
    def __init__(self):
        # --- Player Model ---
        if getattr(config, 'USE_RF_DETR', False):
            print(f"Loading RF-DETR model on {config.DEVICE}...")
            self.player_model = RFDETRMedium()
            # We use Supervision's ByteTrack to track RF-DETR detections over time
            self.tracker = sv.ByteTrack()
        else:
            print(f"Loading player tracking model on {config.DEVICE}...")
            player_model_path = config.PLAYER_MODEL_PATH
            
            if getattr(config, 'USE_ONNX', False):
                trt_path = player_model_path.replace('.pt', '.engine')
                if os.path.exists(trt_path):
                    player_model_path = trt_path
                    print(f"Using TensorRT Engine: {player_model_path}")
                
            self.player_model = YOLO(player_model_path)
            if player_model_path.endswith('.pt'):
                self._configure_model(self.player_model)
        
        # --- Ball Model (basketball_best.pt) ---
        print(f"Loading ball detection model on {config.DEVICE}...")
        self.ball_model = YOLO(config.BALL_MODEL_PATH)
        self._configure_model(self.ball_model)
        # Ball model classes: 0=ball, 1=player, 2=referee
        
        # Tracker config path
        self.tracker_yaml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "basketball_tracker.yaml"
        )
        if not os.path.exists(self.tracker_yaml):
            self.tracker_yaml = "botsort.yaml"  # fallback to default
        
        # State for ball temporal smoothing
        self.ball_ema = None
        self.ema_alpha = 0.5
        self.frames_since_ball = 0
        self.max_ball_coast = 30
        
        # Referee tracking: vote-based (must be matched as referee in N+ frames)
        self._referee_votes = {}  # {player_id: count}
        self._referee_vote_threshold = 3  # Need 3+ matches to be confirmed
        self.referee_ids = set()

    def _configure_model(self, model):
        model.to(config.DEVICE)
        if config.USE_FP16 and "cuda" in config.DEVICE:
            try:
                model.model.half()
            except Exception as e:
                warnings.warn(f"Could not convert model to FP16: {e}")

    def track(self, frames: list):
        """
        Run dual-model tracking on a list of frames.
        
        Returns:
            player_tracks: List of dicts per frame. {player_id: {'bbox': [x1,y1,x2,y2]}}
            ball_tracks: List of dicts per frame. {1: {'bbox': [x1,y1,x2,y2]}} or {}
        """
        if "cuda" in config.DEVICE:
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            import tqdm
            
            player_tracks = []
            raw_ball_tracks = []
            
            for frame in tqdm.tqdm(frames, desc="YOLO Tracking", leave=False):
                frame_h, frame_w = frame.shape[:2]
                
                # ===== PLAYER MODEL =====
                p_frame_tracks = {}
                if getattr(config, 'USE_RF_DETR', False):
                    # RF-DETR native inference
                    detections = self.player_model.predict(frame, threshold=config.PLAYER_CONFIDENCE)
                    # Filter to only persons (class 0)
                    detections = detections[detections.class_id == 0]
                    # Update tracker
                    tracked_detections = self.tracker.update_with_detections(detections)
                    
                    if tracked_detections is not None and len(tracked_detections) > 0:
                        for i in range(len(tracked_detections)):
                            box = tracked_detections.xyxy[i]
                            t_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else None
                            if t_id is not None:
                                # Crowd filter: skip detections in top 25% of frame
                                if box[3] < frame_h * 0.25:
                                    continue
                                p_frame_tracks[int(t_id)] = {'bbox': box.tolist()}
                else:
                    # Uses native BoT-SORT tracking with persist=True for stable IDs
                    player_res = self.player_model.track(
                        frame,
                        verbose=False,
                        classes=[0],  # COCO class 0 = person
                        conf=config.PLAYER_CONFIDENCE,
                        iou=0.45,
                        imgsz=640,  # Native resolution for COCO model
                        persist=True,
                        tracker=self.tracker_yaml
                    )[0]
                    
                    if player_res.boxes is not None and player_res.boxes.id is not None:
                        cls = player_res.boxes.cls.cpu().numpy()
                        ids = player_res.boxes.id.cpu().numpy()
                        xyxy = player_res.boxes.xyxy.cpu().numpy()
                        conf = player_res.boxes.conf.cpu().numpy()
                        
                        for c, i, box, cf in zip(cls, ids, xyxy, conf):
                            x1, y1, x2, y2 = box
                            
                            # Crowd filter: skip detections in top 25% of frame
                            if y2 < frame_h * 0.25:
                                continue
                            
                            p_frame_tracks[int(i)] = {'bbox': box.tolist()}
                
                player_tracks.append(p_frame_tracks)
                
                # ===== BALL MODEL (basketball_best.pt) =====
                # Uses predict (not track) since we only need the single best ball detection
                ball_res = self.ball_model.predict(
                    frame,
                    verbose=False,
                    classes=[0, 2],  # 0=ball, 2=referee
                    conf=config.BALL_CONFIDENCE,
                    iou=0.45,
                    imgsz=1280,  # High resolution for detecting small ball
                )[0]
                
                b_frame_tracks = {}
                
                if ball_res.boxes is not None and len(ball_res.boxes) > 0:
                    b_cls = ball_res.boxes.cls.cpu().numpy()
                    b_xyxy = ball_res.boxes.xyxy.cpu().numpy()
                    b_conf = ball_res.boxes.conf.cpu().numpy()
                    
                    best_ball_conf = 0.0
                    best_ball_box = None
                    
                    for bc, bbox, bcf in zip(b_cls, b_xyxy, b_conf):
                        if int(bc) == 0:  # Ball
                            if bcf > best_ball_conf:
                                best_ball_conf = bcf
                                best_ball_box = bbox.tolist()
                        elif int(bc) == 2 and bcf >= 0.35:  # Referee (high conf only)
                            # Find closest player ID in this frame and vote as referee
                            ref_cx = (bbox[0] + bbox[2]) / 2
                            ref_cy = (bbox[1] + bbox[3]) / 2
                            best_dist = float('inf')
                            best_pid = None
                            for pid, pdata in p_frame_tracks.items():
                                pb = pdata['bbox']
                                pcx = (pb[0] + pb[2]) / 2
                                pcy = (pb[1] + pb[3]) / 2
                                dist = (pcx - ref_cx)**2 + (pcy - ref_cy)**2
                                if dist < best_dist:
                                    best_dist = dist
                                    best_pid = pid
                            # Tight spatial match: within 1.5% of frame width to avoid misclassifying players fighting for the ball
                            if best_pid is not None and best_dist < (frame_w * 0.015)**2:
                                self._referee_votes[best_pid] = self._referee_votes.get(best_pid, 0) + 1
                                if self._referee_votes[best_pid] >= self._referee_vote_threshold:
                                    self.referee_ids.add(best_pid)
                    
                    if best_ball_box is not None:
                        b_frame_tracks[1] = {'bbox': best_ball_box, 'conf': float(best_ball_conf)}
                
                raw_ball_tracks.append(b_frame_tracks)
            
        # Ball Smoothing via sports library
        final_ball_tracks = []
        try:
            from sports.common.ball import BallTracker
            if not hasattr(self, 'ball_tracker'):
                self.ball_tracker = BallTracker(buffer_size=10)
        except ImportError:
            self.ball_tracker = None
            
        for frame_tracks in raw_ball_tracks:
            cleaned_tracks = {}
            
            if self.ball_tracker is not None:
                # Use sports.common.ball.BallTracker
                boxes = []
                confs = []
                if 1 in frame_tracks:
                    boxes.append(frame_tracks[1]['bbox'])
                    confs.append(frame_tracks[1]['conf'])
                
                if boxes:
                    detections = sv.Detections(
                        xyxy=np.array(boxes),
                        confidence=np.array(confs)
                    )
                else:
                    detections = sv.Detections.empty()
                    
                tracked_detections = self.ball_tracker.update(detections)
                if len(tracked_detections) > 0:
                    cleaned_tracks[1] = {
                        'bbox': tracked_detections.xyxy[0].tolist(),
                        'conf': float(tracked_detections.confidence[0]) if tracked_detections.confidence is not None else 1.0
                    }
            else:
                # Fallback to EMA if sports is not installed
                if 1 in frame_tracks:
                    curr_bbox = np.array(frame_tracks[1]['bbox'])
                    if self.ball_ema is None:
                        self.ball_ema = curr_bbox
                    else:
                        self.ball_ema = self.ema_alpha * curr_bbox + (1 - self.ema_alpha) * self.ball_ema
                        
                    cleaned_tracks[1] = {'bbox': self.ball_ema.tolist(), 'conf': frame_tracks[1]['conf']}
                    self.frames_since_ball = 0
                else:
                    self.frames_since_ball += 1
                    if self.frames_since_ball > self.max_ball_coast:
                        self.ball_ema = None
                    elif self.ball_ema is not None:
                        cleaned_tracks[1] = {'bbox': self.ball_ema.tolist(), 'conf': 0.5}
                        
            final_ball_tracks.append(cleaned_tracks)
        
        # Free VRAM
        if "cuda" in config.DEVICE:
            torch.cuda.empty_cache()
            
        return player_tracks, final_ball_tracks
