import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms

import config

# Gracefully handle missing Repos
_has_repo2 = False
_has_repo4 = False

try:
    if config.REPO2_DIR not in sys.path:
        sys.path.insert(0, config.REPO2_DIR)
    from basketball_detection import BasketballDetector
    _has_repo2 = True
except (ImportError, FileNotFoundError, OSError):
    pass

try:
    if config.REPO4_DIR not in sys.path:
        sys.path.insert(0, config.REPO4_DIR)
    from src.models.resnet import generate_resnet
    _has_repo4 = True
except (ImportError, FileNotFoundError, OSError):
    pass


class CourtVisionShotDetector:
    """
    Wraps Repo 2's geometric trajectory detector and (optionally) Repo 4's ResNet50 classifier.
    Gracefully degrades when either repo or their weights are unavailable.
    Optimized to ingest pre-computed ball tracks rather than running YOLO again.
    """
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.device = torch.device(config.DEVICE)
        self.shot_events = []
        self.frame_confidences = []

        print(f"Loading ShotEngine models on {self.device}...")
        
        # Geometric detector (Repo 2)
        self.geometric_detector = None
        if _has_repo2:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("repo2_utils", os.path.join(config.REPO2_DIR, "utils.py"))
                repo2_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(repo2_utils)
                self.geometric_detector = BasketballDetector(model=None, fps=self.fps, class_names=repo2_utils.CLASS_NAMES)
                print("Geometric shot detector loaded (Repo 2).")
            except Exception as e:
                print(f"WARNING: Could not load Repo 2 geometric detector: {e}")
        else:
            print("INFO: Repo 2 (Basketball-Shot-Detection) not found. Geometric shot detection disabled.")
        
        # ResNet50 classifier (Repo 4)
        self.resnet50 = None
        self.use_resnet = False
        
        if _has_repo4 and os.path.exists(config.RESNET_MODEL_PATH):
            try:
                self.resnet50 = generate_resnet(number=50, current_device=self.device)
                self.resnet50.load_state_dict(torch.load(config.RESNET_MODEL_PATH, map_location=self.device))
                self.resnet50.eval()
                self.use_resnet = True
                print("ResNet50 shot classifier loaded successfully.")
            except Exception as e:
                print(f"WARNING: Could not load ResNet50: {e}. Using geometric-only detection.")
        else:
            print("INFO: ResNet50 weights not found. Using geometric-only shot detection.")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
            
    def update_batch(self, frames: list, ball_tracks: list, player_tracks: list):
        """
        Runs geometric tracking and (optionally) CNN classification on a batch of frames.
        Returns list of confidence values per frame.
        """
        batch_resnet_confs = [0.0] * len(frames)
        
        # If no geometric detector is available, just return zeros
        if self.geometric_detector is None:
            self.frame_confidences.extend(batch_resnet_confs)
            return batch_resnet_confs
        
        crops = []
        crop_indices = []

        # 1. Process all geometric detections first
        for idx, (frame, ball_track, p_trk) in enumerate(zip(frames, ball_tracks, player_tracks)):
            has_shot = False
            
            if 1 in ball_track:
                bbox = ball_track[1]['bbox']
                x1, y1, x2, y2 = bbox
                x3, y3 = int((x1 + x2) / 2), int((y1 + y2) / 2)
                r = int((x2 - x1) / 2)
                self.geometric_detector.ball_positions.append((x3, y3, r))
                
                # Find closest person
                closest_dist = float('inf')
                closest_person = None
                for p_id, p_data in p_trk.items():
                    px1, py1, px2, py2 = p_data['bbox']
                    pcx, pcy = (px1 + px2)/2, (py1 + py2)/2
                    dist = (pcx - x3)**2 + (pcy - y3)**2
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_person = (px1, py1, px2, py2)
                        
                if closest_person:
                    self.geometric_detector.person_positions.append(closest_person)
                    has_shot = self.geometric_detector.detect_shooting(
                        person_position=closest_person, ball_position=(x3, y3, r)
                    )
                    self.geometric_detector.has_shot = has_shot

            # 2. Collect crops for CNN if applicable
            if self.use_resnet and self.resnet50 is not None:
                if has_shot and hasattr(self.geometric_detector, 'backboard_bbox') and len(self.geometric_detector.backboard_bbox) > 0:
                    try:
                        x, y, w, h = self.geometric_detector.backboard_bbox
                        crop = frame[y:y+h, x:x+w]
                        if crop.size > 0:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            tensor = self.transform(crop_rgb)
                            crops.append(tensor)
                            crop_indices.append(idx)
                    except Exception:
                        pass
            else:
                # Geometric-only fallback
                if has_shot:
                    batch_resnet_confs[idx] = 0.6
                    
        # 3. Run CNN Batch Inference
        if crops:
            with torch.no_grad():
                batch_tensor = torch.stack(crops).to(self.device)
                output = self.resnet50(batch_tensor)
                confs = torch.sigmoid(output).cpu().numpy()
                for i, conf in zip(crop_indices, confs):
                    batch_resnet_confs[i] = float(conf[0] if isinstance(conf, (list, np.ndarray)) else conf)

        self.frame_confidences.extend(batch_resnet_confs)
        return batch_resnet_confs

    def get_shot_events(self):
        return self.shot_events
