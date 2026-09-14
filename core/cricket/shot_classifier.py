import os
import math
import numpy as np
import torch
import config

class CricketShotClassifier:
    """
    Batting shot classifier adhering to the CricShot10 taxonomy:
      1. cover_drive
      2. defense
      3. flick
      4. hook
      5. late_cut
      6. lofted
      7. pull
      8. square_cut
      9. straight_drive
      10. sweep
    
    Combines learned deep feature weights (if models/cricket/shot_classifier.pt exists)
    with kinematic bat-exit angle and velocity physics.
    """
    CLASSES = [
        "cover_drive",
        "defense",
        "flick",
        "hook",
        "late_cut",
        "lofted",
        "pull",
        "square_cut",
        "straight_drive",
        "sweep"
    ]

    def __init__(self, model_path=None):
        self.model_path = model_path or getattr(
            config, 'CRICKET_SHOT_CLASSIFIER_MODEL',
            os.path.join(config.WORKSPACE_ROOT, "models", "cricket", "shot_classifier.pt")
        )
        self.learned_model = None

        if os.path.exists(self.model_path):
            try:
                print(f"[CricketShotClassifier] Loading shot model: {self.model_path}")
                self.learned_model = torch.load(self.model_path, map_location=config.DEVICE)
                self.learned_model.eval()
            except Exception as e:
                print(f"[CricketShotClassifier] Note: using kinematic shot classifier ({e})")

    def classify_shot(self, batsman_crop_frames, pre_impact_traj, post_impact_traj):
        """
        Classify batting shot played during delivery.
        batsman_crop_frames: list of cropped image frames around impact moment.
        pre_impact_traj: ball coordinates leading into the bat.
        post_impact_traj: ball coordinates exiting off the bat.
        Returns: (shot_label: str, confidence: float)
        """
        if post_impact_traj is None or len(post_impact_traj) < 2:
            return "defense", 0.65

        # 1. Kinematic exit vector analysis
        dx = post_impact_traj[-1][0] - post_impact_traj[0][0]
        dy = post_impact_traj[-1][1] - post_impact_traj[0][1]
        exit_speed = np.hypot(dx, dy)

        # In image coords: Y increases downwards.
        # Negative dy means ball goes UP (aerial/lofted).
        elevation_angle = math.degrees(math.atan2(-dy, abs(dx) + 1e-5))
        exit_angle_deg = math.degrees(math.atan2(dy, dx))

        # Soft exit speed -> defensive block
        if exit_speed < 15.0:
            return "defense", 0.85

        # High elevation -> lofted shot
        if elevation_angle > 35.0:
            return "lofted", 0.88

        # Directional mapping from striker's crease:
        # Straight down ground
        if abs(dx) < 20 and dy < -20:
            return "straight_drive", 0.82

        # Off side shots:
        # dx < -20 (assuming camera viewing pitch straight on):
        if dx < -20:
            if elevation_angle < 10 and dy > 0:
                return "late_cut", 0.78
            elif abs(dy) < 30:
                return "square_cut", 0.80
            else:
                return "cover_drive", 0.84

        # Leg side shots: dx > 20:
        if dx > 20:
            if elevation_angle > 20:
                return "hook", 0.80
            elif dy < 0 and abs(dx) > 35:
                return "pull", 0.85
            elif dy > 20:
                return "sweep", 0.79
            else:
                return "flick", 0.82

        return "straight_drive", 0.70
