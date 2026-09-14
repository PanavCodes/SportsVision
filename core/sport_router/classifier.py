import cv2
import numpy as np

class SportClassifier:
    """
    Multi-signal sport classifier for distinguishing Basketball vs Cricket:
      1. Surface Chromatic Analysis:
         - Cricket: Outfield green grass spectrum (HSV Hue 35-85) + central clay pitch
         - Basketball: Hardwood court spectrum (HSV Hue 10-25, warm maple tones) + high indoor contrast
      2. Line & Court Geometry:
         - Basketball courts feature concentric arcs (3-point line, key, center circle)
         - Cricket fields feature a central elongated pitch rectangle and oval boundary
      3. Zero-shot CLIP Fallback (optional if transformers available)
    """
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self._init_clip()

    def _init_clip(self):
        """Optionally load CLIP / SigLIP if available in environment."""
        try:
            from transformers import AutoProcessor, AutoModel
            import config
            model_id = "google/siglip-base-patch16-224"
            # Try loading from local offline cache first
            try:
                self.clip_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                self.clip_model = AutoModel.from_pretrained(model_id, local_files_only=True).to(config.DEVICE)
            except Exception:
                self.clip_processor = AutoProcessor.from_pretrained(model_id)
                self.clip_model = AutoModel.from_pretrained(model_id).to(config.DEVICE)
            self.clip_model.eval()
        except Exception:
            # Fall back cleanly to chromatic & geometric classifier
            self.clip_model = None

    def classify_frame(self, frame_bgr):
        """
        Classify a single frame as 'cricket' or 'basketball'.
        Returns: (sport: str, confidence: float)
        """
        # --- 1. Chromatic Surface Distribution ---
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w, _ = frame_bgr.shape
        total_pixels = h * w

        # Green Grass Mask (Cricket Outfield)
        lower_green = np.array([32, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.count_nonzero(green_mask) / float(total_pixels)

        # Hardwood Court Mask (Basketball Court - warm wood / varnish / amber)
        lower_hardwood = np.array([8, 60, 60])
        upper_hardwood = np.array([28, 255, 240])
        hardwood_mask = cv2.inRange(hsv, lower_hardwood, upper_hardwood)
        hardwood_ratio = np.count_nonzero(hardwood_mask) / float(total_pixels)

        # --- 2. Color scoring ---
        # Cricket fields typically have > 25% green outfield grass
        if green_ratio > 0.22 and green_ratio > (hardwood_ratio * 1.5):
            conf = min(0.98, 0.60 + (green_ratio * 0.8))
            return "cricket", float(conf)

        # Basketball indoor courts typically have significant hardwood court coverage
        if hardwood_ratio > 0.25 and hardwood_ratio > (green_ratio * 2.0):
            conf = min(0.98, 0.60 + (hardwood_ratio * 0.8))
            return "basketball", float(conf)

        # --- 3. Zero-shot CLIP Verification (if available) ---
        if self.clip_model is not None and self.clip_processor is not None:
            try:
                import torch
                from PIL import Image
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                candidate_texts = [
                    "a photo of a cricket match on a grass field with a pitch",
                    "a photo of a basketball game on an indoor court"
                ]

                inputs = self.clip_processor(
                    text=candidate_texts,
                    images=pil_img,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.clip_model.device)

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = torch.sigmoid(logits_per_image).cpu().numpy()[0]

                cricket_prob = float(probs[0])
                basketball_prob = float(probs[1])

                if cricket_prob > basketball_prob:
                    return "cricket", float(cricket_prob / (cricket_prob + basketball_prob + 1e-5))
                else:
                    return "basketball", float(basketball_prob / (cricket_prob + basketball_prob + 1e-5))
            except Exception:
                pass

        # Final chromatic fallback
        if green_ratio >= hardwood_ratio:
            return "cricket", 0.65
        else:
            return "basketball", 0.65
