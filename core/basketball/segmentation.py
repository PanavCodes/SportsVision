import os
import torch
import numpy as np
from ultralytics import FastSAM, SAM
import config

class PlayerSegmenter:
    """
    High-performance player segmentation engine.
    Supports FastSAM for real-time edge processing (recommended for consumer GPUs like RTX 3050)
    and SAM2.1 for high-fidelity zero-shot mask generation.
    """
    def __init__(self, model_type: str = None):
        if model_type is None:
            model_type = getattr(config, "SEGMENTATION_MODEL", "fastsam").lower()
            
        self.model_type = model_type
        print(f"Loading {self.model_type.upper()} segmentation engine on {config.DEVICE}...")
        
        if self.model_type == "fastsam":
            fastsam_path = getattr(config, "FASTSAM_MODEL_PATH", os.path.join(config.SHARED_MODELS_DIR, "FastSAM-s.pt"))
            if not os.path.exists(fastsam_path):
                alt_fastsam = os.path.join(config.WORKSPACE_ROOT, "FastSAM-s.pt")
                if os.path.exists(alt_fastsam):
                    fastsam_path = alt_fastsam
                else:
                    fastsam_path = "FastSAM-s.pt"
            self.model = FastSAM(fastsam_path)
        else:
            sam_path = getattr(config, "SAM_MODEL_PATH", os.path.join(config.SHARED_MODELS_DIR, "sam2.1_s.pt"))
            self.model = SAM(sam_path)
            
        self.model.to(config.DEVICE)
        
    def segment_players(self, frame, bboxes):
        """
        Generate masks for a list of bounding boxes using promptable segmentation.
        
        Args:
            frame: The BGR numpy array frame (H, W, 3).
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            
        Returns:
            masks: A list of numpy binary masks (shape [H, W]), one for each bbox.
                   If no bboxes, returns empty list.
        """
        if not bboxes or len(bboxes) == 0:
            return []
            
        h, w = frame.shape[:2]
        
        try:
            if self.model_type == "fastsam":
                results = self.model.predict(
                    source=frame,
                    bboxes=bboxes,
                    verbose=False,
                    device=config.DEVICE,
                    retina_masks=True
                )
            else:
                results = self.model.predict(
                    source=frame,
                    bboxes=bboxes,
                    verbose=False,
                    device=config.DEVICE,
                    quantize="fp16" if getattr(config, "USE_FP16", False) else None
                )
                
            masks = []
            if results and results[0].masks is not None:
                masks_tensor = results[0].masks.data
                masks_np = masks_tensor.cpu().numpy()
                masks = [masks_np[i] for i in range(masks_np.shape[0])]
                
            # Guarantee 1-to-1 matching with bboxes
            if len(masks) == len(bboxes):
                return masks
                
            # Fallback if detection count does not match bbox count exactly
            aligned_masks = []
            for i, bbox in enumerate(bboxes):
                if i < len(masks) and masks[i].shape == (h, w):
                    aligned_masks.append(masks[i])
                else:
                    # Bounding box rectangular fallback mask
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    fallback_mask = np.zeros((h, w), dtype=np.uint8)
                    fallback_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
                    aligned_masks.append(fallback_mask)
            return aligned_masks
            
        except Exception as e:
            # Fallback to rectangular masks on any unexpected failure
            fallback_masks = []
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                fm = np.zeros((h, w), dtype=np.uint8)
                fm[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
                fallback_masks.append(fm)
            return fallback_masks
