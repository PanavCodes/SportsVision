import os
import torch
import numpy as np
from ultralytics import SAM
import config

class PlayerSegmenter:
    """
    Uses SAM2 (Segment Anything Model 2) to generate pixel-perfect masks of players
    based on bounding box prompts from our detection model.
    """
    def __init__(self):
        print(f"Loading SAM2 segmentation model on {config.DEVICE}...")
        # SAM2 small is highly accurate but fits in 8GB VRAM easily
        self.model = SAM(config.SAM_MODEL_PATH)
        # Ensure it runs on the correct device (GPU) and half precision if enabled
        self.model.to(config.DEVICE)
        
    def segment_players(self, frame, bboxes):
        """
        Generate masks for a list of bounding boxes.
        
        Args:
            frame: The BGR numpy array frame.
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            
        Returns:
            masks: A list of numpy binary masks (shape [H, W]), one for each bbox.
                   If no bboxes, returns empty list.
        """
        if not bboxes or len(bboxes) == 0:
            return []
            
        if "cuda" in config.DEVICE:
            torch.cuda.empty_cache()
            
        # SAM predict can take bboxes as prompts
        results = self.model.predict(
            source=frame,
            bboxes=bboxes,
            verbose=False,
            device=config.DEVICE,
            half=config.USE_FP16
        )
        
        masks = []
        result = results[0]
        if result.masks is not None:
            # .data is (N, H, W) tensor on GPU
            masks_tensor = result.masks.data
            # Convert to CPU numpy array
            masks = masks_tensor.cpu().numpy()
            # If multiple masks, we convert it to a list of (H, W) arrays
            masks = [masks[i] for i in range(masks.shape[0])]
            
        return masks
