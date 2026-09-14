import cv2
import numpy as np
import easyocr
import config

class CourtVisionPlayerID:
    """
    Uses EasyOCR to read jersey numbers from player crops.
    """
    def __init__(self):
        print(f"Loading EasyOCR model for Player ID (GPU={'cuda' in config.DEVICE})...")
        self.reader = easyocr.Reader(['en'], gpu=('cuda' in config.DEVICE))
        
        # Track history to stabilize OCR readings over time (majority voting)
        self.track_history = {}

    def identify_players(self, video_frames: list, player_tracks: list):
        """
        Reads jersey numbers for each player track.
        """
        for frame_idx, frame in enumerate(video_frames):
            tracks = player_tracks[frame_idx]
            
            for player_id, data in tracks.items():
                if data.get('team') == 0: # Skip referees
                    continue
                    
                bbox = data['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Crop to upper body (roughly where jersey number is)
                crop_h = y2 - y1
                crop_w = x2 - x1
                
                cy1 = int(y1 + crop_h * 0.15)
                cy2 = int(y1 + crop_h * 0.6)
                cx1 = int(x1 + crop_w * 0.1)
                cx2 = int(x2 - crop_w * 0.1)
                
                if cy2 <= cy1 or cx2 <= cx1:
                    continue
                    
                crop = frame[max(0, cy1):min(frame.shape[0], cy2), max(0, cx1):min(frame.shape[1], cx2)].copy()
                
                if 'mask' in data:
                    m = data['mask'][max(0, cy1):min(frame.shape[0], cy2), max(0, cx1):min(frame.shape[1], cx2)]
                    if m.shape == crop.shape[:2]:
                        crop[m == 0] = 0
                
                # Enhance crop for OCR
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Unsharp mask for contrast
                blur = cv2.GaussianBlur(gray, (0,0), 3)
                gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
                
                # Run OCR
                results = self.reader.readtext(gray, allowlist='0123456789')
                
                if player_id not in self.track_history:
                    self.track_history[player_id] = {}
                    
                for (bbox_ocr, text, prob) in results:
                    if prob > 0.3:
                        if text not in self.track_history[player_id]:
                            self.track_history[player_id][text] = 0
                        self.track_history[player_id][text] += prob
                
                # Determine best jersey number
                best_number = None
                best_score = 0
                for text, score in self.track_history[player_id].items():
                    if score > best_score and score > 1.0: # threshold to accept
                        best_score = score
                        best_number = text
                        
                data['jersey_number'] = best_number
                
        return player_tracks
