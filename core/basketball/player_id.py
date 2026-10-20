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
        # Fast-path cache: once a jersey number is locked with high confidence, freeze it
        self.locked_jerseys = {}
        self.track_frame_count = {}
        self.ocr_attempt_count = {}
        self.max_ocr_attempts = 10  # Max OCR evaluations per track if no number detected
        self.check_stride = 5       # Sample OCR every 5 frames per track

    def _get_best_number(self, player_id: int):
        if player_id not in self.track_history:
            return None
        best_num = None
        best_score = 0
        for text, score in self.track_history[player_id].items():
            if score > best_score and score >= 0.8:
                best_score = score
                best_num = text
        return best_num

    def identify_players(self, video_frames: list, player_tracks: list):
        """
        Reads jersey numbers with track-level caching to prevent redundant per-frame OCR inference.
        """
        for frame_idx, frame in enumerate(video_frames):
            tracks = player_tracks[frame_idx]
            
            for player_id, data in tracks.items():
                if data.get('team') == 0 or data.get('is_referee', False) or data.get('is_sideline', False):  # Skip referees & sideline
                    continue
                    
                # 1. Fast Path: If jersey number is already locked, reuse immediately without OCR
                if player_id in self.locked_jerseys:
                    data['jersey_number'] = self.locked_jerseys[player_id]
                    continue
                    
                # 2. Cap attempts on unidentifiable tracks (e.g. facing away or occluded)
                if self.ocr_attempt_count.get(player_id, 0) >= self.max_ocr_attempts:
                    data['jersey_number'] = self._get_best_number(player_id)
                    continue

                # 3. Strided sampling: Only evaluate OCR once every N frames per track
                self.track_frame_count[player_id] = self.track_frame_count.get(player_id, 0) + 1
                if self.track_frame_count[player_id] % self.check_stride != 1:
                    data['jersey_number'] = self._get_best_number(player_id)
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
                blur = cv2.GaussianBlur(gray, (0, 0), 3)
                gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
                
                # Run OCR
                self.ocr_attempt_count[player_id] = self.ocr_attempt_count.get(player_id, 0) + 1
                results = self.reader.readtext(gray, allowlist='0123456789')
                
                if player_id not in self.track_history:
                    self.track_history[player_id] = {}
                    
                for (bbox_ocr, text, prob) in results:
                    clean_text = str(text).strip()
                    # Valid basketball jersey numbers are 1 or 2 digits (0-99)
                    if prob > 0.35 and 1 <= len(clean_text) <= 2 and clean_text.isdigit():
                        self.track_history[player_id][clean_text] = self.track_history[player_id].get(clean_text, 0) + prob
                
                best_number = self._get_best_number(player_id)
                
                # Lock condition: High cumulative confidence (>= 1.5) locks jersey number permanently
                if best_number and self.track_history[player_id].get(best_number, 0) >= 1.5:
                    self.locked_jerseys[player_id] = best_number
                    
                data['jersey_number'] = best_number
                
        return player_tracks
