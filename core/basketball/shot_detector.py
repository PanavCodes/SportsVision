import cv2
import numpy as np
from collections import deque
import config

class CourtVisionShotDetector:
    """
    Pure geometric parabolic trajectory mapper for Make/Miss classification.
    Fits a 2nd degree polynomial to the ball's (x, y) coordinates over time.
    """
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.shot_events = []
        self.frame_confidences = []
        
        # Dynamic frame counter for non-uniform interval tracking
        self.current_frame_idx = 0
        
        # History of ball coordinates: list of (frame_idx, x, y)
        self.ball_history = deque(maxlen=45)
        
        # Default nominal hoop regions will be scaled dynamically to frame resolution
        self.hoop_regions = []
        
        self.is_shot_active = False
        self.shot_start_frame = -1
        self.current_shot_result = None
        
        print("Initialized Parabolic Shot Detector.")

    def update_batch(self, frames: list, ball_tracks: list, player_tracks: list):
        """
        Runs parabolic tracking on a batch of frames.
        Returns list of confidence values (1.0 for Make, 0.5 for Miss, 0.0 for None).
        """
        batch_confs = [0.0] * len(frames)
        
        for idx, (frame, ball_track, p_trk) in enumerate(zip(frames, ball_tracks, player_tracks)):
            self.current_frame_idx += 1
            conf = 0.0
            h_img, w_img = frame.shape[:2]
            
            # Dynamically compute hoop bounding boxes for current frame resolution
            hoop_regions = [
                [int(0.02 * w_img), int(0.22 * h_img), int(0.16 * w_img), int(0.48 * h_img)],  # Left hoop
                [int(0.84 * w_img), int(0.22 * h_img), int(0.98 * w_img), int(0.48 * h_img)]   # Right hoop
            ]
            hoop_bottom_y = int(0.48 * h_img)
            
            if 1 in ball_track:
                bbox = ball_track[1]['bbox']
                x_c = (bbox[0] + bbox[2]) / 2.0
                y_c = (bbox[1] + bbox[3]) / 2.0
                self.ball_history.append((self.current_frame_idx, x_c, y_c))
                
            # Need at least 15 points to fit a reliable parabola
            if len(self.ball_history) > 15:
                pts = list(self.ball_history)
                # Use actual frame deltas for t to correctly handle missed/dropped detection frames
                t = np.array([p[0] - pts[0][0] for p in pts], dtype=np.float64)
                x = np.array([p[1] for p in pts], dtype=np.float64)
                y = np.array([p[2] for p in pts], dtype=np.float64)
                
                # Fit y(t) = a*t^2 + b*t + c
                # In image coordinates, y increases downwards.
                # A shot goes UP (y decreases) then DOWN (y increases), so it's a U-shape.
                # A U-shape parabola has a > 0.
                coeffs_y, res_y, _, _, _ = np.polyfit(t, y, 2, full=True)
                a, b, c = coeffs_y
                
                # Calculate R^2 for goodness of fit
                p_y = np.poly1d(coeffs_y)
                y_mean = np.mean(y)
                ss_tot = np.sum((y - y_mean)**2)
                ss_res = np.sum((y - p_y(t))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # A valid shot has a > 0 (opens upwards in image coords) and high R^2
                if a > 0.1 and r_squared > 0.8:
                    self.is_shot_active = True
                    
                    # Find apex (t = -b / 2a)
                    apex_t = -b / (2 * a)
                    
                    # If we have passed the apex and are descending
                    if apex_t < t[-1]:
                        # Check where the latest point is
                        _, last_x, last_y = pts[-1]
                        
                        # See if it intersects a hoop
                        is_make = False
                        for h in hoop_regions:
                            hx1, hy1, hx2, hy2 = h
                            if hx1 <= last_x <= hx2 and hy1 <= last_y <= hy2:
                                is_make = True
                                break
                                
                        if is_make:
                            self.current_shot_result = "MAKE"
                            conf = 1.0
                        else:
                            # It's falling, but not in a hoop. Could be a miss or a pass.
                            # We'll conservatively call it a MISS if it drops below the hoop level.
                            if last_y > hoop_bottom_y: 
                                self.current_shot_result = "MISS"
                                conf = 0.5
                            else:
                                conf = 0.6  # Shot in progress, descending
                    else:
                        conf = 0.6 # Shot in progress, ascending
                else:
                    # Not a shot arc
                    if self.is_shot_active:
                        # Shot ended
                        if self.current_shot_result:
                            self.shot_events.append(self.current_shot_result)
                        self.is_shot_active = False
                        self.current_shot_result = None
                        self.ball_history.clear() # Reset after a shot
            
            # Draw Shot Result on Frame
            if self.is_shot_active:
                cv2.putText(frame, "SHOT IN PROGRESS", (frame.shape[1]//2 - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            if self.current_shot_result == "MAKE":
                cv2.putText(frame, "MAKE!", (frame.shape[1]//2 - 100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
            elif self.current_shot_result == "MISS":
                cv2.putText(frame, "MISS!", (frame.shape[1]//2 - 100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                
            batch_confs[idx] = conf
            
        self.frame_confidences.extend(batch_confs)
        return batch_confs

    def get_shot_events(self):
        return self.shot_events
