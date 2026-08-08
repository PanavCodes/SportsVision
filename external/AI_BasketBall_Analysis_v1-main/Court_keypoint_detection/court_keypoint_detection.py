from ultralytics import YOLO
import sys
sys.path.append('../')
from utils import read_stub, save_stub
import numpy as np

class CourtKeypointDetector:
    def __init__(self, model_path, conf_threshold=0.6):
        """
        Initialize the CourtKeypointDetector with a YOLO model and confidence threshold.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.last_valid_keypoints = None  # Stores last valid keypoints for fallback
    
    def get_court_keypoints(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect court keypoints for a list of frames.
        """
        court_keypoints = read_stub(read_from_stub, stub_path)
        if court_keypoints is not None and len(court_keypoints) == len(frames):
            return court_keypoints
        
        batch_size = 20
        court_keypoints = []
        
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=self.conf_threshold)

            for frame_idx, detection in enumerate(detections_batch, start=i):
                kps = detection.keypoints
                
                # Handle cases where no keypoints are detected (e.g., close-up shots)
                if kps is None or len(kps.xy) == 0 or len(kps.xy[0]) == 0:
                    if self.last_valid_keypoints is not None:
                        court_keypoints.append(self.last_valid_keypoints)
                    else:
                        court_keypoints.append(np.zeros((14, 2))) # Fallback empty shape
                    continue
                
                if hasattr(kps, 'confidence') and kps.confidence is not None and len(kps.confidence) > 0:
                    filtered_kps = []
                    for idx, (point, conf) in enumerate(zip(kps.xy[0], kps.confidence[0])):
                        if conf >= self.conf_threshold:
                            filtered_kps.append(point.cpu().numpy())
                        else:
                            if self.last_valid_keypoints is not None and idx < len(self.last_valid_keypoints):
                                filtered_kps.append(self.last_valid_keypoints[idx])
                            else:
                                filtered_kps.append(point.cpu().numpy())
                    filtered_kps = np.array(filtered_kps)

                    if np.all(filtered_kps == 0) and self.last_valid_keypoints is not None:
                        filtered_kps = self.last_valid_keypoints

                    court_keypoints.append(filtered_kps)
                    self.last_valid_keypoints = filtered_kps

                else:
                    kp_array = kps.xy[0].cpu().numpy()
                    court_keypoints.append(kp_array)
                    self.last_valid_keypoints = kp_array

        save_stub(stub_path, court_keypoints)
        return court_keypoints
