# CourtKeypointDetector

A Python class for detecting court keypoints in video frames using Ultralytics YOLO.

## Features

- Batch processing for efficient inference on multiple frames.
- Confidence-based filtering of keypoints with fallback to previous valid keypoints.
- Supports caching of keypoints to/from stub files to avoid redundant computation.

## Usage

```python
from court_keypoint_detector import CourtKeypointDetector

# Initialize detector with YOLO model path and confidence threshold
detector = CourtKeypointDetector('path/to/yolo-model.pt', conf_threshold=0.6)

# Load frames (list of images) here
frames = [...]

# Get keypoints, optionally reading from cache stub
keypoints = detector.get_court_keypoints(frames, read_from_stub=False, stub_path='keypoints_stub.pkl')

