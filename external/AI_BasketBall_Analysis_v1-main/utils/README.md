# 🛠️ Utils

This folder contains helper modules used across the project for tasks such as video processing, bounding box manipulation, and caching of intermediate results. These utilities are designed to be reusable and modular to simplify the main codebase.

---

## 📁 Modules Overview

### `__init__.py`
Marks the folder as a Python package. You can also use it to re-export commonly used functions if needed.

---

### `bbox_utils.py`

Provides utilities for working with bounding boxes.

**Functions include:**
- `xywh_to_xyxy(box)` – Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2)
- `xyxy_to_center(box)` – Get center (cx, cy) of a box
- `get_box_area(box)` – Compute area of a bounding box
- `iou(boxA, boxB)` – Compute Intersection-over-Union between two boxes

---

### `stubs_utils.py`

Handles caching of intermediate results using `.pkl` files (called **stubs**).

**Functions include:**
- `read_stub(enabled, path)` – Load cached data if `enabled` is `True`
- `save_stub(path, data)` – Save data to a `.pkl` file

Useful for:
- Avoiding recomputation of object tracks or team assignments
- Experiment reproducibility

---

### `video_utils.py`

Provides functions to read, write, and resize video frames.

**Functions include:**
- `read_video(path)` – Load a video as a list of frames (images)
- `write_video(path, frames, fps)` – Save a list of frames to a video file
- `resize_frames(frames, size)` – Resize each frame to the given resolution

---

## ✅ Example Usage

```python
from utils.bbox_utils import xywh_to_xyxy
from utils.stubs_utils import read_stub, save_stub
from utils.video_utils import read_video, write_video
