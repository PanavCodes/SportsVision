"""
bbox_utils.py

Utility functions for working with bounding boxes.

"""

from typing import Tuple


def get_center_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """
    Compute the center (x, y) of a bounding box.
    """
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return x_center, y_center


def get_width_bbox(bbox: Tuple[float, float, float, float]) -> float:
    """
    Compute the width of a bounding box.
    """
    x1, _, x2, _ = bbox
    return x2 - x1


def measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two (x, y) points.
    """
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def get_foot_position(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

