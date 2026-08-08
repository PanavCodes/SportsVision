"""
draw_utils.py

Drawing utilities for visualizing tracked objects (players and balls) on video frames.
Includes anti-aliased ellipse drawing for player positions and triangle markers for ball detections.
"""

import cv2
import sys
import numpy as np

sys.path.append("../")
from utils import get_center_bbox, get_width_bbox


def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draw an anti-aliased ellipse under the player and optionally show their track ID.
    """
    # Ellipse position and size
    y2 = int(bbox[3])  # Player's feet
    x_center, _ = get_center_bbox(bbox)
    width = get_width_bbox(bbox)

    # Draw ellipse
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA
    )

    # Draw tracking ID
    if track_id is not None:
        rect_w, rect_h = 40, 20
        x1 = x_center - rect_w // 2
        y1 = y2 - rect_h // 2 + 15
        x2 = x_center + rect_w // 2
        y2_rect = y2 + rect_h // 2 + 15

        cv2.rectangle(frame, (x1, y1), (x2, y2_rect), color, thickness=cv2.FILLED)

        # Adjust text position for 2- or 3-digit IDs
        x_text = x1 + 12
        if track_id > 99:
            x_text -= 10

        cv2.putText(
            frame,
            str(track_id),
            (x_text, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return frame


def draw_triangle(frame, bbox, color):
    """
    Draw a filled triangle marker above the object (e.g., ball).
    """
    y = int(bbox[1])  # Top of the bbox
    x, _ = get_center_bbox(bbox)

    triangle = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20]
    ])

    cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle], 0, (0,0,0), 2)

    return frame
