"""Classical motion baseline without a learned detector.

The method thresholds frame differences, extracts connected blobs, and
emits the largest plausible box as a playing-region candidate.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..motion import MotionParams, compute_motion, shrink_roi


def propose_boxes(frame, prev_frame, params: Optional[MotionParams] = None) -> List[Tuple[int, int, int, int, float, int]]:
    if params is None:
        params = MotionParams()
    if frame is None or prev_frame is None:
        return []
    h, w = frame.shape[:2]
    box = (0, 0, w, h)
    nx1, ny1, nx2, ny2 = shrink_roi(*box, params.roi_shrink)
    roi = frame[ny1:ny2, nx1:nx2]
    prev_roi = prev_frame[ny1:ny2, nx1:nx2]
    stats = compute_motion(prev_roi, roi, params)
    if stats.scroll_like or stats.blob_count == 0:
        return []
    gray_a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    diff = cv2.absdiff(gray_a, gray_b)
    _, binary = cv2.threshold(diff, params.T, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if bw * bh < 16:
        return []
    return [(x, y, x + bw, y + bh, 1.0, 0)]
