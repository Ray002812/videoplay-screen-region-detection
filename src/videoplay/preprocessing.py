"""Letterbox resize used by OpenVINO inference (square 480, pad value 114)."""
from __future__ import annotations

import cv2


def letterbox(frame, size=480):
    height, width = frame.shape[:2]
    length = max(height, width)
    if height > width:
        top, bottom, left, right = 0, 0, 0, height - width
    else:
        top, bottom, left, right = 0, width - height, 0, 0
    image = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    scale = length / float(size)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(size, size), swapRB=True)
    return blob, scale, left, top, width, height
