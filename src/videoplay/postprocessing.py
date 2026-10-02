"""Decode YOLO11-style OpenVINO output to image-space boxes."""
from __future__ import annotations

import cv2
import numpy as np


def decode_yolo(outputs, conf, iou, scale, offset_x, offset_y, width, height):
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    boxes, scores, class_ids = [], [], []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, max_score, _, (_, max_class_index) = cv2.minMaxLoc(classes_scores)
        if max_score >= conf:
            boxes.append(
                [
                    outputs[0][i][0] - 0.5 * outputs[0][i][2],
                    outputs[0][i][1] - 0.5 * outputs[0][i][3],
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
            )
            scores.append(max_score)
            class_ids.append(max_class_index)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    results = []
    for index in keep:
        box = boxes[index]
        x1 = int(box[0] * scale) - offset_x
        y1 = int(box[1] * scale) - offset_y
        x2 = int((box[0] + box[2]) * scale) - offset_x
        y2 = int((box[1] + box[3]) * scale) - offset_y
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            results.append((x1, y1, x2, y2, float(scores[index]), int(class_ids[index])))
    return results
