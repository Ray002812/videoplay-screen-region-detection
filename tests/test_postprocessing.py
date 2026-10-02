import numpy as np

from videoplay.postprocessing import decode_yolo


def test_decode_empty_when_all_low_score():
    outputs = [np.zeros((8, 10), dtype=np.float32)]
    results = decode_yolo(outputs, conf=0.99, iou=0.5, scale=1.0, offset_x=0, offset_y=0, width=100, height=100)
    assert results == []
