from videoplay.baselines.classical_motion import propose_boxes
import numpy as np


def test_no_previous_frame_returns_empty():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    assert propose_boxes(frame, None) == []
