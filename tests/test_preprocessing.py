import numpy as np

from videoplay.preprocessing import letterbox


def test_letterbox_blob_shape():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    blob, scale, left, top, width, height = letterbox(frame, size=480)
    assert blob.shape == (1, 3, 480, 480)
    assert width == 320 and height == 240
    assert scale > 0
