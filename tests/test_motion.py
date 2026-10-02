from videoplay.motion import MotionParams, compute_motion, decide_single_frame, decide_multi_frame
import cv2
import numpy as np


def test_identical_frames_uncertain():
    params = MotionParams()
    a = np.full((200, 320, 3), 40, dtype=np.uint8)
    stats = compute_motion(a, a.copy(), params)
    assert decide_single_frame(stats, params) == "uncertain"


def test_multi_blob_confirms_playback():
    params = MotionParams()
    a = np.full((200, 320, 3), 40, dtype=np.uint8)
    video = a.copy()
    for i in range(8):
        x0 = 20 + i * 30
        y0 = 40 + (i % 3) * 40
        cv2.rectangle(video, (x0, y0), (x0 + 24, y0 + 24), (220, 220, 220), -1)
    stats = compute_motion(a, video, params)
    assert decide_single_frame(stats, params) == "confirm"


def test_large_shift_is_excluded():
    params = MotionParams()
    a = np.full((200, 320, 3), 40, dtype=np.uint8)
    scroll = np.roll(a, 40, axis=0)
    scroll[:, :] = np.clip(scroll.astype(np.int16) + 80, 0, 255).astype(np.uint8)
    stats = compute_motion(a, scroll, params)
    assert decide_single_frame(stats, params) == "exclude"


def test_small_cursor_is_not_confirm():
    params = MotionParams()
    a = np.full((200, 320, 3), 40, dtype=np.uint8)
    cursor = a.copy()
    cv2.circle(cursor, (160, 100), 3, (255, 255, 255), -1)
    stats = compute_motion(a, cursor, params)
    assert decide_single_frame(stats, params) != "confirm"


def test_buffer_confirms_low_rate_motion():
    params = MotionParams()
    a = np.full((200, 320, 3), 40, dtype=np.uint8)
    lowfps = a.copy()
    cv2.rectangle(lowfps, (40, 40), (120, 100), (200, 200, 200), -1)
    hist = [a.copy() for _ in range(8)] + [lowfps]
    assert decide_multi_frame(hist, lowfps, params) == "confirm"
