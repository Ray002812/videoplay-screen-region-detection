"""Temporal motion gate used after detector proposals.

Thresholds:
  T  : binarization threshold on |I_t - I_{t-1}|
  T1 : mean of the binary motion mask; values >= T1 exclude the ROI
       (page scroll or large UI motion)
  T2 : valid motion blobs in one frame; values >= T2 confirm playback
  T3 : accumulated blob count over the frame buffer; values >= T3
       confirm slow or low-frame-rate playback
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class MotionParams:
    T: int = 25
    T1: float = 160.0
    T2: int = 3
    T3: int = 5
    buffer_size: int = 5
    min_contour_area: float = 4.0
    cursor_area_ratio: float = 0.004
    scroll_cover_ratio: float = 0.85
    roi_shrink: float = 0.9
    erode_ksize: int = 3
    dilate_ksize: int = 20


@dataclass
class MotionStats:
    blob_count: int = 0
    avg_pixel: float = 0.0
    cover_ratio: float = 0.0
    max_blob_ratio: float = 0.0
    cursor_like: bool = False
    scroll_like: bool = False


def shrink_roi(x1: int, y1: int, x2: int, y2: int, ratio: float) -> Tuple[int, int, int, int]:
    w, h = x2 - x1, y2 - y1
    nw, nh = int(w * ratio), int(h * ratio)
    nx1 = x1 + (w - nw) // 2
    ny1 = y1 + (h - nh) // 2
    return nx1, ny1, nx1 + nw, ny1 + nh


def compute_motion(front_mat: np.ndarray, after_mat: np.ndarray, params: MotionParams) -> MotionStats:
    stats = MotionStats()
    if front_mat is None or after_mat is None:
        return stats
    if front_mat.size == 0 or after_mat.size == 0:
        return stats
    if front_mat.shape[:2] != after_mat.shape[:2]:
        after_mat = cv2.resize(after_mat, (front_mat.shape[1], front_mat.shape[0]))

    front_gray = cv2.cvtColor(front_mat, cv2.COLOR_BGR2GRAY) if front_mat.ndim == 3 else front_mat
    after_gray = cv2.cvtColor(after_mat, cv2.COLOR_BGR2GRAY) if after_mat.ndim == 3 else after_mat
    diff_gray = cv2.absdiff(front_gray, after_gray)
    _, binary = cv2.threshold(diff_gray, params.T, 255, cv2.THRESH_BINARY)

    erode_el = cv2.getStructuringElement(cv2.MORPH_RECT, (params.erode_ksize, params.erode_ksize))
    dilate_el = cv2.getStructuringElement(cv2.MORPH_RECT, (params.dilate_ksize, params.dilate_ksize))
    binary = cv2.erode(binary, erode_el)
    binary = cv2.dilate(binary, dilate_el)

    roi_area = float(binary.shape[0] * binary.shape[1])
    if roi_area <= 0:
        return stats

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = params.min_contour_area
    valid_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_area]

    stats.blob_count = len(valid_areas)
    stats.avg_pixel = float(cv2.mean(binary)[0])
    stats.cover_ratio = float(np.count_nonzero(binary)) / roi_area
    stats.max_blob_ratio = (max(valid_areas) / roi_area) if valid_areas else 0.0
    stats.scroll_like = stats.avg_pixel >= params.T1
    if stats.cover_ratio >= params.scroll_cover_ratio and stats.avg_pixel >= (params.T1 * 0.6):
        stats.scroll_like = True
    stats.cursor_like = (
        stats.blob_count == 1
        and stats.max_blob_ratio < params.cursor_area_ratio
        and stats.cover_ratio < 0.02
        and not stats.scroll_like
    )
    return stats


def decide_single_frame(stats: MotionStats, params: MotionParams) -> str:
    """Return exclude, confirm, or uncertain."""
    if stats.scroll_like:
        return "exclude"
    if stats.cursor_like:
        return "uncertain"
    if stats.blob_count >= params.T2:
        return "confirm"
    return "uncertain"


def decide_multi_frame(
    frame_list: Sequence[np.ndarray],
    current_roi_bgr: np.ndarray,
    params: MotionParams,
) -> str:
    if current_roi_bgr is None or current_roi_bgr.size == 0 or not frame_list:
        return "uncertain"

    total_blobs = 0
    valid = 0
    for past in frame_list:
        if past is None or past.size == 0:
            continue
        if past.shape[:2] != current_roi_bgr.shape[:2]:
            past = cv2.resize(past, (current_roi_bgr.shape[1], current_roi_bgr.shape[0]))
        stats = compute_motion(past, current_roi_bgr, params)
        total_blobs += stats.blob_count
        valid += 1

    if valid == 0:
        return "uncertain"
    if total_blobs >= params.T3:
        return "confirm"

    oldest = frame_list[0]
    if oldest is not None and oldest.size > 0:
        long_stats = compute_motion(oldest, current_roi_bgr, params)
        if (not long_stats.scroll_like) and (not long_stats.cursor_like) and long_stats.blob_count >= params.T2:
            return "confirm"
    return "uncertain"


def filter_decision(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray],
    frame_buffer: Sequence[np.ndarray],
    box: Tuple[int, int, int, int],
    params: MotionParams,
) -> str:
    x1, y1, x2, y2 = box
    nx1, ny1, nx2, ny2 = shrink_roi(x1, y1, x2, y2, params.roi_shrink)
    if nx2 <= nx1 or ny2 <= ny1:
        return "exclude"

    current_roi = frame[ny1:ny2, nx1:nx2]
    if current_roi.size == 0:
        return "exclude"

    if prev_frame is not None:
        prev_roi = prev_frame[ny1:ny2, nx1:nx2]
        stats = compute_motion(prev_roi, current_roi, params)
        decision = decide_single_frame(stats, params)
        if decision != "uncertain":
            return decision

    hist_rois: List[np.ndarray] = []
    for past in frame_buffer:
        hist_rois.append(past[ny1:ny2, nx1:nx2])
    return decide_multi_frame(hist_rois, current_roi, params)


def filter_candidate(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray],
    frame_buffer: Sequence[np.ndarray],
    box: Tuple[int, int, int, int],
    params: MotionParams,
) -> bool:
    """Return True if the box is classified as a playing video region."""
    return filter_decision(frame, prev_frame, frame_buffer, box, params) == "confirm"
