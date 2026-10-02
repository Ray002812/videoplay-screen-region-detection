"""OpenVINO INT8 inference for VideoPlay-YOLO11."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .motion import MotionParams, filter_candidate
from .postprocessing import decode_yolo
from .preprocessing import letterbox


def default_model_xml(repo_root: Path) -> Path:
    return repo_root / "weights" / "videoplay-v1.0" / "openvino-int8" / "best.xml"


def load_compiled_model(model_xml: Path, device: str = "CPU"):
    from openvino import Core

    core = Core()
    model = core.read_model(str(model_xml))
    return core.compile_model(model, device)


def infer_frame(compiled, frame, conf: float, iou: float, size: int = 480):
    blob, scale, offset_x, offset_y, width, height = letterbox(frame, size=size)
    outputs = compiled(blob)
    return decode_yolo(outputs[compiled.output(0)], conf, iou, scale, offset_x, offset_y, width, height)


def apply_motion(
    frame,
    prev_frame,
    frame_buffer: Sequence,
    detections: List[Tuple],
    params: Optional[MotionParams] = None,
) -> List[Tuple]:
    if params is None:
        params = MotionParams()
    kept = []
    for item in detections:
        x1, y1, x2, y2 = item[:4]
        if filter_candidate(frame, prev_frame, frame_buffer, (x1, y1, x2, y2), params):
            kept.append(item)
    return kept
