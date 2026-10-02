#!/usr/bin/env python3
"""Run VideoPlay OpenVINO INT8 inference on a video and write COCO-style detections."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

ROOT = setup_path()

from videoplay.motion import MotionParams, filter_decision  # noqa: E402
from videoplay.postprocessing import decode_yolo  # noqa: E402
from videoplay.preprocessing import letterbox  # noqa: E402

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core

DEFAULT_MODEL = ROOT / "weights" / "videoplay-v1.0" / "openvino-int8" / "best"


def run_video(args):
    params = MotionParams(
        T=args.T,
        T1=args.T1,
        T2=args.T2,
        T3=args.T3,
        buffer_size=args.buffer_size,
    )
    core = Core()
    if args.threads > 0:
        try:
            core.set_property("CPU", {"INFERENCE_NUM_THREADS": args.threads})
        except Exception:
            pass

    xml_path = args.model if args.model.endswith(".xml") else args.model + ".xml"
    compiled = core.compile_model(xml_path, device_name="CPU")
    output_node = compiled.outputs[0]
    infer = compiled.create_infer_request()

    dummy = np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8)
    for _ in range(5):
        blob, *_ = letterbox(dummy, args.imgsz)
        infer.infer(blob)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    annotations = []
    buffer = []
    prev = None
    frame_count = 0
    t_pre = t_inf = t_post = t_mot = 0.0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        t = time.perf_counter()
        blob, scale, ox, oy, width, height = letterbox(frame, args.imgsz)
        t_pre += time.perf_counter() - t

        t = time.perf_counter()
        outputs = infer.infer(blob)[output_node]
        t_inf += time.perf_counter() - t

        t = time.perf_counter()
        dets = decode_yolo(outputs, args.conf, args.iou, scale, ox, oy, width, height)
        t_post += time.perf_counter() - t

        t = time.perf_counter()
        for x1, y1, x2, y2, score, cls_id in dets:
            ok = True
            if not args.no_motion:
                ok = filter_decision(frame, prev, buffer, (x1, y1, x2, y2), params) == "confirm"
            if ok:
                annotations.append(
                    {
                        "image_id": f"{args.video_name}-{frame_count}",
                        "category_id": cls_id + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
        t_mot += time.perf_counter() - t

        original = frame.copy()
        buffer.append(original)
        if len(buffer) > params.buffer_size:
            buffer.pop(0)
        prev = original

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
        if frame_count % 200 == 0:
            print(f"processed {frame_count} frames")

    cap.release()
    wall = time.perf_counter() - t0
    mem_after = proc.memory_info().rss
    timing = {
        "frames": frame_count,
        "preprocess_ms": (t_pre / frame_count) * 1000 if frame_count else 0,
        "inference_ms": (t_inf / frame_count) * 1000 if frame_count else 0,
        "postprocess_ms": (t_post / frame_count) * 1000 if frame_count else 0,
        "motion_ms": (t_mot / frame_count) * 1000 if frame_count else 0,
        "e2e_ms": (wall / frame_count) * 1000 if frame_count else 0,
        "fps": frame_count / wall if wall else 0,
        "rss_before_mb": mem_before / (1024 * 1024),
        "rss_after_mb": mem_after / (1024 * 1024),
        "rss_delta_mb": (mem_after - mem_before) / (1024 * 1024),
        "T": params.T,
        "T1": params.T1,
        "T2": params.T2,
        "T3": params.T3,
        "buffer_size": params.buffer_size,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "motion_enabled": not args.no_motion,
        "model": xml_path,
    }

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(annotations), encoding="utf-8")
    timing_path = out_json.with_suffix(".timing.json")
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2))
    print(f"saved {len(annotations)} dets to {out_json}")
    return timing


def parse_args():
    p = argparse.ArgumentParser(description="VideoPlay OpenVINO inference")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--video", required=True)
    p.add_argument("--video-name", required=True, help="Prefix used in image_id, matching the annotation file")
    p.add_argument("--output", required=True)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--T", type=int, default=25)
    p.add_argument("--T1", type=float, default=160.0)
    p.add_argument("--T2", type=int, default=3)
    p.add_argument("--T3", type=int, default=5)
    p.add_argument("--buffer-size", type=int, default=5)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-motion", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_video(parse_args())
