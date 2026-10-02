#!/usr/bin/env python3
"""Print OpenVINO model I/O and run a short latency probe on random input."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

ROOT = setup_path()

from videoplay.preprocessing import letterbox  # noqa: E402

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core

DEFAULT_XML = ROOT / "weights" / "videoplay-v1.0" / "openvino-int8" / "best.xml"


def port_info(port):
    try:
        name = port.get_any_name()
    except Exception:
        name = "unnamed"
    return {"name": name, "shape": str(port.shape), "type": str(port.element_type)}


def main():
    p = argparse.ArgumentParser(description="Profile OpenVINO model")
    p.add_argument("--model", default=str(DEFAULT_XML))
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=480)
    args = p.parse_args()
    xml = args.model if args.model.endswith(".xml") else args.model + ".xml"
    core = Core()
    compiled = core.compile_model(xml, "CPU")
    infer = compiled.create_infer_request()
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    blob, *_ = letterbox(dummy, args.imgsz)
    for _ in range(5):
        infer.infer(blob)
    t0 = time.perf_counter()
    for _ in range(args.runs):
        infer.infer(blob)
    dt = (time.perf_counter() - t0) / args.runs
    info = {
        "model": xml,
        "inputs": [port_info(i) for i in compiled.inputs],
        "outputs": [port_info(o) for o in compiled.outputs],
        "letterbox_ms": None,
        "infer_ms": dt * 1000,
        "runs": args.runs,
    }
    t1 = time.perf_counter()
    for _ in range(args.runs):
        letterbox(dummy, args.imgsz)
    info["letterbox_ms"] = (time.perf_counter() - t1) / args.runs * 1000
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
