#!/usr/bin/env python3
"""Export a trained .pt checkpoint to OpenVINO INT8."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

ROOT = setup_path()


def main():
    p = argparse.ArgumentParser(description="Export OpenVINO INT8")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="Calibration dataset YAML used by Ultralytics export")
    p.add_argument("--imgsz", type=int, default=480)
    args = p.parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(format="openvino", int8=True, imgsz=args.imgsz, data=args.data, simplify=True)
    print(exported)


if __name__ == "__main__":
    main()
