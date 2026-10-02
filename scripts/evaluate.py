#!/usr/bin/env python3
"""Evaluate a COCO prediction JSON against a ground-truth annotation JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

setup_path()
from videoplay.evaluation import coco_eval  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="COCO bbox evaluation")
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    metrics = coco_eval(Path(args.gt), Path(args.pred))
    print(json.dumps(metrics, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
