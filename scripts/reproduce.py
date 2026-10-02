#!/usr/bin/env python3
"""Run inference and COCO evaluation for the four paper scenes.

Videos are not shipped with the repository. Supply --video-root that contains
the files listed in configs/evaluation/scenarios.yaml.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

ROOT = setup_path()


def load_scenarios():
    text = (ROOT / "configs" / "evaluation" / "scenarios.yaml").read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception as exc:
        raise SystemExit("PyYAML is required to read scenarios.yaml") from exc


def main():
    p = argparse.ArgumentParser(description="Reproduce paper video evaluation")
    p.add_argument("--video-root", required=True)
    p.add_argument("--model", default=str(ROOT / "weights" / "videoplay-v1.0" / "openvino-int8" / "best"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "eval"))
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()
    cfg = load_scenarios()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    infer = str(ROOT / "scripts" / "infer_video.py")
    evaluate = str(ROOT / "scripts" / "evaluate.py")
    summary = []
    for scene in cfg["scenarios"]:
        video = Path(args.video_root) / scene["video"]
        if not video.exists():
            raise FileNotFoundError(f"missing video for {scene['id']}: {video}")
        pred = out_dir / f"{scene['id']}_pred.json"
        cmd = [
            py,
            infer,
            "--model",
            args.model,
            "--video",
            str(video),
            "--video-name",
            scene["image_id_prefix"],
            "--output",
            str(pred),
        ]
        if args.max_frames:
            cmd.extend(["--max-frames", str(args.max_frames)])
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
        metrics_path = out_dir / f"{scene['id']}_metrics.json"
        gt = ROOT / scene["annotation"]
        subprocess.check_call([py, evaluate, "--pred", str(pred), "--gt", str(gt), "--output", str(metrics_path)])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["id"] = scene["id"]
        summary.append(metrics)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
