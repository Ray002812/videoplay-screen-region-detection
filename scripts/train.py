#!/usr/bin/env python3
"""Train VideoPlay and comparison models with the local Ultralytics 8.3.223 tree."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

ROOT = setup_path()
ULTRA = ROOT / "third_party" / "ultralytics"
DEFAULT_CFG = ROOT / "configs" / "training" / "noleak.yaml"


def load_jobs(cfg_path: Path):
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = {"jobs": [], "epochs": 300, "imgsz": 480, "batch": 16, "workers": 4, "amp": True, "project": "outputs/train"}
    return data


def resolve_model(model: str) -> str:
    p = Path(model)
    if p.suffix in {".yaml", ".yml"} and not p.is_absolute():
        cand = ROOT / p
        if cand.exists():
            return str(cand)
    return model


def train_one(name: str, model: str, epochs: int, batch: int, imgsz: int, workers: int, amp: bool, resume: bool, data: str, project: Path):
    os.chdir(ULTRA)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ULTRA) + os.pathsep + str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    sys.path.insert(0, str(ULTRA))
    from ultralytics import YOLO

    weights_dir = project / name / "weights"
    last_pt = weights_dir / "last.pt"
    use_resume = resume and last_pt.exists()
    yolo = YOLO(str(last_pt) if use_resume else resolve_model(model))
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
        resume=use_resume,
        pretrained=True,
        single_cls=True,
        patience=20,
        amp=amp,
        seed=0,
        deterministic=True,
        close_mosaic=10,
        plots=False,
        val=True,
    )
    best = weights_dir / "best.pt"
    meta = {"name": name, "model": model, "best": str(best) if best.exists() else None, "resumed": use_resume}
    (project / name / "job_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("DONE", json.dumps(meta))


def parse_args():
    p = argparse.ArgumentParser(description="Train VideoPlay suite")
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--data", required=True, help="Ultralytics dataset YAML with a machine-local path")
    p.add_argument("--only", default="videoplay")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--project", default=None)
    p.add_argument("--worker", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_jobs(Path(args.config))
    project = Path(args.project) if args.project else ROOT / cfg.get("project", "outputs/train")
    jobs = cfg.get("jobs", [])
    if args.only != "all":
        jobs = [j for j in jobs if j["name"] == args.only]
    jobs = [j for j in jobs if j["name"] not in set(args.skip or [])]
    if not jobs:
        raise SystemExit(f"no matching job: {args.only}")
    if args.worker:
        job = jobs[0]
        epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 300))
        train_one(
            job["name"],
            job["model"],
            epochs,
            args.batch,
            args.imgsz,
            args.workers,
            args.amp,
            args.resume,
            args.data,
            project,
        )
        return
    project.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for job in jobs:
        epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 300))
        log = project / f"{job['name']}.log"
        cmd = [
            py,
            str(Path(__file__).resolve()),
            "--worker",
            "--only",
            job["name"],
            "--data",
            args.data,
            "--config",
            args.config,
            "--epochs",
            str(epochs),
            "--batch",
            str(args.batch),
            "--imgsz",
            str(args.imgsz),
            "--workers",
            str(args.workers),
            "--project",
            str(project),
        ]
        if args.amp:
            cmd.append("--amp")
        if args.resume:
            cmd.append("--resume")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ULTRA) + os.pathsep + str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        print("LAUNCH", " ".join(cmd), flush=True)
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=str(ULTRA), stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"EXIT {job['name']} {r.returncode}", flush=True)
        if r.returncode != 0:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
