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


def abs_from_cwd(path_str: str, cwd: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def load_jobs(cfg_path: Path):
    if not cfg_path.is_file():
        raise SystemExit(f"training config not found: {cfg_path}")
    import yaml

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"invalid training config: {cfg_path}")
    return data


def resolve_model(model: str) -> str:
    p = Path(model)
    if p.suffix in {".yaml", ".yml"} and not p.is_absolute():
        cand = ROOT / p
        if cand.exists():
            return str(cand)
    return model


def train_one(name, model, epochs, batch, imgsz, workers, amp, resume, data, project, device, patience, seed, close_mosaic):
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
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
        resume=use_resume,
        pretrained=True,
        single_cls=True,
        patience=patience,
        amp=amp,
        seed=seed,
        deterministic=True,
        close_mosaic=close_mosaic,
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
    p.add_argument("--data", required=True, help="Ultralytics dataset YAML resolved from the launch working directory")
    p.add_argument("--only", default="videoplay")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--epochs", type=int, default=None, help="Override config max_epochs")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--project", default=None)
    p.add_argument("--worker", action="store_true")
    return p.parse_args()


def settings_from_cfg(cfg, args):
    epochs = args.epochs if args.epochs is not None else int(cfg.get("max_epochs", cfg.get("epochs", 300)))
    batch = args.batch if args.batch is not None else int(cfg.get("batch", 16))
    imgsz = args.imgsz if args.imgsz is not None else int(cfg.get("imgsz", 480))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 4))
    device = args.device if args.device is not None else cfg.get("device", 0)
    if args.no_amp:
        amp = False
    elif args.amp:
        amp = True
    else:
        amp = bool(cfg.get("amp", True))
    patience = int(cfg.get("patience", 20))
    seed = int(cfg.get("seed", 0))
    close_mosaic = int(cfg.get("close_mosaic", 10))
    return {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "device": device,
        "amp": amp,
        "patience": patience,
        "seed": seed,
        "close_mosaic": close_mosaic,
    }


def main():
    args = parse_args()
    launch_cwd = Path.cwd()
    cfg_path = abs_from_cwd(args.config, launch_cwd)
    data_path = abs_from_cwd(args.data, launch_cwd)
    if not data_path.is_file():
        raise SystemExit(f"dataset yaml not found: {data_path}")
    cfg = load_jobs(cfg_path)
    if args.project:
        project = abs_from_cwd(args.project, launch_cwd)
    else:
        project = abs_from_cwd(str(cfg.get("project", "outputs/train")), launch_cwd)
    settings = settings_from_cfg(cfg, args)
    jobs = cfg.get("jobs", [])
    if args.only != "all":
        jobs = [j for j in jobs if j["name"] == args.only]
    jobs = [j for j in jobs if j["name"] not in set(args.skip or [])]
    if not jobs:
        raise SystemExit(f"no matching job: {args.only}")
    if args.worker:
        job = jobs[0]
        train_one(
            job["name"],
            job["model"],
            settings["epochs"],
            settings["batch"],
            settings["imgsz"],
            settings["workers"],
            settings["amp"],
            args.resume,
            str(data_path),
            project,
            settings["device"],
            settings["patience"],
            settings["seed"],
            settings["close_mosaic"],
        )
        return
    project.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for job in jobs:
        log = project / f"{job['name']}.log"
        cmd = [
            py,
            str(Path(__file__).resolve()),
            "--worker",
            "--only",
            job["name"],
            "--data",
            str(data_path),
            "--config",
            str(cfg_path),
            "--epochs",
            str(settings["epochs"]),
            "--batch",
            str(settings["batch"]),
            "--imgsz",
            str(settings["imgsz"]),
            "--workers",
            str(settings["workers"]),
            "--device",
            str(settings["device"]),
            "--project",
            str(project),
        ]
        if settings["amp"]:
            cmd.append("--amp")
        else:
            cmd.append("--no-amp")
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
