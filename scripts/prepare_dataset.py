#!/usr/bin/env python3
"""Build a source-disjoint train/val/test image split.

By default this script only prints the planned assignment (--dry-run).
Pass --apply to create hard links or copies and to delete train images
whose SHA-256 also appears in val or test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

ROOT = setup_path()
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{6})")
YMD_RE = re.compile(r"^(\d{8}_\d+)")
SEED = 0


def source_id(name: str) -> str:
    m = DATE_RE.match(name)
    if m:
        return m.group(1)
    m = YMD_RE.match(name)
    if m:
        return m.group(1)
    return Path(name).stem


def collect(img_root: Path):
    groups = defaultdict(list)
    for split in ("train", "val", "test"):
        folder = img_root / split
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            groups[source_id(p.name)].append(p)
    return groups


def greedy_split(groups, train_frac=0.8, val_frac=0.1):
    items = [(sid, paths) for sid, paths in groups.items()]
    items.sort(key=lambda x: x[0])
    rng = random.Random(SEED)
    rng.shuffle(items)
    items.sort(key=lambda x: -len(x[1]))
    total = sum(len(p) for _, p in items)
    targets = {"train": train_frac * total, "val": val_frac * total, "test": (1 - train_frac - val_frac) * total}
    assigned = {k: [] for k in targets}
    counts = {k: 0 for k in targets}
    for sid, paths in items:
        best = min(targets, key=lambda k: counts[k] / (targets[k] + 1e-9))
        assigned[best].append((sid, paths))
        counts[best] += len(paths)
    return assigned, counts, total


def label_of(im: Path) -> Path:
    return Path(str(im).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Prepare source-disjoint dataset")
    p.add_argument("--source-images", required=True, help="Directory that contains train/val/test image folders")
    p.add_argument("--out", default=str(ROOT / "data" / "processed" / "noleak"))
    p.add_argument("--apply", action="store_true", help="Write files. Without this flag, only print the plan.")
    return p.parse_args()


def main():
    args = parse_args()
    img_root = Path(args.source_images)
    out = Path(args.out)
    groups = collect(img_root)
    assigned, counts, total = greedy_split(groups)
    plan = {
        "seed": SEED,
        "total_images": total,
        "total_sources": len(groups),
        "planned_counts": {k: int(v) for k, v in counts.items()},
        "apply": args.apply,
        "out": str(out),
        "note": "Train images whose SHA-256 appears in val or test are removed only when --apply is set.",
    }
    if not args.apply:
        print(json.dumps(plan, indent=2))
        print("dry-run: pass --apply to materialize the split")
        return
    out.mkdir(parents=True, exist_ok=True)
    overlap_check = {}
    split_files = {}
    used_names = {"train": set(), "val": set(), "test": set()}
    for split, rows in assigned.items():
        img_dir = out / "images" / split
        lab_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        n_img = n_box = n_miss = 0
        srcs = []
        for sid, imgs in rows:
            srcs.append(sid)
            for im in sorted(imgs):
                name = im.name
                if name in used_names[split]:
                    name = f"{im.parent.name}_{im.name}"
                used_names[split].add(name)
                link_or_copy(im.resolve(), img_dir / name)
                lab = label_of(im.resolve())
                n_img += 1
                if not lab.exists():
                    n_miss += 1
                    continue
                link_or_copy(lab, lab_dir / Path(name).with_suffix(".txt"))
                n_box += len([ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()])
        split_files[split] = {"images": n_img, "sources": len(srcs), "boxes": n_box, "missing_labels": n_miss}
        overlap_check[split] = set(srcs)
    blocked = set()
    for split in ("val", "test"):
        img_dir = out / "images" / split
        if img_dir.exists():
            for p in img_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    blocked.add(file_sha256(p))
    removed = []
    train_dir = out / "images" / "train"
    train_lab = out / "labels" / "train"
    if train_dir.exists():
        for p in list(train_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            digest = file_sha256(p)
            if digest in blocked:
                lab = train_lab / (p.stem + ".txt")
                p.unlink(missing_ok=True)
                if lab.exists():
                    lab.unlink()
                removed.append({"file": p.name, "sha256": digest})
    yaml_path = out / "video_data_noleak.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: video",
                "",
            ]
        ),
        encoding="utf-8",
    )
    leak = {
        "train_val": sorted(overlap_check.get("train", set()) & overlap_check.get("val", set())),
        "train_test": sorted(overlap_check.get("train", set()) & overlap_check.get("test", set())),
        "val_test": sorted(overlap_check.get("val", set()) & overlap_check.get("test", set())),
    }
    report = {
        **plan,
        "split_counts": split_files,
        "removed_train_hash_dups": len(removed),
        "overlap": {k: len(v) for k, v in leak.items()},
        "yaml": str(yaml_path),
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(leak.values()):
        raise SystemExit("source leak remains")


if __name__ == "__main__":
    main()
