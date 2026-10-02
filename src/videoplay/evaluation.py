"""COCO-style evaluation helpers for playing-region detections."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List


def normalize_coco_gt(data: dict) -> dict:
    if "info" not in data:
        data["info"] = {"description": "VideoPlay annotations", "version": "1.0"}
    if "licenses" not in data:
        data["licenses"] = []
    if "images" not in data and "annotations" in data:
        unique_ids = {ann["image_id"] for ann in data["annotations"] if "image_id" in ann}
        data["images"] = [
            {"id": img_id, "file_name": f"{img_id}.jpg", "height": 480, "width": 480}
            for img_id in sorted(unique_ids, key=str)
        ]
    return data


def load_gt(gt_path: Path) -> dict:
    data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    return normalize_coco_gt(data)


def xyxy_to_coco_bbox(x1: int, y1: int, x2: int, y2: int) -> List[float]:
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def write_coco_predictions(items: Iterable[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(items), ensure_ascii=False), encoding="utf-8")


def coco_eval(gt_path: Path, pred_path: Path) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = load_gt(gt_path)
    valid_ids = {img["id"] for img in gt.get("images", [])}
    preds = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    if isinstance(preds, dict) and "annotations" in preds:
        preds = preds["annotations"]
    preds = [p for p in preds if p.get("image_id") in valid_ids]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp_gt:
        json.dump(gt, tmp_gt)
        tmp_gt_path = tmp_gt.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp_dt:
        json.dump(preds, tmp_dt)
        tmp_dt_path = tmp_dt.name
    coco_gt = COCO(tmp_gt_path)
    coco_dt = coco_gt.loadRes(tmp_dt_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    precision = ev.eval["precision"]
    recall = ev.eval["recall"]
    p = float(precision[0, :, :, 0, 2].mean()) if precision.size else 0.0
    r = float(recall[0, :, 0, 2].mean()) if recall.size else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "P": p,
        "R": r,
        "F1": f1,
    }
