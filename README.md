# VideoPlay

VideoPlay localizes a playing video region in mixed desktop screen content. A compact YOLO11-derived detector proposes boxes; a deterministic frame-difference gate keeps only candidates that look like playback rather than page scroll or a moving cursor. The published runtime is OpenVINO INT8 on CPU.

Paper (under revision, not a published article): *VideoPlay: Lightweight Video-Region Localization and Temporal Playback Confirmation in Mixed Screen Content*, IET Image Processing, manuscript IPR-2026-03-0268. Authors: Wenxi Zhao, Jiazhen Zhu, Yufan Ye, Changcai Lai (Hangzhou Dianzi University). Corresponding author: cclai@hdu.edu.cn.

Chinese instructions: [README.zh-CN.md](README.zh-CN.md).

## Requirements

- Python 3.10 or newer (development used 3.12).
- Windows 11 or Linux.
- CPU is enough for the default OpenVINO path.
- NVIDIA GPU is required only if you train.

Do not `pip install ultralytics` from PyPI and expect the same graphs. This repository uses the vendored tree `third_party/ultralytics` (8.3.223) with `Conv.default_act = ReLU`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## CPU inference install

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements/inference.txt
pip install -e .
```

On Linux, replace the activate line with `source .venv/bin/activate`.

Add evaluation extras only when you compute COCO metrics:

```text
pip install -r requirements/evaluation.txt
```

## GPU training install

```text
pip install -r requirements/training.txt
```

Install a CUDA PyTorch build that matches your driver. Training in the paper used PyTorch with CUDA on an NVIDIA GeForce RTX 5070. Export to INT8 also needs NNCF (listed in `requirements/training.txt`).

## Published model

The only default weight in this repository is:

```text
weights/videoplay-v1.0/openvino-int8/best
```

which loads `best.xml` and `best.bin`. SHA-256 values are in [weights/manifest.json](weights/manifest.json).

This is not the older local export whose BIN was about 2.3 MiB. The original export `metadata.yaml` is kept for provenance; its historical training path is not used at inference time.

## Run a video

```text
python scripts/infer_video.py --video path/to/20240509_160438.mp4 --video-name 20240509_160438 --output outputs/scene_01_pred.json
```

`--video-name` must match the annotation `image_id` prefix in [configs/evaluation/scenarios.yaml](configs/evaluation/scenarios.yaml). Frame indices start at 1. Scene `scene_01` is Bilibili (`20240509_160438`); `scene_02` is Webpage (`20240509_151154`).

Motion thresholds default to T=25, T1=160, T2=3, T3=5, buffer=5. Pass `--no-motion` for detector boxes only.

Latency for a random 480 input:

```text
python scripts/benchmark.py
```

## Dataset and annotations

Full images and MP4 files are not in Git. [data/README.md](data/README.md) explains how to request them and how `scripts/prepare_dataset.py` builds a source-disjoint split (`--apply` actually writes and may delete hash-duplicate train files).

The four evaluation JSONs under `data/sample/annotations/` are the paper ground truth. PPT (`scene_04`) lists 1348 frames: 1304 positive and 44 negative. AP is computed only on those `image_id` values.

## Train, export, evaluate

```text
python scripts/prepare_dataset.py --source-images path/to/images --out path/to/noleak
python scripts/prepare_dataset.py --source-images path/to/images --out path/to/noleak --apply
python scripts/train.py --data path/to/noleak/video_data_noleak.yaml --only videoplay
python scripts/export_openvino.py --weights outputs/train/videoplay/weights/best.pt --data path/to/noleak/video_data_noleak.yaml
python scripts/evaluate.py --pred outputs/scene_01_pred.json --gt data/sample/annotations/scene_01.json
```

[configs/training/noleak.yaml](configs/training/noleak.yaml) is the default for a **new** run (`max_epochs: 300`). It does not match the archived paper tables. Those runs are listed in [configs/experiments/catalog.yaml](configs/experiments/catalog.yaml): VideoPlay was configured for 100 epochs and the log ends at epoch 60; the other suite jobs were configured for 50 epochs and completed 50. `legacy_run_id` is the original folder name (`abl_ch14`, `sota_yolov8n`, ...).

## Reproduce paper video metrics

Place the four MP4 files in one directory, named as in `configs/evaluation/scenarios.yaml`, then:

```text
python scripts/reproduce.py --video-root path/to/videos
```

Recorded numbers for the released INT8 model plus the temporal gate are in [results/paper/metrics/videoplay_motion_eval.json](results/paper/metrics/videoplay_motion_eval.json).

| id | Display name | AP50:95 | AP50 | FPS (recorded) |
|----|--------------|---------|------|----------------|
| scene_01 | Bilibili | 0.794 | 0.823 | 93.2 |
| scene_02 | Webpage | 0.947 | 0.950 | 100.8 |
| scene_03 | Excel | 0.887 | 0.891 | 115.3 |
| scene_04 | PPT | 0.889 | 0.981 | 34.9 |

FPS is timed over decoded frames and is not a guaranteed SLA. PPT AP uses 1348 annotated frames out of 1698 decoded frames.

Ablation and YOLO-nano detector tables are separate files and separate checkpoints; see [results/paper/README.md](results/paper/README.md).

## Tests

```text
pip install -e .[dev]
pytest
```

## License

AGPL-3.0-or-later ([LICENSE](LICENSE)), because the work includes a modified Ultralytics copy. Screen recordings and third-party UI captured in those recordings are not covered by this license.

## Known limits

- Four evaluation recordings; results do not establish general playback-state recognition.
- Temporal filtering can drop low-motion video (Excel AP falls relative to detector-only).
- Training images are not redistributed here.
- Archived tables were not produced under the 300-epoch retrain default.
- Root `CITATION.cff` is for this software and manuscript, not a journal DOI.
