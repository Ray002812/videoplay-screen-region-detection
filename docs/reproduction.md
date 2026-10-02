# Reproduction

1. Install inference and evaluation requirements.
2. Confirm SHA-256 of `weights/videoplay-v1.0/openvino-int8` against `weights/manifest.json`.
3. Obtain the four evaluation videos (not in Git). File names are in `configs/evaluation/scenarios.yaml`.
4. Run `python scripts/reproduce.py --video-root <dir>`.
5. Compare `outputs/eval/summary.json` with `results/paper/metrics/videoplay_motion_eval.json`. Absolute AP should be close; FPS will vary.

AP for PPT is computed only on the 1348 `image_id` values in `data/sample/annotations/scene_04.json`. Do not score the first 1348 decoded frames as a substitute, and do not treat the remaining 350 frames as negatives.

Training from scratch with `scripts/train.py` uses `max_epochs: 300` from `configs/training/noleak.yaml`. That setting is a new-run default. Archived table JSON files belong to the runs in `configs/experiments/catalog.yaml` (VideoPlay: configured 100, log ends at 60; other suite jobs: 50/50), not to a 300-epoch retrain.
