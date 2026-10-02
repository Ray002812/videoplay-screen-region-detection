# Reproduction

1. Install inference and evaluation requirements.
2. Confirm SHA-256 of `weights/videoplay-v1.0/openvino-int8` against `weights/manifest.json`.
3. Obtain the four evaluation videos (not in Git).
4. Run `python scripts/reproduce.py --video-root <dir>`.
5. Compare `outputs/eval/summary.json` with `results/paper/metrics/videoplay_motion_eval.json`. Absolute AP should be close; FPS will vary.

Training from scratch is a separate path: prepare the no-leak split, run `scripts/train.py`, then `scripts/export_openvino.py`. The archived table JSON files belong to the released INT8 package, not to a newly trained checkpoint, until you replace them.
