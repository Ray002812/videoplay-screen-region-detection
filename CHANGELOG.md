# Changelog

## 1.0.0

First public code layout:

- Default inference model is `weights/videoplay-v1.0/openvino-int8`.
- Temporal gate lives in `src/videoplay/motion.py`.
- Command-line entry points are under `scripts/`.
- Local Ultralytics 8.3.223 tree is vendored in `third_party/ultralytics` because `Conv.default_act` is ReLU.
