# Changelog

## 1.0.1

Public layout aligned with archived experiment records:

- OpenVINO XML/YAML stored as binary to keep export bytes.
- Scene ids `scene_01`–`scene_04` with manuscript display names; original script labels kept as `legacy_case_id`.
- Experiment catalog records max/completed/best epochs from `args.yaml` and `results.csv`.
- Training script resolves data/config/project from the launch working directory and reads YAML settings.
- Result JSON files drop machine paths; numeric values unchanged.

## 1.0.0

First public code layout:

- Default inference model is `weights/videoplay-v1.0/openvino-int8`.
- Temporal gate lives in `src/videoplay/motion.py`.
- Command-line entry points are under `scripts/`.
- Local Ultralytics 8.3.223 tree is vendored in `third_party/ultralytics` because `Conv.default_act` is ReLU.
