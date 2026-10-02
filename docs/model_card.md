# Model card

- Name: VideoPlay-YOLO11 OpenVINO INT8 (`videoplay-v1.0`)
- Task: single-class box for a screen video region, then a temporal confirm/exclude/uncertain rule
- Input: letterboxed RGB, 480, values in [0, 1], pad 114
- Output: OpenVINO detect tensor decoded to xyxy in the original frame
- Size: `best.xml` 501880 bytes, `best.bin` 133960 bytes
- Parameters / FLOPs (PyTorch graph of `configs/models/videoplay.yaml`): about 1.06e5 parameters and 0.148 GFLOPs at 480, as reported in the manuscript
- Intended use: desktop CPU localization of an embedded player in mixed screen sharing
- Out of scope: mobile deployment claims, codec bitrate savings, exhaustive playback-state labels
- Ethical / legal: training and test media may include third-party UI; do not treat the AGPL grant as a media license
