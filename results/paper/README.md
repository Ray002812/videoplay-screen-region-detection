# Paper results

Files in `metrics/` are machine-readable outputs corresponding to the published OpenVINO INT8 model (`weights/videoplay-v1.0/openvino-int8`) on the four evaluation recordings.

- `videoplay_motion_eval.json`: detector plus temporal gate (default pipeline).
- `ablation_detector_eval.json`: detector-only ablations.
- `sota_detector_eval.json`: YOLO nano baselines, detector-only.
- `code_side_complete.json`: combined code-side record.

These numbers are tied to the released INT8 package. They are not interchangeable with older OpenVINO directories whose BIN files have a different size.

`figure_data/` and `manifests/` are reserved for extra table dumps. Manuscript LaTeX is not part of this repository.
