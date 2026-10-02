# Paper results

Numeric values are copied from the archived evaluation dumps. File names, scene labels, and experiment identifiers were rearranged for the public layout. Machine-local weight paths were removed.

Experiment mapping and training counts: [configs/experiments/catalog.yaml](../../configs/experiments/catalog.yaml).

## Files

- `videoplay_motion_eval.json`: released VideoPlay OpenVINO INT8 model plus the temporal gate. Artifact: `weights/manifest.json`.
- `ablation_detector_eval.json`: detector-only results for the custom ablations. Each record has its own `experiment_id`. PyTorch FP32 checkpoints are not in this repository (`checkpoint_in_release: false`).
- `sota_detector_eval.json`: detector-only results for YOLOv8n, YOLO11n, and YOLOv10n trained in the same suite. Checkpoints are not in this repository.
- `code_side_complete.json`: combined code-side record from the archive, with historical commands retained as history rather than as the current CLI.

Do not treat the ablation or SOTA JSON files as outputs of the released INT8 package.

## Scenes

Stable ids are `scene_01`–`scene_04`. Display names follow the manuscript. `legacy_case_id` is the original script label.

PPT localization uses 1348 annotated frames (1304 positive, 44 negative) out of 1698 decoded frames. The remaining 350 frames are excluded from AP. FPS, when reported, is timed over decoded frames and is not the same scope as AP.

`n_predictions_evaluated_frames` is null where the original dump only stored the count over all processed frames.
