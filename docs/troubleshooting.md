# Troubleshooting

**ImportError: ultralytics**  
Run the scripts from this repository so `scripts/_bootstrap.py` can add `third_party/ultralytics`. Do not rely on a random PyPI wheel.

**Wrong boxes compared with the paper**  
Check that `--model` points at `weights/videoplay-v1.0/openvino-int8/best` and that motion is enabled unless you are measuring detector-only AP.

**COCO image_id mismatch**  
`--video-name` must equal the scene `image_id_prefix`. PPT (`scene_04`) uses `20250125_1`, not `01`. Bilibili (`scene_01`) uses `20240509_160438`. Webpage (`scene_02`) uses `20240509_151154`.

**SHA-256 of XML/YAML does not match the manifest after clone**  
This directory is marked `-text` in `.gitattributes`. Re-clone or restore those three files without line-ending conversion.

**prepare_dataset deleted train images**  
That happens only with `--apply`, and only for train files whose SHA-256 appears in val or test. Use the dry run first.

**openvino-dev vs openvino**  
Inference needs the `openvino` package. `openvino-dev` is not required for `scripts/infer_video.py`.
