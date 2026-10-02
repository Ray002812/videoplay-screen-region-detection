# Weights

The default published model is `videoplay-v1.0` OpenVINO INT8:

```text
weights/videoplay-v1.0/openvino-int8/best.xml
weights/videoplay-v1.0/openvino-int8/best.bin
weights/videoplay-v1.0/openvino-int8/metadata.yaml
```

Keep the three files together. Inference loads the XML/BIN pair by the `best` stem.

SHA-256 values are recorded in `manifest.json`. Git stores this directory as binary (`-text`) so XML and YAML keep the original export bytes.

The original export metadata is preserved for provenance. Its historical training path is descriptive and is not required for inference.

This package does not include `.pt` checkpoints except as referenced by the OpenVINO export. Ablation and baseline PyTorch weights remain in the local research archive. Train and export locally if you need new FP32 checkpoints.

Place optional downloads under `weights/downloads/` (ignored by Git).
