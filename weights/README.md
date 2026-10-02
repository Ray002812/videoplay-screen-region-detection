# Weights

The default published model is `videoplay-v1.0` OpenVINO INT8:

```text
weights/videoplay-v1.0/openvino-int8/best.xml
weights/videoplay-v1.0/openvino-int8/best.bin
weights/videoplay-v1.0/openvino-int8/metadata.yaml
```

Keep the three files together. Inference loads the XML/BIN pair by the `best` stem.

SHA-256 values are recorded in `manifest.json`. `metadata.yaml` still contains the original export description string from the training machine; it is not a runtime path.

This package does not include `.pt` checkpoints. Train and export locally if you need FP32 PyTorch weights.

Place optional downloads under `weights/downloads/` (ignored by Git).
