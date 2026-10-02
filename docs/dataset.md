# Dataset

See [../data/README.md](../data/README.md) for redistribution limits and the dry-run split tool.

Evaluation scenes, video file names, annotation paths, and `image_id` prefixes are defined in [../configs/evaluation/scenarios.yaml](../configs/evaluation/scenarios.yaml).

Do not rename the MP4 files only. The prediction `image_id` is `{prefix}-{frame_index}` with `frame_index` starting at 1. Changing the file name without updating `image_id_prefix` breaks COCO matching.

Training YAML must use a path that exists on the training machine. The sample [../configs/datasets/noleak.yaml](../configs/datasets/noleak.yaml) is a layout template, not a populated dataset.
