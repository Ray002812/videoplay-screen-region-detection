# Data

This repository does not contain the full image dataset or the evaluation videos.

## What is included

- `sample/annotations/`: COCO-style boxes for four held-out screen recordings used in the paper tables.
- `configs/evaluation/scenarios.yaml`: scene id, display name, video file name, annotation path, and `image_id` prefix.

PPT evaluation uses `sample/annotations/ppt.json` (1348 images, 1304 boxes). A larger JSON with a similar name exists only in the local research archive and is not this file.

## What is not included

- Raw frames under `data/data` or `data/noleak`.
- `test_video/*.mp4` (the PPT clip exceeds GitHub's ordinary file-size limit).
- Automatically generated caches.

Screen recordings may contain copyrighted UI, sports, film, or presentation material. The software license does not grant redistribution rights for those media. Request access from the corresponding author (`cclai@hdu.edu.cn`) if you need the videos or training images.

## Preparing a training split

Point `scripts/prepare_dataset.py` at a local `images/{train,val,test}` tree. The default is a dry run. `--apply` writes hard links or copies and then removes train images whose SHA-256 also appears in val or test.

The generated YAML uses an absolute `path:` for the output directory so Ultralytics can locate files on that machine. Do not commit that generated YAML if it contains a private path.
