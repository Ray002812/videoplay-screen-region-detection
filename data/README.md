# Data

This repository does not contain the full image dataset or the evaluation videos.

## What is included

- `sample/annotations/scene_01.json` … `scene_04.json`: COCO-style boxes for four held-out screen recordings.
- [configs/evaluation/scenarios.yaml](../configs/evaluation/scenarios.yaml): stable scene id, display name, `legacy_case_id`, video file name, annotation path, and `image_id` prefix.

Display names follow the manuscript: `scene_01` is Bilibili (`20240509_160438`), `scene_02` is Webpage (`20240509_151154`). `legacy_case_id` records the original evaluation-script labels (`Video1_Webpage`, `Video2_Bilibili`) and is not the display name.

PPT (`scene_04`) lists 1348 frames: 1304 positive and 44 negative. The recording has 1698 decoded frames; the extra 350 frames are outside this evaluation set. A larger similarly named JSON exists only in the local research archive and is not this file.

## What is not included

- Raw frames under `data/data` or `data/noleak`.
- `test_video/*.mp4` (the PPT clip exceeds GitHub's ordinary file-size limit).
- Automatically generated caches.

Screen recordings may contain copyrighted UI, sports, film, or presentation material. The software license does not grant redistribution rights for those media. Request access from the corresponding author (`cclai@hdu.edu.cn`) if you need the videos or training images.

## Preparing a training split

Point `scripts/prepare_dataset.py` at a local `images/{train,val,test}` tree. The default is a dry run. `--apply` writes hard links or copies and then removes train images whose SHA-256 also appears in val or test.

The generated YAML uses an absolute `path:` for the output directory so Ultralytics can locate files on that machine. Do not commit that generated YAML if it contains a private path.
