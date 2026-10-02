# Installation

Use a virtual environment. Inference and training dependency lists are separate.

```text
pip install -r requirements/inference.txt
pip install -e .
```

Training and INT8 export:

```text
pip install -r requirements/training.txt
```

COCO evaluation:

```text
pip install -r requirements/evaluation.txt
```

Keep `third_party/ultralytics` on `PYTHONPATH`. The command-line scripts insert that path automatically.

Supported OS: Windows 11 and Linux. Python 3.10+. Default device for `scripts/infer_video.py` is OpenVINO CPU.
