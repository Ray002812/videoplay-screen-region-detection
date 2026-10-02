# Third-party notices

This repository vendors a copy of Ultralytics 8.3.223 under `third_party/ultralytics`.

- Upstream project: https://github.com/ultralytics/ultralytics
- Upstream license: GNU Affero General Public License v3.0 (`third_party/ultralytics/LICENSE`)
- Upstream citation: `third_party/ultralytics/CITATION.cff`

VideoPlay as a whole is therefore distributed under AGPL-3.0-or-later. Do not relicense the combined work as MIT.

## Local change relative to upstream 8.3.223

In `third_party/ultralytics/ultralytics/nn/modules/conv.py`, class `Conv` sets

```python
default_act = nn.ReLU()
```

Upstream 8.3.223 uses SiLU. VideoPlay YAML files also set `activation: nn.ReLU()` where that is part of the architecture. Replacing this tree with an unmodified `pip install ultralytics` can change constructed graphs and exported weights.

Personal training logs, `runs/`, `video_480/`, and editor files from the research working copy are not included here.

OpenVINO, NNCF, PyTorch, OpenCV, and pycocotools are used at runtime and remain under their own licenses.
