# VideoPlay

VideoPlay 用于在混合桌面画面中定位正在播放的视频区域。紧凑的 YOLO11 派生检测器给出候选框，帧差规则再剔除更像滚屏或光标移动的候选。发布运行时是 CPU 上的 OpenVINO INT8。

论文（返修中，尚未正式发表）：*VideoPlay: Lightweight Video-Region Localization and Temporal Playback Confirmation in Mixed Screen Content*，IET Image Processing，稿号 IPR-2026-03-0268。作者：赵文希、朱嘉桢、叶宇凡、赖昌材（杭州电子科技大学）。通讯作者：cclai@hdu.edu.cn。

English: [README.md](README.md).

## 环境

- Python 3.10 及以上（开发时为 3.12）。
- Windows 11 或 Linux。
- 默认推理只需 CPU。
- 训练需要 NVIDIA GPU。

不要用 PyPI 上任意版本的 `ultralytics` 替代本仓库的 `third_party/ultralytics`（8.3.223，且 `Conv.default_act` 为 ReLU）。说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## CPU 推理安装

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements/inference.txt
pip install -e .
```

Linux 使用 `source .venv/bin/activate`。评测再安装：

```text
pip install -r requirements/evaluation.txt
```

## GPU 训练安装

```text
pip install -r requirements/training.txt
```

请安装与本机驱动匹配的 CUDA 版 PyTorch。论文训练使用 RTX 5070。INT8 导出需要 NNCF（已写入 `requirements/training.txt`）。

## 正式模型

默认权重只有：

```text
weights/videoplay-v1.0/openvino-int8/best
```

校验值见 [weights/manifest.json](weights/manifest.json)。这不是体积约 2.3 MiB 的旧导出包。`results/paper/metrics` 对应本目录中 133,960 字节的 BIN。

## 推理一段视频

```text
python scripts/infer_video.py --video path/to/clip.mp4 --video-name 20240509_160438 --output outputs/webpage_pred.json
```

`--video-name` 必须与 [configs/evaluation/scenarios.yaml](configs/evaluation/scenarios.yaml) 中的 `image_id` 前缀一致。帧编号从 1 开始。默认阈值 T=25，T1=160，T2=3，T3=5，缓冲 5 帧。`--no-motion` 关闭时序门控。

## 数据

完整图像和 MP4 不在 Git 中。见 [data/README.md](data/README.md)。评测标注在 `data/sample/annotations/`。PPT 使用 1348 张图、1304 个框的版本。

划分数据：

```text
python scripts/prepare_dataset.py --source-images path/to/images --out path/to/noleak
python scripts/prepare_dataset.py --source-images path/to/images --out path/to/noleak --apply
```

未加 `--apply` 时只打印计划。`--apply` 会写文件，并删除与 val/test SHA-256 重复的训练集图像。

## 训练、导出、评测

```text
python scripts/train.py --data path/to/noleak/video_data_noleak.yaml --only videoplay
python scripts/export_openvino.py --weights outputs/train/videoplay/weights/best.pt --data path/to/noleak/video_data_noleak.yaml
python scripts/evaluate.py --pred outputs/webpage_pred.json --gt data/sample/annotations/webpage.json
python scripts/reproduce.py --video-root path/to/videos
```

任务列表见 [configs/training/noleak.yaml](configs/training/noleak.yaml)。研究档案中的 `abl_ch14` 在此对应 `ablation_half_channels`（宽度系数 0.25）。

带时序门控的 INT8 结果见 [results/paper/metrics/videoplay_motion_eval.json](results/paper/metrics/videoplay_motion_eval.json)。

## 测试

```text
pip install -e .[dev]
pytest
```

## 许可

整体为 AGPL-3.0-or-later（[LICENSE](LICENSE)）。屏幕录像及其中的第三方画面不受该许可自动覆盖。

## 范围

四条评测录像不能推广为通用播放状态识别。时序门控可能压掉低运动视频。训练图像不随仓库分发。根目录 `CITATION.cff` 不是已发表论文的 DOI 记录。
