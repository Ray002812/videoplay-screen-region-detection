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

校验值见 [weights/manifest.json](weights/manifest.json)。导出时的 `metadata.yaml` 原样保留，其中的历史训练路径不参与推理。

## 推理一段视频

```text
python scripts/infer_video.py --video path/to/20240509_160438.mp4 --video-name 20240509_160438 --output outputs/scene_01_pred.json
```

`--video-name` 必须与 [configs/evaluation/scenarios.yaml](configs/evaluation/scenarios.yaml) 中的 `image_id` 前缀一致。`scene_01` 为 Bilibili（`20240509_160438`），`scene_02` 为 Webpage（`20240509_151154`）。默认阈值 T=25，T1=160，T2=3，T3=5，缓冲 5 帧。`--no-motion` 关闭时序门控。

## 数据

完整图像和 MP4 不在 Git 中。见 [data/README.md](data/README.md)。评测标注在 `data/sample/annotations/`。PPT（`scene_04`）共 1348 个标注帧：1304 正样本、44 负样本。AP 只在这些 `image_id` 上计算。

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
python scripts/evaluate.py --pred outputs/scene_01_pred.json --gt data/sample/annotations/scene_01.json
python scripts/reproduce.py --video-root path/to/videos
```

[configs/training/noleak.yaml](configs/training/noleak.yaml) 是**新训练**默认（最多 300 轮），不能用来声称复现现有表格。归档实验见 [configs/experiments/catalog.yaml](configs/experiments/catalog.yaml)：VideoPlay 配置为 100 轮，日志停在第 60 轮；其余套件任务配置为 50 轮并跑满 50 轮。`legacy_run_id` 是原始目录名。

带时序门控的 INT8 结果见 [results/paper/metrics/videoplay_motion_eval.json](results/paper/metrics/videoplay_motion_eval.json)。

| id | 展示名称 | AP50:95 | AP50 | FPS（记录值） |
|----|----------|---------|------|----------------|
| scene_01 | Bilibili | 0.794 | 0.823 | 93.2 |
| scene_02 | Webpage | 0.947 | 0.950 | 100.8 |
| scene_03 | Excel | 0.887 | 0.891 | 115.3 |
| scene_04 | PPT | 0.889 | 0.981 | 34.9 |

## 测试

```text
pip install -e .[dev]
pytest
```

## 许可

整体为 AGPL-3.0-or-later（[LICENSE](LICENSE)）。屏幕录像及其中的第三方画面不受该许可自动覆盖。

## 范围

四条评测录像不能推广为通用播放状态识别。时序门控可能压掉低运动视频。训练图像不随仓库分发。归档表格不是 300 轮默认配置的产物。根目录 `CITATION.cff` 不是已发表论文的 DOI 记录。
