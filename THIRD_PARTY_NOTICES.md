# Third-party notices

This project is licensed under the GNU Affero General Public License v3.0 (see
[LICENSE](LICENSE)). It includes, derives from, or works with the third-party
components below, which remain under their own licenses.

## Included in this repository

### NVIDIA TensorRT Python package loaders

`yolo_tensorrt_windows/inference/tensorrt_853/__init__.py` and
`yolo_tensorrt_windows/inference/tensorrt_1001/__init__.py` are the `__init__.py`
files of NVIDIA's `tensorrt` Python wheels (8.5.3 and 10.0.1), reformatted.

- Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- License: Apache License 2.0 — <https://www.apache.org/licenses/LICENSE-2.0>

### ByteTrack tracker

`yolo_tensorrt_windows/tracking/` is a PyTorch port of the ByteTrack
implementation in Roboflow `supervision`, which is itself based on the original
ByteTrack code.

- Roboflow supervision — <https://github.com/roboflow/supervision> — MIT License,
  Copyright (c) 2022 Roboflow
- ByteTrack — <https://github.com/ifzhang/ByteTrack> — MIT License,
  Copyright (c) 2021 Yifu Zhang

The Kalman filter (`tracking/kalman_filter.py`) follows the one in
[deep_sort](https://github.com/nwojke/deep_sort) by Nicolai Wojke, licensed under
the GNU General Public License v3.0, which may be combined with AGPL-3.0 code
(GPL-3.0 section 13).

Both MIT-licensed projects are distributed under these terms:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### Screenshot sample image

`docs/screenshot.png` shows a frame made from `bus.jpg`, a sample image from
[ultralytics/assets](https://github.com/ultralytics/assets) (AGPL-3.0).

## Downloaded during setup, not included in this repository

`scripts/setup.ps1` downloads these onto your machine. They are not part of this
repository, but executables built with `scripts/build.ps1` bundle some of them;
if you distribute such a build, you must comply with their licenses too.

| Component | Used for | License |
| --- | --- | --- |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n` weights, exported to ONNX) | the detection model | AGPL-3.0 |
| [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) 8.5.3 and 10.0.1 | inference | [NVIDIA TensorRT license](https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html) |
| NVIDIA CUDA runtime, cuBLAS, cuFFT, NVRTC and cuDNN (`nvidia-*-cu11` wheels) | libraries TensorRT loads | NVIDIA software license agreements shipped with each wheel |
| [FFmpeg](https://ffmpeg.org) 6.1.1, [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) full shared build | GPU video decoding (NVDEC) | GPL-3.0 (this build enables GPL components) |
| [Qt](https://www.qt.io) via [PySide6](https://doc.qt.io/qtforpython/) | the desktop window | LGPL-3.0; builds ship the Qt DLLs as separate, replaceable files |
| Other Python packages in `pyproject.toml` (PyTorch, OpenCV, SciPy, ...) | the application | their respective licenses |
