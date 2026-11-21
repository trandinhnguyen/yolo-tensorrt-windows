# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-21

First public release.

### Added

- Windows desktop app (PySide6) to queue videos on four detection pipelines
  (sync/async × CPU/GPU) and compare their throughput, with a job table, details
  and a built-in player for the annotated output.
- YOLOv8n inference with NVIDIA TensorRT 10.0 (TensorRT 8.5 on Pascal and older
  GPUs); the engine is built from the ONNX model on first start.
- GPU pipelines that keep frames on the GPU from NVDEC decoding to
  post-processing.
- ByteTrack multi-object tracking in PyTorch (enabled in the Sync CPU pipeline).
- `scripts/setup.ps1`: one-command setup of TensorRT, CUDA libraries, FFmpeg,
  the ONNX model and the Python environment.
- PyInstaller build (`scripts/build.ps1`), unit tests that run without a GPU, and
  CI on GitHub Actions.

[0.1.0]: https://github.com/trandinhnguyen/yolo-tensorrt-windows/releases/tag/v0.1.0
