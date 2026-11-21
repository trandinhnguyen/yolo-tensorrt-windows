# Contributing

Thanks for your interest in improving YOLO TensorRT for Windows! Bug reports,
fixes, documentation and new features are all welcome.

## Before you start

- **Bugs:** search the [issues](https://github.com/trandinhnguyen/yolo-tensorrt-windows/issues)
  first, then open a bug report with your GPU, driver version, pipeline and the
  console log.
- **Larger changes** (new pipelines, models or dependencies): please open an
  issue to discuss the idea before you invest time in a pull request.
- **Security issues:** don't open a public issue; see [SECURITY.md](SECURITY.md).

## Development setup

Running the app needs Windows, an NVIDIA GPU and the native libraries installed
by the setup script:

```ps1
.\scripts\setup.ps1
.venv\Scripts\Activate.ps1
python app.py
```

The unit tests need **no GPU, TensorRT or FFmpeg**; a plain Python 3.10
environment is enough. PyTorch from PyPI (CPU build) works for them:

```ps1
uv venv --python 3.10
uv pip install -r pyproject.toml --extra dev
.venv\Scripts\python -m pytest
```

## Making a change

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep tunables in `yolo_tensorrt_windows/config.py` and
   file locations in `yolo_tensorrt_windows/paths.py`.
3. Add or update tests in `tests/`. Code that needs a GPU or TensorRT can't run
   in CI; keep it thin, and describe in the pull request how you verified it
   (GPU model, pipeline, video used).
4. Run the same checks as CI:

   ```ps1
   ruff check .
   ruff format --check .
   python -m pytest
   ```

5. Open a pull request with a clear description of what changed and why. Keep
   pull requests focused; several small ones are easier to review than one large
   one.

## Project structure

See the "Project layout" section of the [README](README.md). In short: the
`processors/` package holds the four pipelines, `inference/` the TensorRT
runners, `tracking/` ByteTrack, `ui.py` the desktop window and `jobs.py` the
worker processes behind it.

## License

By contributing, you agree that your contributions are licensed under the
[GNU AGPL v3.0](LICENSE), the license of this project.
