from pathlib import Path

import cv2
import numpy as np
import pytest


def write_video(path: Path, n_frames: int, width: int, height: int, fps: float = 10.0):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    return write_video(tmp_path / "clip.mp4", n_frames=12, width=160, height=90)


@pytest.fixture(autouse=True)
def _isolated_output_dir(tmp_path, monkeypatch):
    """Keep annotated outputs out of the repo."""
    from yolo_tensorrt_windows import paths

    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "output")
