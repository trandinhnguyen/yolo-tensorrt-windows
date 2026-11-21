"""OpenCV video reading/writing shared by the pipelines and the UI."""

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from . import config
from .detection.detections import Detections
from .detection.draw import draw_detections


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def _open(path: str | os.PathLike) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    return cap


def probe_video(path: str | os.PathLike) -> VideoInfo:
    cap = _open(path)
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frames(path: str | os.PathLike) -> Iterator[np.ndarray]:
    """Yield BGR frames until the end of the video."""
    cap = _open(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def annotate_video(
    input_path: str | os.PathLike,
    detections: Sequence[Detections],
    output_path: str | os.PathLike,
) -> None:
    """Write a copy of the input video with ``detections[i]`` drawn on frame i."""
    info = probe_video(input_path)
    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_FOURCC)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, info.fps, (info.width, info.height)
    )
    try:
        for i, frame in enumerate(read_frames(input_path)):
            if i < len(detections):
                draw_detections(frame, detections[i])
            writer.write(frame)
    finally:
        writer.release()
    logger.info(f"Saved to: {output_path}")
