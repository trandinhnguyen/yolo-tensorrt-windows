"""Stand-in processors that need no GPU or TensorRT.

Kept in their own module so spawned worker processes can import them.
"""

import numpy as np

from yolo_tensorrt_windows.detection import Detections, Letterbox
from yolo_tensorrt_windows.processors.base import BaseProcessor
from yolo_tensorrt_windows.video import probe_video, read_frames


def one_box(frame_index: int) -> Detections:
    """One person box moving 5 px right per frame."""
    x = 10 + frame_index * 5
    return Detections(
        np.array([[x, 20, x + 30, 60]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([0]),
    )


class FakeProcessor(BaseProcessor):
    """Behaves like a detector following one moving object."""

    def open_video(self, path):
        info = probe_video(path)
        return info, Letterbox.fit(info.width, info.height), read_frames(path)

    def build_stages(self, letterbox):
        counter = iter(range(10**6))
        return [lambda frame: one_box(next(counter))]


class ThreadedFakeProcessor(FakeProcessor):
    threaded = True
    tracking = True


class FailingProcessor(FakeProcessor):
    def build_stages(self, letterbox):
        def fail(frame):
            raise RuntimeError("inference exploded")

        return [fail]
