"""CPU pipelines: OpenCV decoding, numpy pre/post-processing, TensorRT inference."""

from collections.abc import Iterable

from ..detection import Letterbox
from ..detection.postprocess import postprocess_cpu
from ..detection.preprocess import preprocess_cpu
from ..inference.cuda_python_runner import CudaPythonRunner
from ..video import VideoInfo, probe_video, read_frames
from .base import BaseProcessor
from .pipeline import Stage


class _CPUProcessor(BaseProcessor):
    def __init__(self, q_in, q_out):
        super().__init__(q_in, q_out)
        self.runner = CudaPythonRunner()

    def open_video(self, path: str) -> tuple[VideoInfo, Letterbox, Iterable]:
        info = probe_video(path)
        return info, Letterbox.fit(info.width, info.height), read_frames(path)

    def build_stages(self, letterbox: Letterbox) -> list[Stage]:
        return [
            lambda frame: preprocess_cpu(frame, letterbox),
            lambda batch: self.runner(batch)[0][0],
            lambda raw: postprocess_cpu(raw, letterbox),
        ]


class SyncCPUVideoProcessor(_CPUProcessor):
    tracking = True


class AsyncCPUVideoProcessor(_CPUProcessor):
    threaded = True
