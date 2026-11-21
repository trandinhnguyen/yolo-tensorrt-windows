"""GPU pipelines: NVDEC decoding, torch pre/post-processing, TensorRT inference.

Frames never leave the GPU until the output video is drawn.
"""

from collections.abc import Iterable

from ..paths import add_ffmpeg_dll_directory

add_ffmpeg_dll_directory()  # before torchaudio loads its FFmpeg extension

import torch  # noqa: E402
import torch.cuda as cuda  # noqa: E402
from torchaudio.io import StreamReader  # noqa: E402

from .. import config  # noqa: E402
from ..detection import Detections, Letterbox  # noqa: E402
from ..detection.postprocess import postprocess_gpu  # noqa: E402
from ..detection.preprocess import preprocess_gpu  # noqa: E402
from ..inference.torch_runner import TorchRunner  # noqa: E402
from ..video import VideoInfo  # noqa: E402
from .base import BaseProcessor  # noqa: E402
from .pipeline import Stage  # noqa: E402


def _wait(waiter: cuda.Stream, producer: cuda.Stream) -> None:
    if waiter != producer:
        waiter.wait_stream(producer)


class _GPUProcessor(BaseProcessor):
    def __init__(self, q_in, q_out, n_streams: int):
        super().__init__(q_in, q_out)
        streams = [cuda.Stream() for _ in range(n_streams)]
        # One stream per stage when async; the sync pipeline shares one stream.
        self.pre_stream = streams[0]
        self.trt_stream = streams[min(1, n_streams - 1)]
        self.post_stream = streams[-1]
        self.runner = TorchRunner(self.trt_stream)

    def open_video(self, path: str) -> tuple[VideoInfo, Letterbox, Iterable]:
        reader = StreamReader(path)
        src = reader.get_src_stream_info(0)
        info = VideoInfo(src.width, src.height, src.frame_rate, src.num_frames)
        letterbox = Letterbox.fit(info.width, info.height)

        resized_h, resized_w = letterbox.resized_hw
        reader.add_video_stream(
            frames_per_chunk=1,
            buffer_chunk_size=config.GPU_DECODER_BUFFER_CHUNKS,
            decoder=config.GPU_DECODER,
            decoder_option={"resize": f"{resized_w}x{resized_h}"},
            hw_accel=config.GPU_DEVICE,
        )
        frames = (chunk for (chunk,) in reader.stream())
        return info, letterbox, frames

    def build_stages(self, letterbox: Letterbox) -> list[Stage]:
        decode_stream = torch.cuda.default_stream()

        # Each stage waits for the previous stage's stream, and marks its input
        # as used on its own stream so the caching allocator doesn't recycle it
        # while still in flight.
        def preprocess(frame: torch.Tensor) -> torch.Tensor:
            _wait(self.pre_stream, decode_stream)
            with cuda.stream(self.pre_stream):
                batch = preprocess_gpu(frame, letterbox)
                frame.record_stream(self.pre_stream)
            return batch

        def infer(batch: torch.Tensor) -> torch.Tensor:
            _wait(self.trt_stream, self.pre_stream)
            with cuda.stream(self.trt_stream):
                raw = self.runner(batch)[0][0]
                batch.record_stream(self.trt_stream)
            return raw

        def postprocess(raw: torch.Tensor) -> Detections:
            _wait(self.post_stream, self.trt_stream)
            with cuda.stream(self.post_stream):
                detections = postprocess_gpu(raw, letterbox)
                raw.record_stream(self.post_stream)
            return detections

        return [preprocess, infer, postprocess]

    def synchronize(self) -> None:
        self.post_stream.synchronize()


class SyncGPUVideoProcessor(_GPUProcessor):
    def __init__(self, q_in, q_out):
        super().__init__(q_in, q_out, n_streams=1)


class AsyncGPUVideoProcessor(_GPUProcessor):
    threaded = True

    def __init__(self, q_in, q_out):
        super().__init__(q_in, q_out, n_streams=3)
