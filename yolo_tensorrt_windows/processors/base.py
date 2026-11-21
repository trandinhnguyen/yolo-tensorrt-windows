import multiprocessing as mp
import os
import queue
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from multiprocessing import Queue

import psutil
import torch
from loguru import logger

from .. import config
from ..detection import Detections, Letterbox
from ..log import configure_logging
from ..task import JobRequest, Status, StatusUpdate
from ..tracking import ByteTrack
from ..video import VideoInfo, annotate_video
from .pipeline import Stage, run_inline, run_threaded


def tracking_stage(tracker: ByteTrack) -> Stage:
    """A pipeline stage that fills in ``Detections.tracker_ids``."""

    def track(detections: Detections) -> Detections:
        detections.tracker_ids = tracker.update_with_detections(
            torch.as_tensor(detections.boxes), torch.as_tensor(detections.scores)
        )
        return detections

    return track


class BaseProcessor(ABC):
    """Detects objects in every frame of a video and writes an annotated copy.

    Subclasses provide frame decoding and the preprocess/infer/postprocess
    stages; this class runs them, optionally tracks, times detection, and
    reports progress. Runs inside a worker process (see :func:`run_worker`).
    """

    threaded: bool = False  # run each stage in its own thread
    tracking: bool = False  # assign tracker IDs with ByteTrack

    def __init__(self, q_in: Queue, q_out: Queue):
        self.q_in = q_in
        self.q_out = q_out

    @abstractmethod
    def open_video(self, path: str) -> tuple[VideoInfo, Letterbox, Iterable]:
        """Video metadata, letterbox geometry and an iterable of decoded frames."""

    @abstractmethod
    def build_stages(self, letterbox: Letterbox) -> list[Stage]:
        """Stages turning one decoded frame into Detections."""

    def synchronize(self) -> None:  # noqa: B027 (optional hook)
        """Block until all queued device work is finished (GPU pipelines)."""

    def detect(self, path: str) -> list[Detections]:
        info, letterbox, frames = self.open_video(path)
        stages = self.build_stages(letterbox)
        if self.tracking:
            stages.append(tracking_stage(ByteTrack(frame_rate=info.fps)))
        run = run_threaded if self.threaded else run_inline

        results: list[Detections] = []
        start = time.perf_counter()
        for detections in run(stages, frames):
            results.append(detections)
            if len(results) % config.LOG_EVERY_N_FRAMES == 0:
                fps = len(results) / (time.perf_counter() - start)
                logger.debug(f"{len(results)}/{info.n_frames}: {fps:.2f}fps")
        self.synchronize()
        return results

    def process(self, request: JobRequest) -> None:
        job_id, path = request.job_id, request.path
        logger.info(f"Processing video: {path}")
        self.q_out.put(StatusUpdate(job_id, Status.PROCESSING))

        start = time.perf_counter()
        detections = self.detect(path)
        elapsed = time.perf_counter() - start
        logger.info(f"Detection done in {elapsed:.2f}s")
        self.q_out.put(StatusUpdate(job_id, Status.ANNOTATING, elapsed=elapsed))

        annotate_video(path, detections, request.output_path)
        self.q_out.put(StatusUpdate(job_id, Status.DONE))

    def serve_forever(self) -> None:
        """Process queued jobs until a ``None`` arrives or the parent exits."""
        parent = psutil.Process(os.getppid())
        while parent.is_running():
            try:
                request: JobRequest | None = self.q_in.get(timeout=1)
            except queue.Empty:
                continue
            if request is None:
                break
            try:
                self.process(request)
            except Exception as e:
                logger.exception(f"Failed to process {request.path}")
                self.q_out.put(
                    StatusUpdate(request.job_id, Status.FAILED, error=str(e))
                )
        logger.info(f"{type(self).__name__} exiting.")


def run_worker(
    processor_cls: type[BaseProcessor],
    q_in: Queue,
    q_out: Queue,
    log_dir: str | None = None,
) -> None:
    """``multiprocessing.Process`` target: build a processor and serve jobs.

    With ``log_dir``, logs go to a file named after the process.
    """
    if log_dir is not None:
        configure_logging(mp.current_process().name, log_dir)
    try:
        processor = processor_cls(q_in, q_out)
    except Exception:
        logger.exception(f"{processor_cls.__name__} failed to start")
        raise
    processor.serve_forever()
