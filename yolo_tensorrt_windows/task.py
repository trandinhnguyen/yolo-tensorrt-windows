"""Jobs (one video through one pipeline) and the messages exchanged with workers."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .paths import output_path_for
from .video import VideoInfo, probe_video


class Status(str, Enum):
    QUEUED = "Queued"
    PROCESSING = "Processing"  # running detection
    ANNOTATING = "Annotating"  # detection done, writing the output video
    DONE = "Done"
    FAILED = "Failed"


@dataclass(frozen=True)
class JobRequest:
    """Sent from the UI process to a worker process."""

    job_id: int
    path: str
    output_path: str


@dataclass(frozen=True)
class StatusUpdate:
    """Sent from a worker process back to the UI process."""

    job_id: int
    status: Status
    elapsed: float | None = None  # detection time in seconds, with ANNOTATING
    error: str | None = None  # with FAILED


@dataclass
class Job:
    id: int
    path: str
    pipeline: str
    info: VideoInfo
    output_path: Path
    status: Status = Status.QUEUED
    processed_time: float = 0.0
    error: str | None = field(default=None, repr=False)

    @classmethod
    def create(cls, job_id: int, path: str | os.PathLike, pipeline: str) -> "Job":
        """Raises OSError if the video can't be opened."""
        return cls(
            job_id,
            str(path),
            pipeline,
            probe_video(path),
            output_path_for(path, pipeline),
        )

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def throughput(self) -> float:
        """Detection frames per second."""
        if self.processed_time <= 0:
            return 0.0
        return self.info.n_frames / self.processed_time

    def request(self) -> JobRequest:
        return JobRequest(self.id, self.path, str(self.output_path))

    def apply(self, update: StatusUpdate) -> None:
        self.status = update.status
        if update.elapsed is not None:
            self.processed_time = update.elapsed
        if update.error is not None:
            self.error = update.error

    def describe(self) -> str:
        info = self.info
        lines = [
            f"Video: {self.path}",
            f"Pipeline: {self.pipeline}",
            f"Resolution: {info.width}x{info.height}, {info.fps:.2f} fps, "
            f"{info.duration:.2f} s ({info.n_frames} frames)",
            f"Status: {self.status.value}",
        ]
        if self.processed_time > 0:
            lines.append(
                f"Detection: {self.processed_time:.2f} s, "
                f"throughput {self.throughput:.2f} fps"
            )
        if self.status == Status.DONE:
            lines.append(f"Output: {self.output_path}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)
