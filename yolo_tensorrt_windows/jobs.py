"""Worker processes and job bookkeeping, independent of the UI."""

import itertools
import multiprocessing as mp
import os
import queue

from .processors.base import BaseProcessor, run_worker
from .task import Job, Status, StatusUpdate

FINAL_STATUSES = (Status.DONE, Status.FAILED)


class WorkerPool:
    """One worker process per pipeline, each fed by its own job queue.

    Not thread-safe: call it from one thread (the UI thread).
    """

    def __init__(
        self,
        pipelines: dict[str, type[BaseProcessor]],
        log_dir: str | os.PathLike | None = None,
    ):
        self.pipelines = dict(pipelines)
        self.jobs: dict[int, Job] = {}
        self._log_dir = None if log_dir is None else str(log_dir)
        self._ids = itertools.count(1)
        self._queues = {name: mp.Queue() for name in self.pipelines}
        self._updates: mp.Queue = mp.Queue()
        self._processes: dict[str, mp.Process] = {}

    def start(self) -> None:
        for name, processor_cls in self.pipelines.items():
            process = mp.Process(
                target=run_worker,
                args=(processor_cls, self._queues[name], self._updates, self._log_dir),
                name=name,
                daemon=True,
            )
            process.start()
            self._processes[name] = process

    def stop(self, timeout: float = 2.0) -> None:
        """Ask workers to exit; terminate those still busy after ``timeout``."""
        for q in self._queues.values():
            q.put(None)
        for process in self._processes.values():
            process.join(timeout)
            if process.is_alive():
                process.terminate()
        self._processes.clear()

    def submit(self, path: str | os.PathLike, pipeline: str) -> Job:
        """Queue ``path`` on ``pipeline``. Raises OSError for unreadable videos."""
        job = Job.create(next(self._ids), path, pipeline)
        self.jobs[job.id] = job
        self._queues[pipeline].put(job.request())
        return job

    def poll(self) -> list[Job]:
        """Apply pending worker updates; return the jobs that changed.

        Unfinished jobs of a worker that has died are marked as failed.
        """
        changed: dict[int, Job] = {}
        while True:
            try:
                update = self._updates.get_nowait()
            except queue.Empty:
                break
            job = self.jobs.get(update.job_id)
            if job is not None:
                job.apply(update)
                changed[job.id] = job

        dead = {name for name, p in self._processes.items() if not p.is_alive()}
        for job in self.jobs.values():
            if job.pipeline in dead and job.status not in FINAL_STATUSES:
                job.apply(
                    StatusUpdate(
                        job.id,
                        Status.FAILED,
                        error=f"The {job.pipeline} worker stopped unexpectedly; "
                        "see its log file for details.",
                    )
                )
                changed[job.id] = job
        return list(changed.values())
