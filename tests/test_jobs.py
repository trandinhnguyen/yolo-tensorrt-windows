"""WorkerPool with real worker processes running fake processors."""

import time

import pytest
from fakes import FailingProcessor, FakeProcessor, ThreadedFakeProcessor

from yolo_tensorrt_windows.jobs import WorkerPool
from yolo_tensorrt_windows.task import Status
from yolo_tensorrt_windows.video import probe_video

FINAL = (Status.DONE, Status.FAILED)


def wait_for(pool: WorkerPool, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pool.poll()
        if all(job.status in FINAL for job in pool.jobs.values()):
            return
        time.sleep(0.1)
    raise TimeoutError({job.id: job.status for job in pool.jobs.values()})


def test_pool_runs_jobs_across_processes(video_path):
    pool = WorkerPool(
        {
            "Sync": FakeProcessor,
            "Async": ThreadedFakeProcessor,
            "Broken": FailingProcessor,
        }
    )
    pool.start()
    try:
        sync = pool.submit(video_path, "Sync")
        same_video_other_pipeline = pool.submit(video_path, "Async")
        broken = pool.submit(video_path, "Broken")
        assert [sync.id, same_video_other_pipeline.id, broken.id] == [1, 2, 3]
        wait_for(pool)
    finally:
        pool.stop()

    assert sync.status == Status.DONE and sync.throughput > 0
    assert same_video_other_pipeline.status == Status.DONE
    assert sync.output_path != same_video_other_pipeline.output_path
    for job in (sync, same_video_other_pipeline):
        assert probe_video(job.output_path).n_frames == 12
    assert broken.status == Status.FAILED and broken.error == "inference exploded"


def test_submit_rejects_unreadable_video(tmp_path):
    pool = WorkerPool({"Sync": FakeProcessor})  # not started: nothing runs
    with pytest.raises(OSError):
        pool.submit(tmp_path / "missing.mp4", "Sync")
    assert pool.jobs == {}
