"""Logging setup, worker failure handling and the headless benchmark mode."""

import sys
import time

import pytest
from fakes import CrashingProcessor, FakeProcessor, ThreadedFakeProcessor
from loguru import logger

from yolo_tensorrt_windows.app import run_benchmark
from yolo_tensorrt_windows.jobs import FINAL_STATUSES, WorkerPool
from yolo_tensorrt_windows.log import configure_logging
from yolo_tensorrt_windows.task import Status


@pytest.fixture(autouse=True)
def _restore_logger():
    yield
    logger.remove()
    logger.add(sys.stderr)


def test_configure_logging_writes_a_file_per_process(tmp_path):
    path = configure_logging("Sync CPU", tmp_path / "logs")
    logger.info("hello from the test")
    assert path == tmp_path / "logs" / "sync-cpu.log"
    assert "hello from the test" in path.read_text(encoding="utf-8")


def test_dead_worker_fails_its_jobs_and_logs_why(video_path, tmp_path):
    pool = WorkerPool({"GPU": CrashingProcessor}, log_dir=tmp_path)
    pool.start()
    try:
        job = pool.submit(video_path, "GPU")
        for _ in range(600):
            pool.poll()
            if job.status in FINAL_STATUSES:
                break
            time.sleep(0.1)
    finally:
        pool.stop()
    assert job.status == Status.FAILED
    assert "worker stopped unexpectedly" in job.error
    worker_log = (tmp_path / "gpu.log").read_text(encoding="utf-8")
    assert "failed to start" in worker_log and "CUDA driver not found" in worker_log


def test_benchmark_reports_all_pipelines(video_path, tmp_path):
    configure_logging("main", tmp_path)
    pipelines = {"Sync": FakeProcessor, "Async": ThreadedFakeProcessor}
    assert run_benchmark(str(video_path), pipelines, log_dir=tmp_path) == 0
    log = (tmp_path / "main.log").read_text(encoding="utf-8")
    assert "Benchmark results" in log
    assert "Sync       Done" in log and "Async      Done" in log
    assert (tmp_path / "sync.log").exists() and (tmp_path / "async.log").exists()


def test_benchmark_fails_when_a_pipeline_fails(video_path, tmp_path):
    pipelines = {"Sync": FakeProcessor, "Broken": CrashingProcessor}
    assert run_benchmark(str(video_path), pipelines, log_dir=tmp_path) == 1
