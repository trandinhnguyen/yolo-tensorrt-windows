"""Video I/O, jobs and the processor base class, with a fake model."""

import queue

import cv2
import pytest
from fakes import FailingProcessor, FakeProcessor, one_box

from yolo_tensorrt_windows.paths import output_path_for
from yolo_tensorrt_windows.task import Job, JobRequest, Status, StatusUpdate
from yolo_tensorrt_windows.video import annotate_video, probe_video, read_frames


def test_probe_and_read(video_path):
    info = probe_video(video_path)
    assert (info.width, info.height, info.n_frames) == (160, 90, 12)
    assert info.duration == pytest.approx(1.2)
    assert sum(1 for _ in read_frames(video_path)) == 12


def test_unreadable_video_raises(tmp_path):
    with pytest.raises(OSError):
        probe_video(tmp_path / "missing.mp4")


def test_annotate_video_writes_every_frame(video_path, tmp_path):
    out = tmp_path / "annotated.mp4"
    # Fewer detections than frames: remaining frames are copied unannotated.
    annotate_video(video_path, [one_box(i) for i in range(5)], out)

    assert probe_video(out).n_frames == 12
    first = next(read_frames(out))
    assert (
        first.max() > 0 and cv2.absdiff(first, next(read_frames(video_path))).max() > 50
    )


def test_output_paths_are_per_pipeline(tmp_path):
    assert output_path_for("a/clip.mp4").name == "clip.mp4"
    assert output_path_for("a/clip.mov", "Sync CPU").name == "clip_sync-cpu.mp4"


def test_job_lifecycle(video_path):
    job = Job.create(7, video_path, "Async GPU")
    assert job.name == "clip.mp4" and job.status == Status.QUEUED
    assert job.output_path == output_path_for(video_path, "Async GPU")
    assert job.request() == JobRequest(7, str(video_path), str(job.output_path))
    assert "Resolution: 160x90" in job.describe() and job.throughput == 0

    job.apply(StatusUpdate(7, Status.ANNOTATING, elapsed=0.5))
    assert job.throughput == pytest.approx(24)
    assert "throughput 24.00 fps" in job.describe()

    job.apply(StatusUpdate(7, Status.FAILED, error="boom"))
    assert job.status == Status.FAILED and "Error: boom" in job.describe()


def request_for(video_path, job_id=1):
    return JobRequest(job_id, str(video_path), str(output_path_for(video_path, "x")))


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("tracking", [False, True])
def test_processor_reports_progress_and_writes_output(video_path, threaded, tracking):
    q_out = queue.Queue()
    proc = FakeProcessor(queue.Queue(), q_out)
    proc.threaded, proc.tracking = threaded, tracking

    detections = proc.detect(str(video_path))
    assert len(detections) == 12
    if tracking:
        assert [int(d.tracker_ids[0]) for d in detections] == [1] * 12
    else:
        assert all(d.tracker_ids is None for d in detections)

    request = request_for(video_path, job_id=3)
    proc.process(request)
    updates = [q_out.get_nowait() for _ in range(q_out.qsize())]
    assert [u.status for u in updates] == [
        Status.PROCESSING,
        Status.ANNOTATING,
        Status.DONE,
    ]
    assert {u.job_id for u in updates} == {3}
    assert probe_video(request.output_path).n_frames == 12


def test_worker_loop_reports_failures_and_keeps_serving(video_path):
    q_in, q_out = queue.Queue(), queue.Queue()
    for item in (request_for(video_path, 1), request_for(video_path, 2), None):
        q_in.put(item)
    FailingProcessor(q_in, q_out).serve_forever()  # returns at the None sentinel

    updates = [q_out.get_nowait() for _ in range(q_out.qsize())]
    failures = [u for u in updates if u.status == Status.FAILED]
    assert [u.job_id for u in failures] == [1, 2]
    assert failures[0].error == "inference exploded"
