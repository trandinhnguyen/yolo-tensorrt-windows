"""Desktop window, driven offscreen with an in-process stand-in for WorkerPool."""

import itertools
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from fakes import one_box  # noqa: E402

from yolo_tensorrt_windows.task import Job, Status, StatusUpdate  # noqa: E402
from yolo_tensorrt_windows.ui import MainWindow  # noqa: E402
from yolo_tensorrt_windows.video import annotate_video  # noqa: E402

PIPELINES = {"Sync CPU": None, "Async CPU": None, "Sync GPU": None, "Async GPU": None}


class FakeJobs:
    """Records submissions; tests push status updates through ``updates``."""

    def __init__(self):
        self.pipelines = PIPELINES
        self.jobs: dict[int, Job] = {}
        self.updates: list[StatusUpdate] = []
        self._ids = itertools.count(1)

    def submit(self, path, pipeline):
        job = Job.create(next(self._ids), path, pipeline)
        self.jobs[job.id] = job
        return job

    def poll(self):
        changed = []
        for update in self.updates:
            job = self.jobs[update.job_id]
            job.apply(update)
            changed.append(job)
        self.updates.clear()
        return changed


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp):
    win = MainWindow(FakeJobs(), poll_ms=10_000)  # poll manually
    win.show()
    yield win
    win.close()


def cell(win, row, column):
    return win.table.item(row, column).text()


def test_buttons_need_a_video_first(window, video_path):
    assert not any(b.isEnabled() for b in window.pipeline_buttons.values())
    window.set_video(str(video_path))
    assert all(b.isEnabled() for b in window.pipeline_buttons.values())


def test_job_rows_follow_status_updates(window, video_path):
    window.set_video(str(video_path))
    window.pipeline_buttons["Sync CPU"].click()
    window.pipeline_buttons["Async GPU"].click()
    assert window.table.rowCount() == 2
    assert [cell(window, 0, c) for c in range(3)] == ["clip.mp4", "Sync CPU", "Queued"]
    assert window.selected_job().pipeline == "Async GPU"  # newest job selected

    # The first job finishes; its annotated video gets written.
    job = window.jobs.jobs[1]
    annotate_video(video_path, [one_box(i) for i in range(12)], job.output_path)
    window.jobs.updates += [
        StatusUpdate(1, Status.ANNOTATING, elapsed=0.5),
        StatusUpdate(1, Status.DONE),
        StatusUpdate(2, Status.FAILED, error="no GPU"),
    ]
    window.poll()
    assert cell(window, 0, 2) == "Done" and cell(window, 0, 3) == "24.0"
    assert cell(window, 1, 2) == "Failed"
    assert "Error: no GPU" in window.details.toPlainText()  # selected row refreshed
    assert window.player.source is None

    window.table.selectRow(0)
    assert "throughput 24.00 fps" in window.details.toPlainText()
    assert window.open_folder_button.isEnabled()
    assert window.player.source == job.output_path
    assert window.player.screen.pixmap() is not None  # first frame shown


def test_player_controls(window, video_path, tmp_path):
    out = tmp_path / "annotated.mp4"
    annotate_video(video_path, [one_box(i) for i in range(12)], out)
    player = window.player
    player.load(out)
    assert player.slider.maximum() == 11 and player.play_button.isEnabled()

    player.toggle()
    assert player.playing
    player.toggle()
    assert not player.playing

    player.seek(11)
    assert player.slider.value() == 11
    player.clear()
    assert player.source is None and not player.play_button.isEnabled()


def test_unreadable_video_shows_error_in_status_bar(window, tmp_path):
    window.set_video(str(tmp_path / "missing.mp4"))
    window.pipeline_buttons["Sync CPU"].click()
    assert window.table.rowCount() == 0
    assert "Could not open video" in window.statusBar().currentMessage()


def test_engine_build_progress_dialog(qapp, monkeypatch, tmp_path):
    import time

    from yolo_tensorrt_windows import app as app_module
    from yolo_tensorrt_windows.inference import engine

    target = tmp_path / "engine.trt"
    monkeypatch.setattr(engine, "engine_path", lambda: target)

    def slow_build():
        time.sleep(0.3)
        target.write_bytes(b"engine")

    monkeypatch.setattr(engine, "ensure_engine", slow_build)
    app_module._ensure_engine_with_progress(qapp)  # returns once the build is done
    assert target.exists()
    app_module._ensure_engine_with_progress(qapp)  # engine exists: no-op

    def failing_build():
        raise RuntimeError("ONNX parse failed")

    monkeypatch.setattr(engine, "engine_path", lambda: tmp_path / "missing.trt")
    monkeypatch.setattr(engine, "ensure_engine", failing_build)
    with pytest.raises(RuntimeError, match="ONNX parse failed"):
        app_module._ensure_engine_with_progress(qapp)
