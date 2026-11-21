import threading
import time

import pytest

from yolo_tensorrt_windows.processors.pipeline import run_inline, run_threaded

STAGES = [lambda x: x + 1, lambda x: x * 10, str]


@pytest.mark.parametrize("run", [run_inline, run_threaded])
def test_stages_applied_in_order(run):
    assert list(run(STAGES, range(100))) == [str((i + 1) * 10) for i in range(100)]


@pytest.mark.parametrize("run", [run_inline, run_threaded])
def test_empty_source(run):
    assert list(run(STAGES, [])) == []


def test_threaded_preserves_order_with_uneven_stage_speed():
    def jitter(x):
        time.sleep(0.001 * (x % 3))
        return x

    assert list(run_threaded([jitter, jitter], range(50), maxsize=2)) == list(range(50))


def _wait_for_threads(before: int, timeout: float = 2.0) -> int:
    deadline = time.monotonic() + timeout
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.01)
    return threading.active_count()


def test_threaded_stage_error_propagates_without_hanging():
    before = threading.active_count()

    def boom(x):
        if x == 5:
            raise ValueError("bad frame")
        return x

    with pytest.raises(ValueError, match="bad frame"):
        list(run_threaded([boom], range(1000), maxsize=2))
    assert _wait_for_threads(before) == before


def test_threaded_source_error_propagates():
    def source():
        yield 1
        raise OSError("decode failed")

    with pytest.raises(OSError, match="decode failed"):
        list(run_threaded(STAGES, source()))


def test_threaded_consumer_can_stop_early():
    before = threading.active_count()
    closed = threading.Event()

    def source():
        try:
            yield from range(10_000)
        finally:
            closed.set()

    it = run_threaded(STAGES, source(), maxsize=2)
    assert next(it) == "10"
    it.close()
    assert closed.wait(2)
    assert _wait_for_threads(before) == before
