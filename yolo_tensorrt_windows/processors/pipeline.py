"""Run frames through a chain of stages, either inline or one thread per stage."""

import queue
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from .. import config

Stage = Callable[[Any], Any]

_END = object()  # end-of-stream marker passed down the queues
_POLL_SECONDS = 0.1


def run_inline(stages: Sequence[Stage], source: Iterable) -> Iterator:
    """Apply every stage to each item in turn, on the calling thread."""
    for item in source:
        for stage in stages:
            item = stage(item)
        yield item


def run_threaded(
    stages: Sequence[Stage],
    source: Iterable,
    maxsize: int = config.PIPELINE_QUEUE_SIZE,
) -> Iterator:
    """Like :func:`run_inline`, but reading the source and each stage run in
    their own thread, connected by bounded queues. Output order is preserved.

    If the source or a stage raises, all threads stop and the exception is
    re-raised to the consumer.
    """
    queues = [queue.Queue(maxsize) for _ in range(len(stages) + 1)]
    stop = threading.Event()
    errors: list[BaseException] = []

    def put(q: queue.Queue, item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                pass
        return False

    def get(q: queue.Queue):
        while not stop.is_set():
            try:
                return q.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                pass
        return _END

    def fail(e: BaseException) -> None:
        errors.append(e)
        stop.set()

    def feed() -> None:
        it = iter(source)
        try:
            for item in it:
                if not put(queues[0], item):
                    return
            put(queues[0], _END)
        except BaseException as e:
            fail(e)
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()

    def work(stage: Stage, q_in: queue.Queue, q_out: queue.Queue) -> None:
        try:
            while (item := get(q_in)) is not _END:
                if not put(q_out, stage(item)):
                    return
            put(q_out, _END)
        except BaseException as e:
            fail(e)

    threads = [threading.Thread(target=feed, daemon=True)] + [
        threading.Thread(
            target=work, args=(stage, queues[i], queues[i + 1]), daemon=True
        )
        for i, stage in enumerate(stages)
    ]
    for t in threads:
        t.start()

    try:
        while (item := get(queues[-1])) is not _END:
            yield item
    finally:
        stop.set()  # unblock the workers if the consumer stopped early
        for t in threads:
            t.join()

    if errors:
        raise errors[0]
