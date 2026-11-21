"""Application entry point: start the worker processes and the desktop window.

Run with ``--benchmark VIDEO`` to process a video with every pipeline without
opening the window; the results go to the log.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

from loguru import logger

from . import paths
from .log import configure_logging
from .paths import setup_dll_search_paths

APP_NAME = "YOLO TensorRT for Windows"


def _pipelines() -> dict:
    # Imported lazily: these load TensorRT and FFmpeg, which needs the DLL
    # search paths to be set up first.
    from .processors.cpu import AsyncCPUVideoProcessor, SyncCPUVideoProcessor
    from .processors.gpu import AsyncGPUVideoProcessor, SyncGPUVideoProcessor

    return {
        "Sync CPU": SyncCPUVideoProcessor,
        "Async CPU": AsyncCPUVideoProcessor,
        "Sync GPU": SyncGPUVideoProcessor,
        "Async GPU": AsyncGPUVideoProcessor,
    }


def run_benchmark(video: str, pipelines: dict, log_dir: Path | None = None) -> int:
    """Run ``video`` through each pipeline in turn; 0 if all succeeded."""
    from .jobs import FINAL_STATUSES, WorkerPool

    pool = WorkerPool(pipelines, log_dir=log_dir)
    pool.start()
    jobs = []
    try:
        for name in pipelines:
            job = pool.submit(video, name)
            logger.info(f"Benchmark: {name}...")
            while job.status not in FINAL_STATUSES:
                time.sleep(0.1)
                pool.poll()
            jobs.append(job)
    finally:
        pool.stop()

    rows = [
        f"  {job.pipeline:10s} {job.status.value:7s} {job.throughput:8.1f} fps"
        + (f"  ({job.error})" if job.error else "")
        for job in jobs
    ]
    logger.info(f"Benchmark results for {video}:\n" + "\n".join(rows))
    return 0 if all(job.error is None for job in jobs) else 1


def _ensure_engine_with_progress(app) -> None:
    """Build the TensorRT engine if needed, with a progress dialog meanwhile."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QProgressDialog

    from .inference import engine

    if engine.engine_path().is_file():
        return
    dialog = QProgressDialog(
        "Building the TensorRT engine for this GPU.\n"
        "This happens only on the first start and takes a few minutes.",
        None,  # no cancel button
        0,
        0,  # busy indicator
    )
    dialog.setWindowTitle(APP_NAME)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.show()

    errors: list[BaseException] = []

    def build() -> None:
        try:
            engine.ensure_engine()
        except BaseException as e:
            errors.append(e)

    thread = threading.Thread(target=build, daemon=True)
    thread.start()
    while thread.is_alive():
        app.processEvents()
        thread.join(0.05)
    dialog.close()
    if errors:
        raise errors[0]


def run_gui(qt_args: list[str], log_file: Path) -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from .jobs import WorkerPool
    from .ui import MainWindow

    app = QApplication([sys.argv[0], *qt_args])
    try:
        pipelines = _pipelines()
        _ensure_engine_with_progress(app)
        pool = WorkerPool(pipelines, log_dir=paths.LOG_DIR)
        pool.start()
    except Exception as e:
        logger.exception("Startup failed")
        QMessageBox.critical(
            None,
            APP_NAME,
            f"The app could not start:\n\n{e}\n\nDetails are in the log:\n"
            f"{log_file.resolve()}",
        )
        return 1

    try:
        window = MainWindow(pool)
        window.show()
        logger.info("Window ready")
        return app.exec()
    finally:
        logger.info("Shutting down workers")
        pool.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolo-tensorrt-windows", description=APP_NAME)
    parser.add_argument(
        "--benchmark",
        metavar="VIDEO",
        help="process VIDEO with every pipeline without opening the window, "
        "log the throughput (see logs\\main.log) and exit",
    )
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    log_file = configure_logging("main", paths.LOG_DIR)
    logger.info(f"{APP_NAME} starting (frozen={paths.IS_FROZEN})")
    setup_dll_search_paths()

    if args.benchmark:
        try:
            from .inference.engine import ensure_engine

            pipelines = _pipelines()
            ensure_engine()
            return run_benchmark(args.benchmark, pipelines, log_dir=paths.LOG_DIR)
        except Exception:
            logger.exception("Benchmark failed")
            return 1
    return run_gui(qt_args, log_file)
