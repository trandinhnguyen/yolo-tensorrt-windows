"""Application entry point: start the worker processes and the desktop window."""

import sys

from loguru import logger

from .paths import setup_dll_search_paths


def main() -> int:
    setup_dll_search_paths()

    # Imported after the DLL paths are set: these load TensorRT and FFmpeg.
    from PySide6.QtWidgets import QApplication

    from .inference.engine import ensure_engine
    from .jobs import WorkerPool
    from .processors.cpu import AsyncCPUVideoProcessor, SyncCPUVideoProcessor
    from .processors.gpu import AsyncGPUVideoProcessor, SyncGPUVideoProcessor
    from .ui import MainWindow

    ensure_engine()  # build once here instead of racing in every worker

    pool = WorkerPool(
        {
            "Sync CPU": SyncCPUVideoProcessor,
            "Async CPU": AsyncCPUVideoProcessor,
            "Sync GPU": SyncGPUVideoProcessor,
            "Async GPU": AsyncGPUVideoProcessor,
        }
    )
    pool.start()
    try:
        app = QApplication(sys.argv)
        window = MainWindow(pool)
        window.show()
        return app.exec()
    finally:
        logger.info("Shutting down workers.")
        pool.stop()
