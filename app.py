"""Entry point for ``python app.py`` and the PyInstaller build (app.spec).

Kept minimal on purpose: worker processes re-import this module, so heavy
imports stay inside the ``__main__`` guard.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    # Must run first so frozen worker processes don't start another UI.
    multiprocessing.freeze_support()

    from yolo_tensorrt_windows.app import main

    sys.exit(main())
