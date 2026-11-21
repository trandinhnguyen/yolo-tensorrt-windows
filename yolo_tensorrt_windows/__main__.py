"""``python -m yolo_tensorrt_windows``"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from yolo_tensorrt_windows.app import main

    sys.exit(main())
