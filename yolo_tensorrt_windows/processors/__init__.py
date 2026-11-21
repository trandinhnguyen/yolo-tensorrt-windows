"""Video processing pipelines, each run in its own worker process.

The concrete processors live in ``cpu`` and ``gpu``; import them only after
``paths.setup_dll_search_paths()``, since they load TensorRT and FFmpeg DLLs.
"""
