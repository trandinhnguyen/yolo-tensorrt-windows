"""Filesystem layout and DLL search-path setup.

Works both from a source checkout and from a PyInstaller bundle, where data
files and DLLs are unpacked into ``sys._MEIPASS``.
"""

import os
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_FROZEN = hasattr(sys, "_MEIPASS")
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

MODELS_DIR = BUNDLE_DIR / "models"
OUTPUT_DIR = Path("output")  # relative to the working directory

# Third-party native libraries (source checkout only; bundled into BUNDLE_DIR
# by app.spec when frozen).
LIB_DIR = PROJECT_ROOT / "lib"
TRT_LIB_DIRS = (
    LIB_DIR / "TensorRT-8.5.3.1" / "lib",
    LIB_DIR / "TensorRT-10.0.1.6" / "lib",
)
NVIDIA_DIR = LIB_DIR / "nvidia"  # CUDA 11 cuBLAS/cuDNN/... DLLs (see scripts/setup.ps1)
FFMPEG_BIN_DIR = LIB_DIR / "ffmpeg" / "bin"


def output_path_for(input_path: str | os.PathLike, tag: str = "") -> Path:
    """Where the annotated copy of ``input_path`` is written.

    ``tag`` (e.g. the pipeline name) keeps outputs of different runs apart.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = "-".join(tag.lower().split())
    return OUTPUT_DIR / (Path(input_path).stem + (f"_{slug}" if slug else "") + ".mp4")


def setup_dll_search_paths() -> None:
    """Put the TensorRT DLLs and the CUDA libraries they need on PATH.

    The vendored ``tensorrt`` packages locate their DLLs through PATH; TensorRT
    8.5 refuses to import without cuBLAS/cuDNN. Call this in the main process
    before spawning workers; children inherit PATH.
    """
    if IS_FROZEN:
        dirs = (BUNDLE_DIR,)
    else:
        dirs = (*sorted(NVIDIA_DIR.glob("*/bin")), *TRT_LIB_DIRS)
    for d in dirs:  # each is prepended, so the last one ends up first
        os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


def add_ffmpeg_dll_directory() -> None:
    """Let torchaudio's FFmpeg extension find the FFmpeg DLLs (per process)."""
    ffmpeg_dir = BUNDLE_DIR if IS_FROZEN else FFMPEG_BIN_DIR
    if ffmpeg_dir.is_dir():
        os.add_dll_directory(str(ffmpeg_dir))
    else:
        logger.warning(
            f"FFmpeg DLLs not found at {ffmpeg_dir}; GPU pipelines will fail."
        )
