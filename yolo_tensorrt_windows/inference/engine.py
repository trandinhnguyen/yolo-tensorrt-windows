"""TensorRT version selection, engine building and loading.

Two TensorRT builds are vendored: 8.5 for Pascal-and-older GPUs and 10.0 for
newer ones. The right one is chosen from the GPU's compute capability the first
time :func:`tensorrt` is called.
"""

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

from .. import config, paths


@lru_cache(maxsize=1)
def gpu_compute_capability_major() -> int:
    """Major compute capability of the first GPU, e.g. 8 for an RTX 30xx."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        first_gpu = result.stdout.strip().splitlines()[0]  # e.g. "8.6"
        return int(first_gpu.split(".")[0])
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        IndexError,
        ValueError,
    ) as e:
        raise RuntimeError(
            "Could not read the GPU compute capability from nvidia-smi. "
            "Is the NVIDIA driver installed and nvidia-smi on PATH?"
        ) from e


def uses_trt8() -> bool:
    return gpu_compute_capability_major() <= config.TRT8_MAX_SM_MAJOR


@lru_cache(maxsize=1)
def tensorrt():
    """The vendored ``tensorrt`` module matching this GPU."""
    if uses_trt8():
        from . import tensorrt_853 as trt
    else:
        from . import tensorrt_1001 as trt
    return trt


def engine_path() -> Path:
    name = config.TRT8_ENGINE_FILE if uses_trt8() else config.TRT10_ENGINE_FILE
    return paths.MODELS_DIR / name


def build_engine() -> None:
    """Build a serialized engine from the ONNX model and save it to engine_path()."""
    trt = tensorrt()
    onnx_path = paths.MODELS_DIR / config.ONNX_MODEL_FILE
    out_path = engine_path()

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, config.TRT_WORKSPACE_BYTES
    )
    if builder.platform_has_fast_fp16:
        builder_config.set_flag(trt.BuilderFlag.FP16)

    logger.info(f"Building TensorRT engine from {onnx_path}...")
    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build an engine from {onnx_path}")

    # Write-then-rename so concurrent workers never read a half-written file.
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    tmp_path.write_bytes(serialized)
    os.replace(tmp_path, out_path)
    logger.info(f"Saved TensorRT engine: {out_path}")


def ensure_engine() -> None:
    """Build the engine if it doesn't exist yet."""
    if not engine_path().is_file():
        build_engine()


@lru_cache(maxsize=1)
def _runtime():
    # One runtime per process, kept alive for as long as its engines.
    trt = tensorrt()
    return trt.Runtime(trt.Logger(trt.Logger.WARNING))


def load_engine():
    """Deserialize the engine, rebuilding it if it was built for another TRT/GPU."""
    ensure_engine()
    engine = _runtime().deserialize_cuda_engine(engine_path().read_bytes())
    if engine is None:
        logger.warning("TensorRT engine is incompatible with this setup. Rebuilding.")
        build_engine()
        engine = _runtime().deserialize_cuda_engine(engine_path().read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not load TensorRT engine {engine_path()}")
    return engine


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: Any  # trt.DataType
    is_input: bool


def tensor_specs(engine) -> list[TensorSpec]:
    """Name, shape, dtype and direction of each engine I/O tensor, in order."""
    trt = tensorrt()
    specs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        specs.append(
            TensorSpec(
                name=name,
                shape=tuple(engine.get_tensor_shape(name)),
                dtype=engine.get_tensor_dtype(name),
                is_input=engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
            )
        )
    return specs


def dtype_map(by_name: dict[str, Any]) -> dict[Any, Any]:
    """Map ``trt.DataType`` -> framework dtype for the types this TRT build has.

    TensorRT 8.5 has no ``int64``, so names missing from the module are skipped.
    """
    trt = tensorrt()
    return {
        getattr(trt, name): dt for name, dt in by_name.items() if hasattr(trt, name)
    }
