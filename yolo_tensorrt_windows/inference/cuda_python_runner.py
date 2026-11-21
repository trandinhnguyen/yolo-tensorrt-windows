"""TensorRT runner for numpy inputs, using cuda-python for device memory."""

import math

import numpy as np
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart
from loguru import logger

from .engine import dtype_map, load_engine, tensor_specs

_NUMPY_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
    "uint8": np.uint8,
}


def _check_cuda_err(err) -> None:
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {err}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime error: {err}")
    else:
        raise RuntimeError(f"Unknown CUDA error type: {err}")


def _cuda_call(call):
    """Unwrap cuda-python's (err, *results) return convention."""
    err, *res = call
    _check_cuda_err(err)
    return res[0] if len(res) == 1 else res


class _Binding:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: np.dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = math.prod(shape) * dtype.itemsize
        self.device_ptr = _cuda_call(cudart.cudaMalloc(self.nbytes))


class CudaPythonRunner:
    """Run the engine on host numpy arrays; outputs come back as numpy arrays."""

    def __init__(self):
        engine = load_engine()
        to_numpy = dtype_map(_NUMPY_DTYPES)

        self.stream = _cuda_call(cudart.cudaStreamCreate())
        self.context = engine.create_execution_context()
        self.inputs: list[_Binding] = []
        self.outputs: list[_Binding] = []
        for spec in tensor_specs(engine):
            binding = _Binding(spec.name, spec.shape, np.dtype(to_numpy[spec.dtype]))
            (self.inputs if spec.is_input else self.outputs).append(binding)
            self.context.set_tensor_address(spec.name, binding.device_ptr)

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        for binding, inp in zip(self.inputs, inputs):
            if inp.shape != binding.shape:
                raise ValueError(
                    f"Shape mismatch for {binding.name}: "
                    f"input {inp.shape}, model {binding.shape}"
                )
            if inp.dtype != binding.dtype:
                logger.warning(
                    f"Dtype mismatch for {binding.name}: "
                    f"input {inp.dtype}, model {binding.dtype}. Converting."
                )
                inp = inp.astype(binding.dtype)

            _cuda_call(
                cudart.cudaMemcpyAsync(
                    binding.device_ptr,
                    np.ascontiguousarray(inp),
                    binding.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            )

        self.context.execute_async_v3(stream_handle=self.stream)

        outputs = []
        for binding in self.outputs:
            output = np.empty(binding.shape, binding.dtype)
            _cuda_call(
                cudart.cudaMemcpyAsync(
                    output,
                    binding.device_ptr,
                    binding.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )
            outputs.append(output)

        _cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return outputs

    def free(self) -> None:
        for binding in self.inputs + self.outputs:
            _cuda_call(cudart.cudaFree(binding.device_ptr))
        _cuda_call(cudart.cudaStreamDestroy(self.stream))
