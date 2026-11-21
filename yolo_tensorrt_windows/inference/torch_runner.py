"""TensorRT runner for CUDA torch tensors (zero-copy, async on a given stream)."""

import torch
import torch.cuda as cuda
from loguru import logger

from .. import config
from .engine import dtype_map, load_engine, tensor_specs

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "uint8": torch.uint8,
}


class TorchRunner:
    """Run the engine on CUDA tensors, enqueued on ``stream``.

    Outputs are freshly allocated each call, so they can safely be handed to
    the next pipeline stage while this one moves on.
    """

    def __init__(self, stream: cuda.Stream):
        engine = load_engine()
        to_torch = dtype_map(_TORCH_DTYPES)

        self.stream = stream
        self.context = engine.create_execution_context()
        self.inputs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        self.outputs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        for spec in tensor_specs(engine):
            meta = (spec.name, spec.shape, to_torch[spec.dtype])
            (self.inputs if spec.is_input else self.outputs).append(meta)

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        bound = []  # keeps converted/contiguous copies alive until enqueued
        for (name, shape, dtype), inp in zip(self.inputs, inputs):
            if tuple(inp.shape) != shape:
                raise ValueError(
                    f"Shape mismatch for {name}: input {tuple(inp.shape)}, model {shape}"
                )
            if inp.dtype != dtype:
                logger.warning(
                    f"Dtype mismatch for {name}: input {inp.dtype}, model {dtype}. "
                    "Converting."
                )
                inp = inp.to(dtype)
            inp = inp.contiguous()
            bound.append(inp)
            self.context.set_tensor_address(name, inp.data_ptr())

        outputs = []
        for name, shape, dtype in self.outputs:
            output = torch.empty(shape, dtype=dtype, device=config.GPU_DEVICE)
            self.context.set_tensor_address(name, output.data_ptr())
            outputs.append(output)

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        return outputs
