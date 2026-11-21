from dataclasses import dataclass

import numpy as np
import torch

Array = np.ndarray | torch.Tensor


@dataclass
class Detections:
    """Detections for one frame, in original-frame pixel coordinates.

    Arrays are numpy on the CPU pipelines and torch tensors (possibly on the
    GPU) on the GPU pipelines.
    """

    boxes: Array  # (N, 4) x1, y1, x2, y2
    scores: Array  # (N,)
    class_ids: Array  # (N,)
    tracker_ids: Array | None = None  # (N,), -1 where no track matched

    def __len__(self) -> int:
        return len(self.scores)
