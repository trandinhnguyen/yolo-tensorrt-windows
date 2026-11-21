"""Tracker helpers: ID counters and box IoU.

Ported to PyTorch from Roboflow supervision (MIT, Copyright (c) 2022 Roboflow),
itself based on ByteTrack (MIT, Copyright (c) 2021 Yifu Zhang).
See THIRD_PARTY_NOTICES.md.
"""

import torch
from torch import Tensor


class IdCounter:
    NO_ID = -1

    def __init__(self, start_id: int = 0):
        if start_id <= self.NO_ID:
            raise ValueError(f"start_id must be greater than {self.NO_ID}")
        self.start_id = start_id
        self.reset()

    def reset(self) -> None:
        self._id = self.start_id

    def new_id(self) -> int:
        """Return the next ID and advance the counter."""
        returned_id = self._id
        self._id += 1
        return returned_id


def box_iou_batch(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise IoU of (N, 4) and (M, 4) x1y1x2y2 boxes -> (N, M).

    Pairs whose union is empty get IoU 0.
    """
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return torch.where(union > 0, inter / union, torch.zeros_like(inter))
