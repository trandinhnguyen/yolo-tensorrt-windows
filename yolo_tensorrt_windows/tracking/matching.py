"""Track/detection association.

Ported to PyTorch from Roboflow supervision (MIT, Copyright (c) 2022 Roboflow),
itself based on ByteTrack (MIT, Copyright (c) 2021 Yifu Zhang).
See THIRD_PARTY_NOTICES.md.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .strack import STrack
from .utils import box_iou_batch

Matches = tuple[np.ndarray, list[int], list[int]]


def linear_assignment(cost_matrix: Tensor, thresh: float) -> Matches:
    """Optimal one-to-one matching with costs above ``thresh`` disallowed.

    Returns (matches as a (K, 2) array of row/col pairs, unmatched rows,
    unmatched cols).
    """
    n_rows, n_cols = cost_matrix.shape
    if cost_matrix.numel() == 0:
        return np.empty((0, 2), dtype=int), list(range(n_rows)), list(range(n_cols))

    cost = cost_matrix.detach().cpu().numpy().copy()  # scipy works on numpy
    cost[cost > thresh] = thresh + 1e-4
    rows, cols = linear_sum_assignment(cost)

    matched = cost[rows, cols] <= thresh
    matches = np.column_stack((rows[matched], cols[matched]))
    unmatched_rows = sorted(set(range(n_rows)) - set(matches[:, 0].tolist()))
    unmatched_cols = sorted(set(range(n_cols)) - set(matches[:, 1].tolist()))
    return matches, unmatched_rows, unmatched_cols


def iou_distance(atracks: list[STrack], btracks: list[STrack]) -> Tensor:
    """1 - IoU between every pair of tracks, shape (len(atracks), len(btracks))."""
    if not atracks or not btracks:
        return torch.zeros((len(atracks), len(btracks)), dtype=torch.float64)

    atlbrs = torch.stack([track.tlbr for track in atracks])
    btlbrs = torch.stack([track.tlbr for track in btracks])
    return 1 - box_iou_batch(atlbrs, btlbrs)


def fuse_score(cost_matrix: Tensor, detections: list[STrack]) -> Tensor:
    """Weight IoU similarity by detection confidence."""
    if cost_matrix.numel() == 0:
        return cost_matrix
    scores = cost_matrix.new_tensor([det.score for det in detections])
    return 1 - (1 - cost_matrix) * scores[None, :]
