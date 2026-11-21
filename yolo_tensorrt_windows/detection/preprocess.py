import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from .letterbox import Letterbox


def preprocess_cpu(img: np.ndarray, letterbox: Letterbox) -> np.ndarray:
    """BGR uint8 HWC frame -> letterboxed RGB float32 NCHW in [0, 1]."""
    if tuple(img.shape[:2]) != letterbox.resized_hw:
        resized_h, resized_w = letterbox.resized_hw
        img = cv2.resize(img, (resized_w, resized_h))

    top, bottom, left, right = letterbox.pad
    img = np.pad(
        img,
        ((top, bottom), (left, right), (0, 0)),
        constant_values=config.PAD_VALUE,
    )

    img = img.astype(np.float32) / 255
    img = np.flip(img, 2)  # BGR -> RGB
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, 0)


def yuv_to_rgb(frames: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) YUV planes (0-255) -> RGB float32 in [0, 1]."""
    frames = frames.to(torch.float32)
    y = frames[..., 0, :, :] / 255
    u = frames[..., 1, :, :] / 255 - 0.5
    v = frames[..., 2, :, :] / 255 - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    return torch.stack([r, g, b], 1).clamp(0, 1)


def preprocess_gpu(frames: torch.Tensor, letterbox: Letterbox) -> torch.Tensor:
    """Decoder output (already resized to ``letterbox.resized_hw``) -> model input."""
    top, bottom, left, right = letterbox.pad
    frames = yuv_to_rgb(frames)
    return F.pad(frames, (left, right, top, bottom), value=config.PAD_VALUE / 255)
