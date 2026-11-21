"""Kalman filter for box tracking.

Ported to PyTorch from Roboflow supervision (MIT, Copyright (c) 2022 Roboflow),
itself based on ByteTrack (MIT, Copyright (c) 2021 Yifu Zhang).
The filter follows deep_sort by Nicolai Wojke (GPL-3.0).
See THIRD_PARTY_NOTICES.md.
"""

import torch
from torch import Tensor


class KalmanFilter:
    """Constant-velocity Kalman filter for bounding boxes in image space.

    The 8-dimensional state is (x, y, a, h, vx, vy, va, vh): box center, aspect
    ratio (w / h), height, and their velocities. (x, y, a, h) is observed
    directly (linear observation model).
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        ndim, dt = 4, 1.0
        opts = {"dtype": torch.float64, "device": device}

        self._motion_mat = torch.eye(2 * ndim, **opts)
        self._motion_mat[:ndim, ndim:] += dt * torch.eye(ndim, **opts)
        self._update_mat = torch.eye(ndim, 2 * ndim, **opts)

        # Noise std devs are proportional to box height (state index 3), except
        # for the aspect ratio, which gets a small constant: std = h * scale + const.
        pos, vel = 1.0 / 20, 1.0 / 160
        self._init_scale = torch.tensor([2 * pos] * 2 + [0, 2 * pos] + [10 * vel] * 2 + [0, 10 * vel], **opts)  # fmt: skip
        self._init_const = torch.tensor([0, 0, 1e-2, 0, 0, 0, 1e-5, 0], **opts)
        self._motion_scale = torch.tensor([pos, pos, 0, pos, vel, vel, 0, vel], **opts)
        self._motion_const = torch.tensor([0, 0, 1e-2, 0, 0, 0, 1e-5, 0], **opts)
        self._innovation_scale = torch.tensor([pos, pos, 0, pos], **opts)
        self._innovation_const = torch.tensor([0, 0, 1e-1, 0], **opts)

    def initiate(self, measurement: Tensor) -> tuple[Tensor, Tensor]:
        """Mean (8,) and covariance (8, 8) of a new track from an (x, y, a, h) box."""
        mean = torch.cat([measurement, torch.zeros_like(measurement)])
        std = measurement[3] * self._init_scale + self._init_const
        return mean, torch.diag(std.square())

    def multi_predict(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        """Prediction step for N tracks: mean (N, 8), covariance (N, 8, 8)."""
        std = mean[:, 3:4] * self._motion_scale + self._motion_const
        motion_cov = torch.diag_embed(std.square())

        mean = mean @ self._motion_mat.T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        """Project a state distribution into (x, y, a, h) measurement space."""
        std = mean[3] * self._innovation_scale + self._innovation_const
        innovation_cov = torch.diag(std.square())

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(
        self, mean: Tensor, covariance: Tensor, measurement: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Correction step with an (x, y, a, h) measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol = torch.linalg.cholesky(projected_cov)
        kalman_gain = torch.cholesky_solve((covariance @ self._update_mat.T).T, chol).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance
