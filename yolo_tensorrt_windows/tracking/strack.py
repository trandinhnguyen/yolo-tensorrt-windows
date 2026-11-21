"""Single tracked object (STrack).

Ported to PyTorch from Roboflow supervision (MIT, Copyright (c) 2022 Roboflow),
itself based on ByteTrack (MIT, Copyright (c) 2021 Yifu Zhang).
See THIRD_PARTY_NOTICES.md.
"""

from enum import Enum

import torch
from torch import Tensor

from .kalman_filter import KalmanFilter
from .utils import IdCounter


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """A single tracked object. Box tensors are float64 on the tracker's device."""

    def __init__(
        self,
        tlwh: Tensor,
        score: float,
        minimum_consecutive_frames: int,
        shared_kalman: KalmanFilter,
        internal_id_counter: IdCounter,
        external_id_counter: IdCounter,
    ):
        self.state = TrackState.New
        self.is_activated = False
        self.start_frame = 0
        self.frame_id = 0

        self._tlwh = tlwh
        self.kalman_filter: KalmanFilter | None = None
        self.shared_kalman = shared_kalman
        self.mean: Tensor | None = None
        self.covariance: Tensor | None = None

        self.score = float(score)
        self.tracklet_len = 0
        self.minimum_consecutive_frames = minimum_consecutive_frames

        self.internal_id_counter = internal_id_counter
        self.external_id_counter = external_id_counter
        self.internal_track_id = IdCounter.NO_ID
        self.external_track_id = IdCounter.NO_ID

    @staticmethod
    def multi_predict(stracks: list["STrack"], shared_kalman: KalmanFilter) -> None:
        if not stracks:
            return

        multi_mean = torch.stack([st.mean for st in stracks])  # copies
        multi_covariance = torch.stack([st.covariance for st in stracks])
        not_tracked = [
            i for i, st in enumerate(stracks) if st.state != TrackState.Tracked
        ]
        multi_mean[not_tracked, 7] = 0  # freeze height velocity of lost tracks

        multi_mean, multi_covariance = shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for st, mean, cov in zip(stracks, multi_mean, multi_covariance):
            st.mean, st.covariance = mean, cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.internal_track_id = self.internal_id_counter.new_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
            if self.minimum_consecutive_frames == 1:
                self.external_track_id = self.external_id_counter.new_id()

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int) -> None:
        """Resume a lost track with a newly matched detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.score = new_track.score

    def update(self, new_track: "STrack", frame_id: int) -> None:
        """Update a tracked track with a newly matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        if self.tracklet_len == self.minimum_consecutive_frames:
            self.is_activated = True
            if self.external_track_id == IdCounter.NO_ID:
                self.external_track_id = self.external_id_counter.new_id()

        self.score = new_track.score

    @property
    def tlwh(self) -> Tensor:
        """Current box as (top-left x, top-left y, width, height)."""
        if self.mean is None:
            return self._tlwh.clone()
        ret = self.mean[:4].clone()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> Tensor:
        """Current box as (x1, y1, x2, y2)."""
        return self.tlwh_to_tlbr(self.tlwh)

    @staticmethod
    def tlwh_to_xyah(tlwh: Tensor) -> Tensor:
        """(top-left x, top-left y, w, h) -> (center x, center y, w / h, h)."""
        ret = tlwh.clone()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: Tensor) -> Tensor:
        ret = tlbr.clone()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: Tensor) -> Tensor:
        ret = tlwh.clone()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"OT_{self.internal_track_id}_({self.start_frame}-{self.frame_id})"
