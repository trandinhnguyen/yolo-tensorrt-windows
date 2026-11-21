"""ByteTrack multi-object tracker.

Ported to PyTorch from Roboflow supervision (MIT, Copyright (c) 2022 Roboflow),
itself based on ByteTrack (MIT, Copyright (c) 2021 Yifu Zhang).
See THIRD_PARTY_NOTICES.md.
"""

import torch
from torch import Tensor

from .. import config
from . import matching
from .kalman_filter import KalmanFilter
from .strack import STrack, TrackState
from .utils import IdCounter, box_iou_batch


class ByteTrack:
    def __init__(
        self,
        track_activation_threshold: float = config.TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = config.LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = config.MINIMUM_MATCHING_THRESHOLD,
        frame_rate: float = 30,
        minimum_consecutive_frames: int = config.MINIMUM_CONSECUTIVE_FRAMES,
        device: torch.device | str = "cpu",
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold
        self.det_thresh = track_activation_threshold + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.device = torch.device(device)

        self.kalman_filter = KalmanFilter(self.device)
        self.shared_kalman = KalmanFilter(self.device)

        # Internal IDs must start at 0 and external IDs at 1; starting both at 1
        # links traces across different objects.
        self.internal_id_counter = IdCounter()
        self.external_id_counter = IdCounter(start_id=1)
        self.reset()

    def reset(self) -> None:
        """Clear all tracks, e.g. before starting a new video."""
        self.frame_id = 0
        self.internal_id_counter.reset()
        self.external_id_counter.reset()
        self.tracked_tracks: list[STrack] = []
        self.lost_tracks: list[STrack] = []
        self.removed_tracks: list[STrack] = []

    def update_with_detections(self, xyxy: Tensor, confidence: Tensor) -> Tensor:
        """Advance one frame.

        Args:
            xyxy: (N, 4) detection boxes, any device/dtype.
            confidence: (N,) detection scores.

        Returns:
            (N,) int64 tracker ID per detection on the tracker's device, -1 where
            the detection is not (yet) part of a confirmed track.
        """
        xyxy = xyxy.to(self.device, torch.float64)
        confidence = confidence.to(self.device, torch.float64)
        tracks = self._update(xyxy, confidence)

        tracker_ids = torch.full(
            (len(xyxy),), -1, dtype=torch.int64, device=self.device
        )
        if not tracks:
            return tracker_ids

        track_boxes = torch.stack([track.tlbr for track in tracks])
        iou_costs = 1 - box_iou_batch(xyxy, track_boxes)
        matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
        for i_detection, i_track in matches:
            tracker_ids[i_detection] = tracks[i_track].external_track_id
        return tracker_ids

    def _new_tracks(self, boxes: Tensor, scores: Tensor) -> list[STrack]:
        return [
            STrack(
                STrack.tlbr_to_tlwh(box),
                score,
                self.minimum_consecutive_frames,
                self.shared_kalman,
                self.internal_id_counter,
                self.external_id_counter,
            )
            for box, score in zip(boxes, scores.tolist())
        ]

    def _update(self, bboxes: Tensor, scores: Tensor) -> list[STrack]:
        """Run one ByteTrack step and return the active tracks."""
        self.frame_id += 1
        activated_stracks: list[STrack] = []
        refind_stracks: list[STrack] = []
        lost_stracks: list[STrack] = []
        removed_stracks: list[STrack] = []

        high = scores > self.track_activation_threshold
        low = (scores > 0.1) & (scores < self.track_activation_threshold)
        detections = self._new_tracks(bboxes[high], scores[high])
        detections_second = self._new_tracks(bboxes[low], scores[low])

        # Step 1: split existing tracks into confirmed and unconfirmed.
        unconfirmed: list[STrack] = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_tracks:
            if track.is_activated:
                tracked_stracks.append(track)
            else:
                unconfirmed.append(track)

        # Step 2: first association, high-score detections vs. all tracks.
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        STrack.multi_predict(strack_pool, self.shared_kalman)
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )
        for itracked, idet in matches:
            track, det = strack_pool[itracked], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # Step 3: second association, low-score detections vs. remaining
        # tracked tracks.
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track, det = r_tracked_stracks[itracked], detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        # Unconfirmed tracks (usually seen in a single frame) vs. leftover
        # high-score detections.
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        # Step 4: start new tracks from unmatched confident detections.
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: drop tracks lost for too long, then update the track lists.
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_stracks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        return [track for track in self.tracked_tracks if track.is_activated]


def joint_tracks(
    track_list_a: list[STrack], track_list_b: list[STrack]
) -> list[STrack]:
    """Concatenate two track lists, dropping repeated internal IDs."""
    seen_track_ids = set()
    result = []
    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)
    return result


def sub_tracks(track_list_a: list[STrack], track_list_b: list[STrack]) -> list[STrack]:
    """Tracks of ``track_list_a`` whose internal ID is not in ``track_list_b``."""
    track_ids_b = {track.internal_track_id for track in track_list_b}
    return [t for t in track_list_a if t.internal_track_id not in track_ids_b]


def remove_duplicate_tracks(
    tracks_a: list[STrack], tracks_b: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    """Of each near-identical (IoU > 0.95) pair, keep the longer-lived track."""
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    duplicates_a, duplicates_b = set(), set()
    for index_a, index_b in (pairwise_distance < 0.05).nonzero().tolist():
        time_a = tracks_a[index_a].frame_id - tracks_a[index_a].start_frame
        time_b = tracks_b[index_b].frame_id - tracks_b[index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(index_b)
        else:
            duplicates_a.add(index_a)

    result_a = [t for i, t in enumerate(tracks_a) if i not in duplicates_a]
    result_b = [t for i, t in enumerate(tracks_b) if i not in duplicates_b]
    return result_a, result_b
