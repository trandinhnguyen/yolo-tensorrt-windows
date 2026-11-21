import numpy as np
import pytest
import torch

from yolo_tensorrt_windows.tracking import ByteTrack
from yolo_tensorrt_windows.tracking.kalman_filter import KalmanFilter
from yolo_tensorrt_windows.tracking.matching import linear_assignment
from yolo_tensorrt_windows.tracking.utils import IdCounter, box_iou_batch


def test_box_iou_batch():
    a = torch.tensor([[0.0, 0, 10, 10], [0, 0, 0, 0]])
    b = torch.tensor([[0.0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    expected = torch.tensor([[1, 1 / 3, 0], [0, 0, 0]])
    assert torch.allclose(box_iou_batch(a, b), expected)
    assert box_iou_batch(a, b[:0]).shape == (2, 0)


def test_id_counter():
    counter = IdCounter(start_id=1)
    assert [counter.new_id(), counter.new_id()] == [1, 2]
    counter.reset()
    assert counter.new_id() == 1
    with pytest.raises(ValueError):
        IdCounter(start_id=-1)


def test_linear_assignment_respects_threshold():
    cost = torch.tensor([[0.1, 0.9], [0.8, 0.95], [0.2, 0.3]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, thresh=0.5)
    assert sorted(map(tuple, matches.tolist())) == [(0, 0), (2, 1)]
    assert unmatched_rows == [1] and unmatched_cols == []

    matches, rows, cols = linear_assignment(torch.zeros((0, 3)), 0.5)
    assert matches.shape == (0, 2) and rows == [] and cols == [0, 1, 2]


def test_kalman_filter_follows_constant_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(torch.tensor([100.0, 50, 0.5, 40], dtype=torch.float64))
    for step in range(1, 30):
        mean, cov = kf.multi_predict(mean[None], cov[None])
        mean, cov = mean[0], cov[0]
        z = torch.tensor([100.0 + 3 * step, 50, 0.5, 40], dtype=torch.float64)
        mean, cov = kf.update(mean, cov, z)

    assert mean[4].item() == pytest.approx(3, abs=0.1)  # learned x velocity
    assert torch.allclose(cov, cov.T)


def run_tracker(tracker, frames):
    """frames: list of (boxes, scores) -> list of tracker id lists."""
    return [
        tracker.update_with_detections(
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(scores, dtype=torch.float32),
        ).tolist()
        for boxes, scores in frames
    ]


def box(x, y, w=40, h=80):
    return [x, y, x + w, y + h]


def test_two_moving_objects_keep_their_ids():
    frames = [
        ([box(10 + 4 * t, 10), box(300 - 4 * t, 200)], [0.9, 0.8]) for t in range(20)
    ]
    ids = run_tracker(ByteTrack(), frames)
    assert ids[0] == [1, 2]
    assert all(frame_ids == [1, 2] for frame_ids in ids)


def test_detection_order_does_not_change_ids():
    frames = [([box(10 + 4 * t, 10), box(300, 200)], [0.9, 0.8]) for t in range(5)]
    frames.append(([box(300, 200), box(30, 10)], [0.8, 0.9]))
    ids = run_tracker(ByteTrack(), frames)
    assert ids[-1] == [2, 1]


def test_lost_object_recovers_id_within_buffer():
    visible = [([box(10 + 4 * t, 10)], [0.9]) for t in range(10)]
    gap = [([], [])] * 5
    back = [([box(10 + 4 * t, 10)], [0.9]) for t in range(15, 20)]
    ids = run_tracker(ByteTrack(), visible + gap + back)
    assert ids[9] == [1] and ids[10:15] == [[]] * 5
    assert ids[15:] == [[1]] * 5


def test_lost_object_gets_new_id_after_buffer():
    tracker = ByteTrack(lost_track_buffer=3)
    ids = run_tracker(
        tracker,
        [([box(10, 10)], [0.9])] * 3 + [([], [])] * 10 + [([box(10, 10)], [0.9])] * 3,
    )
    assert ids[0] == [1] and ids[-1] == [2]


def test_low_score_detection_continues_track_but_cannot_start_one():
    frames = [([box(10 + 4 * t, 10)], [0.9]) for t in range(5)]
    frames += [([box(30, 10), box(400, 300)], [0.2, 0.2])]  # second association
    ids = run_tracker(ByteTrack(), frames)
    assert ids[-1] == [1, -1]


def test_new_object_gets_new_id():
    frames = [([box(10, 10)], [0.9])] * 3 + [
        ([box(10, 10), box(300, 300)], [0.9, 0.9])
    ] * 3
    ids = run_tracker(ByteTrack(), frames)
    assert ids[-1] == [1, 2]


def test_reset_restarts_ids():
    tracker = ByteTrack()
    run_tracker(tracker, [([box(10, 10)], [0.9])] * 3)
    tracker.reset()
    assert run_tracker(tracker, [([box(300, 300)], [0.9])]) == [[1]]


def test_accepts_numpy_backed_tensors_without_mutating_them():
    boxes = np.array([box(10, 10)], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    tracker = ByteTrack()
    for _ in range(3):
        ids = tracker.update_with_detections(
            torch.as_tensor(boxes), torch.as_tensor(scores)
        )
    assert ids.dtype == torch.int64 and ids.tolist() == [1]
    np.testing.assert_array_equal(boxes, [box(10, 10)])
