import numpy as np
import pytest
import torch

from yolo_tensorrt_windows import config
from yolo_tensorrt_windows.detection import Detections, Letterbox
from yolo_tensorrt_windows.detection.draw import _label, draw_detections
from yolo_tensorrt_windows.detection.postprocess import postprocess_cpu, postprocess_gpu
from yolo_tensorrt_windows.detection.preprocess import preprocess_cpu, preprocess_gpu

SIZE = config.MODEL_INPUT_SIZE


@pytest.mark.parametrize(
    ("width", "height"), [(1920, 1080), (1080, 1920), (640, 640), (1000, 721), (7, 3)]
)
def test_letterbox_fills_model_input(width, height):
    lb = Letterbox.fit(width, height)
    top, bottom, left, right = lb.pad
    assert lb.resized_hw[0] + top + bottom == SIZE
    assert lb.resized_hw[1] + left + right == SIZE
    assert max(lb.resized_hw) == SIZE
    assert 0 <= bottom - top <= 1 and 0 <= right - left <= 1
    assert lb.offset == (top, left)


def test_letterbox_1080p():
    lb = Letterbox.fit(1920, 1080)
    assert lb.scale == pytest.approx(1 / 3)
    assert lb.resized_hw == (360, 640)
    assert lb.pad == (140, 140, 0, 0)


def test_preprocess_cpu_layout():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[..., 0] = 255  # blue in BGR
    lb = Letterbox.fit(160, 90)
    out = preprocess_cpu(frame, lb)

    assert out.shape == (1, 3, SIZE, SIZE) and out.dtype == np.float32
    top, _, left, _ = lb.pad
    assert out[0, 2, top + 1, left + 1] == 1.0  # blue ends up in the RGB B plane
    assert out[0, 0, top + 1, left + 1] == 0.0
    assert out[0, 0, 0, 0] == pytest.approx(config.PAD_VALUE / 255)


def test_preprocess_gpu_pads_and_converts_on_cpu_tensors():
    lb = Letterbox.fit(160, 90)
    h, w = lb.resized_hw
    yuv = torch.full((1, 3, h, w), 128, dtype=torch.uint8)  # mid grey
    out = preprocess_gpu(yuv, lb)

    assert out.shape == (1, 3, SIZE, SIZE)
    top, _, left, _ = lb.pad
    assert torch.allclose(
        out[0, :, top + 1, left + 1],
        torch.tensor(128 / 255),
        atol=1e-2,  # U/V=128 is ~neutral
    )
    assert out[0, 0, 0, 0].item() == pytest.approx(config.PAD_VALUE / 255)
    assert yuv.dtype == torch.uint8 and int(yuv[0, 0, 0, 0]) == 128  # input untouched


def make_raw_output(boxes_cxcywh, class_scores, n_anchors=50, n_classes=80):
    """Fake YOLOv8 output (4 + classes, anchors) with the given anchors filled."""
    raw = np.zeros((4 + n_classes, n_anchors), dtype=np.float32)
    for i, (box, scores) in enumerate(zip(boxes_cxcywh, class_scores)):
        raw[:4, i] = box
        for class_id, score in scores.items():
            raw[4 + class_id, i] = score
    return raw


@pytest.fixture
def raw_and_letterbox():
    lb = Letterbox.fit(1920, 1080)  # scale 1/3, top pad 140
    raw = make_raw_output(
        boxes_cxcywh=[
            (320, 320, 100, 50),  # car, kept
            (322, 321, 100, 50),  # overlapping duplicate, lower score -> NMS
            (100, 200, 40, 80),  # person, kept
            (500, 300, 30, 30),  # below confidence threshold
        ],
        class_scores=[{2: 0.9}, {2: 0.8}, {0: 0.7, 2: 0.1}, {5: 0.3}],
    )
    return raw, lb


def test_postprocess_cpu_maps_to_frame_coords(raw_and_letterbox):
    raw, lb = raw_and_letterbox
    det = postprocess_cpu(raw.copy(), lb)

    assert len(det) == 2
    order = np.argsort(-det.scores)
    np.testing.assert_allclose(det.scores[order], [0.9, 0.7], rtol=1e-6)
    np.testing.assert_array_equal(det.class_ids[order], [2, 0])
    # car: center (320, 320 - 140) * 3 = (960, 540); size (300, 150)
    np.testing.assert_allclose(det.boxes[order][0], [810, 465, 1110, 615], rtol=1e-5)


def test_postprocess_gpu_matches_cpu(raw_and_letterbox):
    raw, lb = raw_and_letterbox
    cpu = postprocess_cpu(raw.copy(), lb)
    gpu = postprocess_gpu(torch.from_numpy(raw.copy()), lb)

    cpu_order, gpu_order = np.argsort(-cpu.scores), torch.argsort(-gpu.scores)
    np.testing.assert_allclose(
        gpu.boxes[gpu_order].numpy(), cpu.boxes[cpu_order], rtol=1e-5
    )
    np.testing.assert_array_equal(
        gpu.class_ids[gpu_order].numpy(), cpu.class_ids[cpu_order]
    )


def test_postprocess_cpu_no_detections():
    lb = Letterbox.fit(640, 640)
    det = postprocess_cpu(make_raw_output([], []), lb)
    assert len(det) == 0 and det.boxes.shape == (0, 4)


def test_label_shows_tracker_id_only_when_tracked():
    assert _label(0, 0.876, 3) == "ID 3 - person - 0.88"
    assert _label(2, 0.5, -1) == "car - 0.50"
    assert _label(2, 0.5, None) == "car - 0.50"


@pytest.mark.parametrize("as_tensor", [False, True])
def test_draw_detections_draws(as_tensor):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = np.array([[10, 20, 60, 80]], dtype=np.float32)
    det = Detections(boxes, np.array([0.9]), np.array([2]), np.array([7]))
    if as_tensor:
        det = Detections(
            *(torch.as_tensor(a) for a in (det.boxes, det.scores, det.class_ids))
        )
    draw_detections(img, det)
    assert img.any()
