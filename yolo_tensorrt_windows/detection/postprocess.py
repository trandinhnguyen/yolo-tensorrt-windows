"""Decode raw YOLOv8 output (4 box + C class rows per anchor) into Detections."""

import cv2
import numpy as np
import torch
import torchvision

from .. import config
from .detections import Detections
from .letterbox import Letterbox


def postprocess_cpu(
    output: np.ndarray,
    letterbox: Letterbox,
    confidence_thres: float = config.CONFIDENCE_THRESHOLD,
    iou_thres: float = config.IOU_THRESHOLD,
) -> Detections:
    output = np.transpose(output)  # (anchors, 4 + classes)

    boxes = output[:, :4]  # center x, center y, w, h
    scores = output[:, 4:].max(1)
    class_ids = output[:, 4:].argmax(1)

    mask = scores >= confidence_thres
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    top, left = letterbox.offset
    boxes[:, 0] -= left
    boxes[:, 1] -= top
    boxes /= letterbox.scale

    boxes[:, :2] -= boxes[:, 2:] / 2  # -> x1, y1, w, h (what NMSBoxes expects)
    keep = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), confidence_thres, iou_thres
    )
    keep = np.asarray(keep, dtype=np.int64).reshape(-1)

    boxes[:, 2:] += boxes[:, :2]  # -> x1, y1, x2, y2
    return Detections(boxes[keep], scores[keep], class_ids[keep])


def postprocess_gpu(
    output: torch.Tensor,
    letterbox: Letterbox,
    confidence_thres: float = config.CONFIDENCE_THRESHOLD,
    iou_thres: float = config.IOU_THRESHOLD,
) -> Detections:
    output = output.transpose(0, 1)  # (anchors, 4 + classes)

    boxes = output[:, :4]  # center x, center y, w, h
    scores, class_ids = output[:, 4:].max(1)

    mask = scores >= confidence_thres
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    top, left = letterbox.offset
    boxes[:, 0] -= left
    boxes[:, 1] -= top

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]  # -> x1, y1, x2, y2 (what nms expects)
    boxes /= letterbox.scale

    keep = torchvision.ops.nms(boxes, scores, iou_thres)
    return Detections(boxes[keep], scores[keep], class_ids[keep])
