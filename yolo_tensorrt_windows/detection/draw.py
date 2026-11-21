import cv2
import numpy as np

from .classes import COCO_CLASSES
from .detections import Detections

# Fixed seed so every worker process (and every run) uses the same colours.
_PALETTE = [
    tuple(int(c) for c in rgb)
    for rgb in np.random.default_rng(0).uniform(0, 255, size=(len(COCO_CLASSES), 3))
]
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5


def _label(class_id: int, score: float, tracker_id: int | None) -> str:
    label = f"{COCO_CLASSES[class_id]} - {score:.2f}"
    if tracker_id is not None and tracker_id >= 0:
        label = f"ID {tracker_id} - {label}"
    return label


def draw_detections(img: np.ndarray, detections: Detections) -> None:
    """Draw boxes and labels onto ``img`` in place."""
    boxes = detections.boxes.tolist()
    scores = detections.scores.tolist()
    class_ids = detections.class_ids.tolist()
    tracker_ids = (
        detections.tracker_ids.tolist()
        if detections.tracker_ids is not None
        else [None] * len(boxes)
    )

    for box, score, class_id, tracker_id in zip(boxes, scores, class_ids, tracker_ids):
        x1, y1, x2, y2 = map(int, box)
        color = _PALETTE[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = _label(class_id, score, tracker_id)
        (label_w, label_h), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_h else y1 + 10

        cv2.rectangle(
            img,
            (label_x, label_y - label_h),
            (label_x + label_w, label_y + label_h),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            _FONT,
            _FONT_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
