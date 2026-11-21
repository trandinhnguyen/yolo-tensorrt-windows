"""Model-agnostic detection helpers: geometry, pre/post-processing, drawing."""

from .detections import Detections
from .letterbox import Letterbox

__all__ = ["Detections", "Letterbox"]
