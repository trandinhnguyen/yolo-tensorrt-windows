from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class Letterbox:
    """Resize-and-pad geometry that fits a frame into the square model input."""

    scale: float  # model pixels per frame pixel
    resized_hw: tuple[int, int]  # frame size after scaling, (height, width)
    pad: tuple[int, int, int, int]  # (top, bottom, left, right)

    @classmethod
    def fit(
        cls, width: int, height: int, size: int = config.MODEL_INPUT_SIZE
    ) -> "Letterbox":
        scale = size / max(width, height)
        resized_h, resized_w = int(round(height * scale)), int(round(width * scale))

        dh = (size - resized_h) / 2
        dw = (size - resized_w) / 2
        # The +-0.1 splits odd padding as top/left = floor, bottom/right = ceil.
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        return cls(scale, (resized_h, resized_w), (top, bottom, left, right))

    @property
    def offset(self) -> tuple[int, int]:
        """(top, left) padding: where the resized frame starts in the model input."""
        return self.pad[0], self.pad[2]
