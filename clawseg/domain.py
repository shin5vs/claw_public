"""Data types shared by the components: bounding box, region of interest and binary mask."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# binary mask, uint8 with values 0/255, H x W
Mask = np.ndarray


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates, (x1, y1) to (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None   # detector confidence, if known

    @classmethod
    def from_xyxy(cls, values, conf: float | None = None) -> "BBox":
        x1, y1, x2, y2 = (float(v) for v in values)
        return cls(x1, y1, x2, y2, None if conf is None else float(conf))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    def shifted(self, dx: float, dy: float) -> "BBox":
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy, self.conf)

    def split_halves(self) -> tuple["BBox", "BBox"]:
        """Splits the box at its vertical midline into a left and a right half."""
        cx = self.center_x
        return BBox(self.x1, self.y1, cx, self.y2, self.conf), BBox(cx, self.y1, self.x2, self.y2, self.conf)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class Roi:
    """Cropped image region and its pixel bounds in the source image (integer coordinates)."""

    image: np.ndarray
    bounds: tuple[int, int, int, int]  # x1, y1, x2, y2 in the source image

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def origin(self) -> tuple[int, int]:
        return self.bounds[0], self.bounds[1]
