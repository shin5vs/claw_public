"""Cropping the region of interest (ROI) around a hoof box."""
from __future__ import annotations

import numpy as np

from .domain import BBox, Roi


def crop_expanded_roi(image: np.ndarray, bbox: BBox, pad_ratio: float) -> Roi:
    """Enlarges the box by `pad_ratio` times its size on each side and clips it to the image."""
    h_img, w_img = image.shape[:2]

    pad_x = int(bbox.width * pad_ratio)
    pad_y = int(bbox.height * pad_ratio)

    nx1 = max(0, int(bbox.x1) - pad_x)
    ny1 = max(0, int(bbox.y1) - pad_y)
    nx2 = min(w_img, int(bbox.x2) + pad_x)
    ny2 = min(h_img, int(bbox.y2) + pad_y)

    return Roi(image=np.ascontiguousarray(image[ny1:ny2, nx1:nx2]), bounds=(nx1, ny1, nx2, ny2))
