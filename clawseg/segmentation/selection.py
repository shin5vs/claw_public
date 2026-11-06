"""Selects the two claw masks of a hoof from the raw masks of a segmenter.

Order: drop masks centred outside the hoof box or touching a cut ROI edge -> mask NMS ->
the two largest areas.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..domain import BBox, Mask

logger = logging.getLogger(__name__)

# (left, top, right, bottom): True if that ROI side is not the image border, i.e. the crop cut the content
CutSides = tuple[bool, bool, bool, bool]


def cut_sides(roi_bounds: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> CutSides:
    x1, y1, x2, y2 = roi_bounds
    h, w = image_shape[:2]
    return x1 > 0, y1 > 0, x2 < w, y2 < h


@dataclass(frozen=True)
class MaskSelectionSettings:
    max_claws: int = 2                 # number of masks returned (the largest)
    mask_iou: float = 0.5              # masks overlapping more than this are duplicates, 1.0 disables NMS
    drop_cut_edge: bool = True         # drop masks touching a cut ROI edge
    cut_edge_tolerance: int = 2        # px
    require_center_in_box: bool = True # the mask centre must lie inside the hoof box

    def __post_init__(self) -> None:
        if self.max_claws < 1:
            raise ValueError("max_claws must be at least 1")
        if not 0.0 < self.mask_iou <= 1.0:
            raise ValueError(f"mask_iou must be in (0, 1], got {self.mask_iou}")
        if self.cut_edge_tolerance < 0:
            raise ValueError("cut_edge_tolerance must not be negative")


class MaskSelector:
    def __init__(self, settings: MaskSelectionSettings | None = None) -> None:
        self.settings = settings or MaskSelectionSettings()

    def select(self, masks: Sequence[Mask], hoof_box: BBox, sides: CutSides = (False,) * 4) -> list[Mask]:
        """`masks` are in decreasing confidence of the segmenter, which NMS preserves.

        `hoof_box` is in ROI coordinates, `sides` is the result of `cut_sides()`.
        """
        return [mask for mask, _ in self.select_scored([(m, None) for m in masks], hoof_box, sides)]

    def select_scored(self, items: Sequence[tuple[Mask, float | None]], hoof_box: BBox,
                      sides: CutSides = (False,) * 4) -> list[tuple[Mask, float | None]]:
        """Like `select`, but each mask keeps its confidence."""
        s = self.settings
        masks = [m for m, _ in items]
        kept: list[tuple[Mask, float | None]] = []
        dropped = {"empty": 0, "cut_edge": 0, "center": 0, "duplicate": 0}
        for mask, conf in items:
            binary = mask > 0
            ys, xs = np.nonzero(binary)
            if xs.size == 0:
                dropped["empty"] += 1
                continue
            if s.drop_cut_edge and self._touches_cut_edge(xs, ys, binary.shape, sides):
                dropped["cut_edge"] += 1
                continue
            if s.require_center_in_box and not (
                hoof_box.x1 <= xs.mean() <= hoof_box.x2 and hoof_box.y1 <= ys.mean() <= hoof_box.y2
            ):
                dropped["center"] += 1
                continue
            if any(_mask_iou(binary, k > 0) > s.mask_iou for k, _ in kept):
                dropped["duplicate"] += 1
                continue
            kept.append((mask, conf))
        if any(dropped.values()):
            logger.debug("mask filtering: %d -> %d (dropped: %s)", len(masks), len(kept), dropped)
        kept.sort(key=lambda item: -int(np.count_nonzero(item[0])))
        return kept[: s.max_claws]

    def _touches_cut_edge(self, xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int], sides: CutSides) -> bool:
        h, w = shape
        tol = self.settings.cut_edge_tolerance
        left, top, right, bottom = sides
        return bool((left and xs.min() <= tol) or (top and ys.min() <= tol)
                    or (right and xs.max() >= w - 1 - tol) or (bottom and ys.max() >= h - 1 - tol))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0
