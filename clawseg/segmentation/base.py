"""Claw segmenter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..components import Component, Registry
from ..domain import BBox, Mask


class ClawSegmenter(ABC, Component):
    """Produces claw masks from a hoof ROI."""

    @property
    @abstractmethod
    def default_roi_padding(self) -> float:
        """Recommended ROI margin for this model (depends on its training scale)."""

    @abstractmethod
    def segment(self, roi_image: np.ndarray, hoof_box: BBox) -> list[Mask]:
        """Candidate masks from the ROI (BGR) and the hoof box (in ROI coordinates).

        Masks are uint8 (0/255) of ROI size, in decreasing confidence, unfiltered.
        Filtering (cut edges, hoof box, NMS) and keeping the best two is done by `MaskSelector`.
        """

    def segment_scored(self, roi_image: np.ndarray, hoof_box: BBox) -> list[tuple[Mask, float | None]]:
        """Like `segment`, but with the model confidence of each mask (None if unknown).

        The default wraps `segment`. Segmenters that know their confidence override it.
        """
        return [(mask, None) for mask in self.segment(roi_image, hoof_box)]


SEGMENTERS: Registry[ClawSegmenter] = Registry("segmenter")
