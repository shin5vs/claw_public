"""Hoof detector interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..components import Component, Registry
from ..domain import BBox


class HoofDetector(ABC, Component):
    """Finds the hooves in an image."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[BBox]:
        """BGR image -> bounding boxes of the detected hooves."""


DETECTORS: Registry[HoofDetector] = Registry("detektor")
