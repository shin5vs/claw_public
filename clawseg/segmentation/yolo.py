"""YOLO-seg claw segmenter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..components import Settings
from ..domain import BBox, Mask
from ..exceptions import ModelLoadError, SegmentationError
from ..security import require_file, verify_sha256
from .base import SEGMENTERS, ClawSegmenter


@SEGMENTERS.register("yolo")
class YoloClawSegmenter(ClawSegmenter):
    @dataclass(frozen=True)
    class Settings(Settings):
        weights: Path
        conf: float = 0.25
        # The model was trained on hoof crops in which a claw covers about 16% of the image. On a tight ROI
        # the claw is too large to be found.
        roi_padding: float = 1.5
        device: str | None = None  # e.g. "cpu" or "cuda:0", None lets Ultralytics choose
        sha256: str | None = None

        def __post_init__(self) -> None:
            if not 0.0 <= self.conf <= 1.0:
                raise ValueError(f"conf must be between 0 and 1, got {self.conf}")
            if self.roi_padding < 0:
                raise ValueError("roi_padding must not be negative")

    def __init__(self, settings: "YoloClawSegmenter.Settings") -> None:
        self.settings = settings
        require_file(settings.weights, "YOLO claw segmenter weight file")
        verify_sha256(settings.weights, settings.sha256)
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(settings.weights))
        except Exception as exc:
            raise ModelLoadError(f"Cannot load the YOLO claw segmenter ({settings.weights}): {exc}") from exc

    @property
    def default_roi_padding(self) -> float:
        return self.settings.roi_padding

    def segment(self, roi_image: np.ndarray, hoof_box: BBox) -> list[Mask]:
        return [mask for mask, _ in self.segment_scored(roi_image, hoof_box)]

    def segment_scored(self, roi_image: np.ndarray, hoof_box: BBox) -> list[tuple[Mask, float | None]]:
        try:
            result = self._model.predict(roi_image, conf=self.settings.conf, device=self.settings.device, verbose=False)[0]
            if result.masks is None:
                return []
            h, w = roi_image.shape[:2]
            masks: list[tuple[Mask, float | None]] = []
            for raw, conf in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
                masks.append(((resized > 0.5).astype(np.uint8) * 255, float(conf)))
            return masks
        except Exception as exc:
            raise SegmentationError(f"YOLO segmentation error: {exc}") from exc
