"""YOLO-based hoof detector."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..components import Settings
from ..domain import BBox
from ..exceptions import DetectionError, ModelLoadError
from ..security import require_file, verify_sha256
from .base import DETECTORS, HoofDetector


@DETECTORS.register("yolo")
class YoloHoofDetector(HoofDetector):
    @dataclass(frozen=True)
    class Settings(Settings):
        weights: Path
        conf: float = 0.28
        device: str | None = None  # e.g. "cpu" or "cuda:0", None lets Ultralytics choose
        sha256: str | None = None  # optional integrity check

        def __post_init__(self) -> None:
            if not 0.0 <= self.conf <= 1.0:
                raise ValueError(f"conf must be between 0 and 1, got {self.conf}")

    def __init__(self, settings: "YoloHoofDetector.Settings") -> None:
        self.settings = settings
        require_file(settings.weights, "hoof detector weight file")
        verify_sha256(settings.weights, settings.sha256)
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(settings.weights))
        except Exception as exc:
            raise ModelLoadError(f"Cannot load the hoof detector ({settings.weights}): {exc}") from exc

    def detect(self, image: np.ndarray) -> list[BBox]:
        try:
            result = self._model.predict(image, conf=self.settings.conf, device=self.settings.device, verbose=False)[0]
            return [BBox.from_xyxy(b, c) for b, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
        except Exception as exc:
            raise DetectionError(f"Hoof detection error: {exc}") from exc
