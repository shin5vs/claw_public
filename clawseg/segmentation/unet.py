"""U-Net claw segmenter: background / claw / gap per pixel of the hoof ROI, claw map split into two claws.

The gap is a class of its own, so the two claws are separate regions even where they touch. The split uses
`split_claws` (distance-transform watershed). Training: `scripts/train_unet.py` and `scripts/train_variants.py`.

"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..components import Settings
from ..domain import BBox, Mask
from ..exceptions import ModelLoadError, SegmentationError
from ..security import require_file, verify_sha256
from .base import SEGMENTERS, ClawSegmenter
from .pixelmap import split_claws
from .unet_data import CLAW

logger = logging.getLogger(__name__)


@SEGMENTERS.register("unet")
class UnetClawSegmenter(ClawSegmenter):
    @dataclass(frozen=True)
    class Settings(Settings):
        weights: Path
        sha256: str | None = None
        roi_padding: float = 0.5           # same margin as the training ROIs (unet_data.hoof_rois)
        box_margin: float = 0.1            # claw pixels outside the hoof box enlarged by this ratio are removed
        min_claw_ratio: float = 0.03       # regions smaller than this share of the box area are discarded
        min_claw_height_px: float = 0.0    # claws lower than this are discarded (0 in the paper)
        tta: bool = False                  # average with the mirrored ROI (not used in the paper)
        device: str | None = None

        def __post_init__(self) -> None:
            if min(self.roi_padding, self.box_margin, self.min_claw_height_px) < 0:
                raise ValueError("roi_padding, box_margin and min_claw_height_px must not be negative")
            if not 0.0 <= self.min_claw_ratio < 0.5:
                raise ValueError("min_claw_ratio must be between 0 and 0.5")

    def __init__(self, settings: "UnetClawSegmenter.Settings") -> None:
        self.settings = settings
        require_file(settings.weights, "U-Net claw segmenter weight file")
        verify_sha256(settings.weights, settings.sha256)
        try:
            import torch

            from .unet_model import IMAGENET_MEAN, IMAGENET_STD, ClawUNet

            self._torch = torch
            ckpt = torch.load(settings.weights, map_location="cpu", weights_only=True)
            self.input_size = int(ckpt["input_size"])
            model = ClawUNet(ckpt["encoder"], classes=len(ckpt["classes"]), pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            self.device = torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = model.eval().to(self.device)
            self._mean = np.array(IMAGENET_MEAN, np.float32)
            self._std = np.array(IMAGENET_STD, np.float32)
        except Exception as exc:
            raise ModelLoadError(f"Cannot load the U-Net ({settings.weights}): {exc}") from exc

    @property
    def default_roi_padding(self) -> float:
        return self.settings.roi_padding

    def segment(self, roi_image: np.ndarray, hoof_box: BBox) -> list[Mask]:
        return [mask for mask, _ in self.segment_scored(roi_image, hoof_box)]

    def segment_scored(self, roi_image: np.ndarray, hoof_box: BBox) -> list[tuple[Mask, float | None]]:
        s = self.settings
        if hoof_box.height < s.min_claw_height_px:
            return []
        try:
            probs = self.predict(roi_image)
        except Exception as exc:
            raise SegmentationError(f"U-Net segmentation error: {exc}") from exc
        claw = probs.argmax(axis=0) == CLAW
        h, w = claw.shape
        mx, my = hoof_box.width * s.box_margin, hoof_box.height * s.box_margin
        x1, y1 = max(0, int(hoof_box.x1 - mx)), max(0, int(hoof_box.y1 - my))
        x2, y2 = min(w, int(np.ceil(hoof_box.x2 + mx))), min(h, int(np.ceil(hoof_box.y2 + my)))
        inside = np.zeros_like(claw)
        inside[y1:y2, x1:x2] = True
        claw &= inside
        masks = split_claws(claw, s.min_claw_ratio * max(hoof_box.width * hoof_box.height, 1.0))
        masks = [m for m in masks if np.ptp(np.nonzero(m)[0]) + 1 >= s.min_claw_height_px]
        return [(m, float(probs[CLAW][m > 0].mean())) for m in masks]

    def predict(self, roi_image: np.ndarray) -> np.ndarray:
        """Class probabilities (classes, H, W) at ROI size. The ROI is resized to a square as in training.
        With `tta`, the probabilities of the mirrored ROI are averaged in."""
        torch = self._torch
        h, w = roi_image.shape[:2]
        size = self.input_size
        x = cv2.cvtColor(cv2.resize(roi_image, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255.0 - self._mean) / self._std
        batch = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
        if self.settings.tta:
            batch = torch.cat([batch, batch.flip(-1)])
        with torch.no_grad():
            logits = self.model(batch.to(self.device))
            probs = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
            probs = probs.softmax(dim=1)
            if self.settings.tta:
                probs = (probs[:1] + probs[1:].flip(-1)) / 2
            return probs[0].cpu().numpy()
