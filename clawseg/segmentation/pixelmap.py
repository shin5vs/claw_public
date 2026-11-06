"""Pixel-map mode: a binary claw map from the hoof-box ROI, split into two claws.

On the map, claw pixels are True and everything else (including the gap between the claws) is False. The
pixel classification is chosen by `method` (the paper uses sam2_box, sam2_halves and sam2_claws):

    otsu         CLAHE lightness with an Otsu threshold; the claw class is the one filling the lower inner box
    kmeans       k-means on Lab colour and position; the claw cluster is chosen by position as for otsu
    grabcut      GrabCut: outside the box sure background, top (hair) probable background, lower inner probable claw
    sam2_box     SAM 2.1 (zero-shot or fine-tuned) prompted with the hoof box
    sam2_halves  SAM 2.1 prompted with the left and right halves of the hoof box (one box per claw)
    sam2_cleft   the SAM 2.1 hoof outline (hoof-box prompt) with the gap cut out: the darkest top-to-bottom seam
                 and the surrounding pixels darker than the claw
    sam2_claws   like sam2_cleft, then each claw prompted again with its own bounding box (one box per claw, as
                 in training); the gap position comes from the image (the dark seam), not from halving the box
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..components import Settings
from ..domain import BBox, Mask
from ..exceptions import ModelLoadError, SegmentationError
from ..security import require_file, verify_sha256
from .base import SEGMENTERS, ClawSegmenter

logger = logging.getLogger(__name__)

METHODS = ("otsu", "kmeans", "grabcut", "sam2_box", "sam2_halves", "sam2_cleft", "sam2_claws")
SAM_METHODS = ("sam2_box", "sam2_halves", "sam2_cleft", "sam2_claws")

# Prior for front views in box coordinates (u, v in [0, 1]): the claws fill the lower inner part of the box,
# hair and coronet are at the top, ground at the edges and corners.
CLAW_REGION = (0.12, 0.88, 0.45, 0.95)   # u0, u1, v0, v1
HAIR_BAND_V = 0.22


@SEGMENTERS.register("pixelmap")
class PixelMapClawSegmenter(ClawSegmenter):
    @dataclass(frozen=True)
    class Settings(Settings):
        method: str = "otsu"
        roi_padding: float = 0.5  # context around the box (GrabCut background, SAM crop); the map stays inside the box
        min_claw_ratio: float = 0.03  # regions smaller than this share of the box area are discarded
        # Claws (and hoof boxes) lower than this are not measured: the gap is only 1-3 px high there
        min_claw_height_px: float = 30.0
        kmeans_clusters: int = 3
        grabcut_iterations: int = 4
        sam2_weights: Path | None = None   # SAM 2.1 hiera base_plus base checkpoint (sam2.1_hiera_base_plus.pt)
        sam2_finetuned: Path | None = None  # fine-tuned layers (`trainable`), e.g. SAM2.1-clawseg-lora/best.pt
        sam2_crop_scale: float = 2.0  # square SAM input crop: this times the longer box side (2.0 in training)
        sam2_sha256: str | None = None
        sam2_finetuned_sha256: str | None = None
        device: str | None = None

        def __post_init__(self) -> None:
            if self.method not in METHODS:
                raise ValueError(f"method: {METHODS} egyike kell legyen, kapott: {self.method!r}")
            if self.roi_padding < 0 or self.min_claw_height_px < 0:
                raise ValueError("roi_padding and min_claw_height_px must not be negative")
            if not 0.0 <= self.min_claw_ratio < 0.5:
                raise ValueError("min_claw_ratio must be between 0 and 0.5")
            if self.kmeans_clusters < 2 or self.grabcut_iterations < 1 or self.sam2_crop_scale < 1.0:
                raise ValueError("kmeans_clusters >= 2, grabcut_iterations >= 1, sam2_crop_scale >= 1 kell")
            if self.method in SAM_METHODS and self.sam2_weights is None:
                raise ValueError(f"method {self.method} needs sam2_weights")

    def __init__(self, settings: "PixelMapClawSegmenter.Settings | None" = None) -> None:
        self.settings = settings or self.Settings()
        self._sam = _Sam2BoxPrompter(self.settings) if self.settings.method in SAM_METHODS else None

    @property
    def default_roi_padding(self) -> float:
        return self.settings.roi_padding

    def segment(self, roi_image: np.ndarray, hoof_box: BBox) -> list[Mask]:
        return [mask for mask, _ in self.segment_scored(roi_image, hoof_box)]

    def segment_scored(self, roi_image: np.ndarray, hoof_box: BBox) -> list[tuple[Mask, float | None]]:
        min_height = self.settings.min_claw_height_px
        if hoof_box.height < min_height:  # a claw cannot be taller than its hoof box, so the model is not called
            logger.debug("pm: hoof box %.0f px high (< %.0f), skipped", hoof_box.height, min_height)
            return []
        try:
            claw_map, conf = self.pixel_map(roi_image, hoof_box)
            min_area = self.settings.min_claw_ratio * max(hoof_box.width * hoof_box.height, 1.0)
            masks = [m for m in split_claws(claw_map, min_area) if np.ptp(np.nonzero(m)[0]) + 1 >= min_height]
            return [(mask, conf) for mask in masks]
        except (ModelLoadError, SegmentationError):
            raise
        except Exception as exc:
            raise SegmentationError(f"Pixel-map error ({self.settings.method}): {exc}") from exc

    def pixel_map(self, roi_image: np.ndarray, hoof_box: BBox) -> tuple[np.ndarray, float | None]:
        """Boolean map of ROI size, True = claw (inside the box only). The confidence is known only for SAM."""
        x1, y1, x2, y2 = _box_pixels(hoof_box, roi_image.shape)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return np.zeros(roi_image.shape[:2], bool), None
        method = self.settings.method
        if method in SAM_METHODS:
            prompts = list(hoof_box.split_halves()) if method == "sam2_halves" else [hoof_box]
            claw_map, conf = self._sam.segment(roi_image, prompts)
            if method in ("sam2_cleft", "sam2_claws"):
                claw_map &= ~_dark_cleft(roi_image, claw_map)
            if method == "sam2_claws":
                min_area = self.settings.min_claw_ratio * max(hoof_box.width * hoof_box.height, 1.0)
                claws = split_claws(claw_map, min_area)
                if len(claws) == 2:
                    each = self._sam.segment_each(roi_image, [_mask_box(m) for m in claws])
                    claw_map, conf = _separated([m for m, _ in each]), float(np.mean([c for _, c in each]))
        else:
            box_img = roi_image[y1:y2, x1:x2]
            if method == "grabcut":
                inside = _grabcut(roi_image, (x1, y1, x2, y2), self.settings.grabcut_iterations)[y1:y2, x1:x2]
            else:
                labels = _otsu_labels(box_img) if method == "otsu" else _kmeans_labels(box_img, self.settings.kmeans_clusters)
                inside = labels == _claw_label(labels)
            claw_map = np.zeros(roi_image.shape[:2], bool)
            claw_map[y1:y2, x1:x2] = inside
            conf = None
        claw_map[:y1] = claw_map[y2:] = False
        claw_map[:, :x1] = claw_map[:, x2:] = False
        return claw_map, conf


# ---------------------------------------------------------------------- map -> two claws

def split_claws(claw_map: np.ndarray, min_area: float) -> list[Mask]:
    """The two claw masks from the claw map (0/255, decreasing area). Touching claws are cut by a watershed.

    Holes are filled only slightly: the gap between curled claws (scores 7-8) is a closed hole and must not vanish.
    """
    binary = cv2.morphologyEx(claw_map.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    holes = ndi.binary_fill_holes(binary) & ~binary
    hole_labels, n_holes = ndi.label(holes)
    if n_holes:
        sizes = ndi.sum(holes, hole_labels, index=np.arange(1, n_holes + 1))
        small = np.isin(hole_labels, 1 + np.nonzero(sizes < 0.1 * min_area)[0])
        binary |= small

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    blobs = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area),
                   key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    if not blobs:
        return []
    if len(blobs) >= 2 and stats[blobs[1], cv2.CC_STAT_AREA] >= 0.25 * stats[blobs[0], cv2.CC_STAT_AREA]:
        return [((labels == i) * 255).astype(np.uint8) for i in blobs[:2]]
    halves = _watershed_split(labels == blobs[0], min_area)
    return halves if halves else [((labels == blobs[0]) * 255).astype(np.uint8)]


def _watershed_split(blob: np.ndarray, min_area: float, max_neck_ratio: float = 0.5) -> list[Mask]:
    """Cuts a connected region at its narrowest neck into two parts (watershed on the distance transform, seeded at
    the maxima of the left and right halves). Only at a real neck: if the cut is longer than `max_neck_ratio` times
    the region height, the region is one convex shape (e.g. a hoof without a visible gap) and is not split."""
    from skimage.segmentation import watershed

    dist = ndi.distance_transform_edt(blob)
    xs = np.nonzero(blob.any(axis=0))[0]
    ys = np.nonzero(blob.any(axis=1))[0]
    mid = (xs.min() + xs.max()) // 2
    markers = np.zeros(blob.shape, np.int32)
    for label, (lo, hi) in ((1, (xs.min(), mid)), (2, (mid + 1, xs.max() + 1))):
        part = np.zeros_like(dist)
        part[:, lo:hi] = dist[:, lo:hi]
        if part.max() <= 0:
            return []
        markers[np.unravel_index(np.argmax(part), part.shape)] = label
    ws = watershed(-dist, markers, mask=blob, watershed_line=True)
    cut = int(np.count_nonzero(blob & (ws == 0)))
    if cut > max_neck_ratio * (ys.max() - ys.min() + 1):
        return []
    halves = [ws == label for label in (1, 2)]
    if min(h.sum() for h in halves) < min_area:
        return []
    return [(h * 255).astype(np.uint8) for h in sorted(halves, key=lambda h: -h.sum())]


def _dark_cleft(image: np.ndarray, hoof: np.ndarray) -> np.ndarray:
    """Gap pixels inside the hoof mask (bool): the darkest seam running from top to bottom (at most 1 px sideways
    per row, within the central 20-80% of the masked row), plus, on both sides of the seam, the pixels darker than the
    midpoint between the seam and the median claw lightness of the row."""
    gap = np.zeros(hoof.shape, bool)
    ys, xs = np.nonzero(hoof)
    if ys.size == 0:
        return gap
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = hoof[y1:y2, x1:x2]
    light = _lightness(image[y1:y2, x1:x2]).astype(np.float64)
    h, w = sub.shape
    cost = np.full((h, w), np.inf)
    rows = [r for r in range(h) if np.count_nonzero(sub[r]) >= 3]
    if len(rows) < 2:
        return gap
    for r in rows:
        cols = np.nonzero(sub[r])[0]
        lo, hi = cols.min(), cols.max()
        a, b = int(np.ceil(lo + 0.2 * (hi - lo))), int(np.floor(lo + 0.8 * (hi - lo)))
        cost[r, a:b + 1] = light[r, a:b + 1]
    r0, r1 = rows[0], rows[-1]
    acc = cost[r0:r1 + 1].copy()
    acc[~np.isfinite(acc)] = 1e9  # outside the mask or central band: costly but passable, so the seam is not interrupted
    back = np.zeros(acc.shape, np.int64)
    for i in range(1, len(acc)):
        prev = np.pad(acc[i - 1], 1, constant_values=np.inf)
        steps = np.stack([prev[:-2], prev[1:-1], prev[2:]])  # upper-left, upper, upper-right
        choice = np.argmin(steps, axis=0)
        back[i] = np.arange(w) + choice - 1
        acc[i] += steps[choice, np.arange(w)]
    seam = np.empty(len(acc), np.int64)
    seam[-1] = int(np.argmin(acc[-1]))
    for i in range(len(acc) - 1, 0, -1):
        seam[i - 1] = back[i, seam[i]]
    for i, c in enumerate(seam):
        r = r0 + i
        row_mask = sub[r]
        if not row_mask[c]:
            continue
        level = light[r, c] + 0.5 * (np.median(light[r][row_mask]) - light[r, c])
        lo = hi = c
        while lo - 1 >= 0 and row_mask[lo - 1] and light[r, lo - 1] < level:
            lo -= 1
        while hi + 1 < w and row_mask[hi + 1] and light[r, hi + 1] < level:
            hi += 1
        gap[y1 + r, x1 + lo:x1 + max(hi, c + 1) + 1] = True  # at least 2 px, so that the claws do not touch even with 8-connectivity
    return gap


# ---------------------------------------------------------------------- classical pixel classifiers

def _box_pixels(box: BBox, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    return (max(0, int(round(box.x1))), max(0, int(round(box.y1))),
            min(w, int(round(box.x2))), min(h, int(round(box.y2))))


def _region_weights(h: int, w: int) -> np.ndarray:
    """+1 in the expected claw region, -1 in the hair band and at the box edges, 0 elsewhere."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    u0, u1, v0, v1 = CLAW_REGION
    weights = np.zeros((h, w))
    weights[(uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)] = 1.0
    weights[(vv < HAIR_BAND_V) | (uu < 0.04) | (uu > 0.96)] = -1.0
    return weights


def _claw_label(labels: np.ndarray) -> int:
    """The claw class is the one filling the expected claw region best (and the hair band and edges least)."""
    weights = _region_weights(*labels.shape)
    scores = {int(k): float(weights[labels == k].mean()) for k in np.unique(labels)}
    return max(scores, key=scores.get)


def _lightness(box_img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(box_img, cv2.COLOR_BGR2LAB)
    tile = max(1, min(box_img.shape[:2]) // 32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(max(2, 8 // tile), max(2, 8 // tile)))
    return clahe.apply(lab[..., 0])


def _otsu_labels(box_img: np.ndarray) -> np.ndarray:
    light = cv2.GaussianBlur(_lightness(box_img), (3, 3), 0)
    _, th = cv2.threshold(light, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th.astype(np.int32)


def _kmeans_labels(box_img: np.ndarray, k: int) -> np.ndarray:
    h, w = box_img.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(box_img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = _lightness(box_img)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # position with a low weight: gives connected regions, but colour decides
    feats = np.dstack([lab, 40.0 * xx / w, 40.0 * yy / h]).reshape(-1, 5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(feats, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w).astype(np.int32)


def _grabcut(roi_image: np.ndarray, box: tuple[int, int, int, int], iterations: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    mask = np.full(roi_image.shape[:2], cv2.GC_BGD, np.uint8)
    inner = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    weights = _region_weights(h, w)
    inner[weights > 0] = cv2.GC_PR_FGD
    mask[y1:y2, x1:x2] = inner
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(roi_image, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


# ---------------------------------------------------------------------- SAM 2.1 (box prompt)

class _Sam2BoxPrompter:
    """SAM 2.1 base_plus with box prompts, using the Ultralytics SAM2 implementation.

    The input has the training geometry: for each prompt a square crop around the prompt box (`sam2_crop_scale` x
    the longer side), resized to 1024 x 1024, black outside the image. The box enters the prompt encoder as two
    corner points (labels 2 and 3). (On the same input the logits match `sam2_common.segment_boxes`.)
    """

    RES = 1024
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, settings: PixelMapClawSegmenter.Settings) -> None:
        self.settings = settings
        require_file(settings.sam2_weights, "SAM 2.1 base checkpoint")
        verify_sha256(settings.sam2_weights, settings.sam2_sha256)
        if settings.sam2_finetuned is not None:
            require_file(settings.sam2_finetuned, "SAM 2.1 fine-tuned layers")
            verify_sha256(settings.sam2_finetuned, settings.sam2_finetuned_sha256)
        try:
            import torch
            from ultralytics.models.sam.build import build_sam2_b

            self._torch = torch
            self.device = torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model = build_sam2_b()  # = SAM 2.1 hiera base_plus architecture
            state = torch.load(settings.sam2_weights, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
            if missing:
                raise ModelLoadError(f"missing keys in the base checkpoint: {missing[:5]}")
            # the video (memory) keys of SAM 2.1 are not needed by the Ultralytics image model
            logger.debug("SAM 2.1: kihagyott kulcsok: %s", unexpected)
            if settings.sam2_finetuned is not None:
                trainable = torch.load(settings.sam2_finetuned, map_location="cpu", weights_only=True)["trainable"]
                _, unexpected = model.load_state_dict(trainable, strict=False)
                if unexpected:
                    raise ModelLoadError(f"unexpected keys in the fine-tuned weights: {unexpected[:5]}")
            # Fine-tuning trained only the single-mask token (apply_postprocessing=False in the training code), whereas
            # Ultralytics would choose among the multi-mask outputs for uncertain masks, which differs on small hooves.
            model.sam_mask_decoder.dynamic_multimask_via_stability = False
            self.model = model.eval().to(self.device)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Cannot load SAM 2.1 ({settings.sam2_weights}): {exc}") from exc

    def segment(self, image: np.ndarray, prompts: list[BBox]) -> tuple[np.ndarray, float]:
        """Union of the prompt masks (bool, image size) and the mean predicted IoU."""
        each = self.segment_each(image, prompts)
        return np.logical_or.reduce([m for m, _ in each]), float(np.mean([c for _, c in each]))

    def segment_each(self, image: np.ndarray, prompts: list[BBox]) -> list[tuple[np.ndarray, float]]:
        """Mask per prompt (bool, image size) and the predicted IoU of the model."""
        from PIL import Image

        torch = self._torch
        rgb_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = []
        with torch.no_grad():
            for prompt in prompts:
                # geometry of the training code (crop_window / crop_image / box_to_crop / paste_back), integer pixel window
                side = max(max(prompt.width, prompt.height) * self.settings.sam2_crop_scale, 64.0)
                x0 = (prompt.x1 + prompt.x2) / 2 - side / 2
                y0 = (prompt.y1 + prompt.y2) / 2 - side / 2
                xi, yi = round(x0), round(y0)
                # PIL: identical to the training crop bit for bit (cv2 resizing differs slightly and changes masks of small hooves)
                crop = rgb_image.crop((xi, yi, round(x0 + side), round(y0 + side))).resize((self.RES, self.RES), Image.BILINEAR)
                rgb = (np.asarray(crop, np.float32) / 255.0 - self.MEAN) / self.STD
                embed, high_res = self._encode(torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device))

                corners = np.array([[prompt.x1, prompt.y1], [prompt.x2, prompt.y2]], np.float32)
                corners = ((corners - [x0, y0]) * (self.RES / side)).astype(np.float32)
                points = torch.from_numpy(corners).to(self.device)[None]
                labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)
                sparse, dense = self.model.sam_prompt_encoder(points=(points, labels), boxes=None, masks=None)
                low, iou, _, _ = self.model.sam_mask_decoder(
                    image_embeddings=embed, image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                    multimask_output=False, repeat_image=False, high_res_features=high_res)
                logits = torch.nn.functional.interpolate(low.float(), size=(self.RES, self.RES), mode="bilinear",
                                                         align_corners=False)[0, 0]
                mask = np.zeros(image.shape[:2], bool)
                _paste(mask, (logits > 0).cpu().numpy(), xi, yi, max(round(side), 1))
                results.append((mask, float(iou.reshape(-1)[0])))
        return results

    def _encode(self, x):
        """Image embedding (bf16 autocast on CUDA). The prompt encoder and decoder run in float32: under autocast
        Ultralytics would compute the positional encoding of the box corners in bf16, which spoils small boxes."""
        torch = self._torch
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            backbone = self.model.forward_image(x)
            _, feats, _, sizes = self.model._prepare_backbone_features(backbone)
            if self.model.directly_add_no_mem_embed:
                feats[-1] = feats[-1] + self.model.no_mem_embed
            maps = [f.permute(1, 2, 0).reshape(1, -1, *s).float() for f, s in zip(feats, sizes)]
        return maps[-1], maps[:-1]


def _mask_box(mask: np.ndarray) -> BBox:
    ys, xs = np.nonzero(mask)
    return BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _separated(masks: list[np.ndarray]) -> np.ndarray:
    """Union of two (possibly overlapping or touching) claw masks with the contact band cut out, leaving two regions."""
    kernel = np.ones((3, 3), np.uint8)
    touch = np.logical_and.reduce([cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in masks])
    return np.logical_or.reduce(masks) & ~touch


def _paste(canvas: np.ndarray, mask_crop: np.ndarray, x: int, y: int, side: int) -> None:
    """Resizes the mask of the 1024 crop to `side` (nearest neighbour) and ORs it onto the canvas."""
    m = cv2.resize(mask_crop.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST).astype(bool)
    sx0, sy0 = max(x, 0), max(y, 0)
    sx1, sy1 = min(x + side, canvas.shape[1]), min(y + side, canvas.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0:sy1, sx0:sx1] |= m[sy0 - y:sy1 - y, sx0 - x:sx1 - x]
