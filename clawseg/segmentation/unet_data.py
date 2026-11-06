"""U-Net training data: three-class label maps (background / claw / gap) and hoof ROIs from claw polygons.

Pure functions (numpy + cv2). The gap is the area between the two claws of a hoof, row by row, within the
rows shared by both claws. Where the claws touch or overlap, the contact columns are labelled as gap, so
the two claws always form separate regions.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

BACKGROUND, CLAW, GAP = 0, 1, 2
CLASS_NAMES = ("background", "claw", "gap")


def read_yolo_polygons(label_path: str | Path, width: int, height: int) -> list[np.ndarray]:
    """YOLO-seg label file (class and normalised x y pairs per line) -> polygons in pixels (N, 2)."""
    polygons = []
    for line in Path(label_path).read_text().splitlines():
        values = line.split()
        if len(values) < 7:          # class and at least 3 points
            continue
        xy = np.asarray(values[1:], np.float64).reshape(-1, 2) * [width, height]
        polygons.append(xy)
    return polygons


def rasterize(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def _box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def pair_claws(boxes: list[tuple[float, float, float, float]]) -> tuple[list[tuple[int, int]], list[int]]:
    """Pairs side-by-side claws into hooves (greedy, lowest-cost pair first).

    A pair must be side by side, overlap vertically by at least 40% and be at most 0.8 claw widths apart.
    Returns: (pairs, unpaired claws).
    """
    cands = []
    for i, j in combinations(range(len(boxes)), 2):
        (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = boxes[i], boxes[j]
        aw, ah, bw, bh = ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1
        if min(ay2, by2) - max(ay1, by1) < 0.4 * min(ah, bh):
            continue
        mean_w, mean_h = (aw + bw) / 2, (ah + bh) / 2
        gap = max(ax1, bx1) - min(ax2, bx2)
        if gap > 0.8 * mean_w or abs((ax1 + ax2) - (bx1 + bx2)) / 2 < 0.25 * mean_w:
            continue
        cands.append((max(gap, 0) / mean_w + abs((ay1 + ay2) - (by1 + by2)) / 2 / mean_h, i, j))
    used, pairs = set(), []
    for _, i, j in sorted(cands):
        if i not in used and j not in used:
            pairs.append((i, j))
            used |= {i, j}
    return pairs, [i for i in range(len(boxes)) if i not in used]


def label_map(shape: tuple[int, int], polygons: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]], list[np.ndarray]]:
    """Label map (uint8: 0 background, 1 claw, 2 gap), the pairs and the claw masks."""
    masks = [m for m in (rasterize(p, shape) for p in polygons) if m.any()]
    labels = np.zeros(shape, np.uint8)
    for m in masks:
        labels[m] = CLAW
    pairs, _ = pair_claws([_box(m) for m in masks])
    for i, j in pairs:
        left, right = sorted((masks[i], masks[j]), key=lambda m: np.nonzero(m)[1].mean())
        for y in np.nonzero(left.any(axis=1) & right.any(axis=1))[0]:
            xl = np.nonzero(left[y])[0].max()
            xr = np.nonzero(right[y])[0].min()
            if xr - xl > 1:
                labels[y, xl + 1:xr] = GAP
            else:                     # touching or overlapping: the contact columns separate them
                labels[y, min(xl, xr):max(xl, xr) + 1] = GAP
    return labels, pairs, masks


def hoof_rois(shape: tuple[int, int], masks: list[np.ndarray], pairs: list[tuple[int, int]],
              pad: float = 0.5) -> list[tuple[int, int, int, int]]:
    """ROI per hoof (pair or unpaired claw): the bounding box with a margin of `pad`, clipped to the image
    (same geometry as `crop_expanded_roi`)."""
    h, w = shape
    paired = {k for p in pairs for k in p}
    groups = [list(p) for p in pairs] + [[k] for k in range(len(masks)) if k not in paired]
    rois = []
    for group in groups:
        boxes = [_box(masks[k]) for k in group]
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        rois.append((max(0, int(x1) - px), max(0, int(y1) - py), min(w, int(x2) + px), min(h, int(y2) + py)))
    return rois
