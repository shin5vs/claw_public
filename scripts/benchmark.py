"""Unified claw-surface segmentation benchmark on the ClawSet2 test split.

Every model receives the same ground-truth hoof box (bounding box of the hoof's annotated claw polygons) and its
predicted claw instances pass through the same selection step (centre inside the hoof box enlarged by 10%, mask NMS,
the two largest masks). Metrics are computed per hoof inside the hoof evaluation window (hoof box enlarged by 0.5 of
its size on each side, i.e. the U-Net training ROI). A hoof for which a model returns no mask scores IoU = 0 and the
penalty HD95 (window diagonal); nothing is averaged over successful cases only.

    python scripts/benchmark.py --out results/benchmark                      # internal test set, all configurations
    python scripts/benchmark.py --data data/ClawTest --models unet_cnx yolo_roi --out results/external
    python scripts/benchmark.py --ablation --models abl_cnx_boxscale_s0 --out results/primary   # U-Net variants

Outputs: per_hoof.csv, summary.csv, stats.json, masks.npz (predicted claw instances per hoof and model, window frame).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

from clawseg.components import BuildContext  # noqa: E402
from clawseg.domain import BBox  # noqa: E402
from clawseg.roi import crop_expanded_roi  # noqa: E402
from clawseg.segmentation import SEGMENTERS, MaskSelector  # noqa: E402
from clawseg.segmentation.selection import MaskSelectionSettings  # noqa: E402
from clawseg.segmentation.unet_data import pair_claws, rasterize, read_yolo_polygons  # noqa: E402

W = paths.WEIGHTS
DATA = paths.DATA
RFDETR = W / "RFDETR-SEG-claw" / "rfdetrmedium.pt"
SAM_BASE = W / "SAM2.1-base" / "sam2.1_hiera_base_plus.pt"
SAM_LORA = W / "SAM2.1-clawseg-lora" / "best.pt"
MATCHED = W / "matched"
SAM_METHODS = ("sam2_box", "sam2_halves", "sam2_claws")


# ----------------------------------------------------------------------------------------------- ground truth

def load_hoofs(split: str):
    """Per hoof: image path, claw masks (full frame), hoof box, evaluation window, subset (close-up / video)."""
    hoofs = []
    for img_path in sorted((DATA / split / "images").iterdir()):
        lbl = DATA / split / "labels" / (img_path.stem + ".txt")
        if not lbl.is_file():
            continue
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        masks = [m for m in (rasterize(p, (h, w)) for p in read_yolo_polygons(lbl, w, h)) if m.any()]
        if not masks:
            continue
        boxes = [_box(m) for m in masks]
        pairs, single = pair_claws(boxes)
        side_by_side = [list(p) for p in pairs]
        stacked, single = stack_claws(boxes, single)
        for k, group in enumerate(side_by_side + stacked + [[i] for i in single]):
            claws = sorted((masks[i] for i in group), key=lambda m: np.nonzero(m)[1].mean())
            x1, y1, x2, y2 = _box(np.logical_or.reduce(claws))
            bw, bh = x2 - x1, y2 - y1
            win = (max(0, int(x1 - 0.5 * bw)), max(0, int(y1 - 0.5 * bh)), min(w, int(x2 + 0.5 * bw)), min(h, int(y2 + 0.5 * bh)))
            hoofs.append({"id": f"{img_path.name.split('_jpg')[0]}#{k}", "path": img_path, "claws": claws,
                          "side_by_side": group in side_by_side,
                          "box": BBox.from_xyxy((x1, y1, x2, y2), 1.0), "window": win, "height": bh,
                          "subset": "close-up" if img_path.name.startswith("IMG_") else "video"})
    return hoofs


def stack_claws(boxes, single, grow=0.05):
    """Side views show the two claws of a hoof one above the other, which `pair_claws` (side-by-side claws only) does
    not pair. Unpaired claws whose boxes touch (after growing each box by `grow` of its size) form one hoof."""
    def touch(a, b):
        ga = [grow * (a[2] - a[0]), grow * (a[3] - a[1])]
        gb = [grow * (b[2] - b[0]), grow * (b[3] - b[1])]
        return (a[0] - ga[0] < b[2] + gb[0] and b[0] - gb[0] < a[2] + ga[0]
                and a[1] - ga[1] < b[3] + gb[1] and b[1] - gb[1] < a[3] + ga[1])
    left, groups = list(single), []
    while left:
        i = left.pop(0)
        j = next((j for j in left if touch(boxes[i], boxes[j])), None)
        if j is None:
            groups.append([i])
        else:
            left.remove(j)
            groups.append([i, j])
    return [g for g in groups if len(g) == 2], [g[0] for g in groups if len(g) == 1]


def _box(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# ----------------------------------------------------------------------------------------------- models

class PipelineModel:
    """A clawseg segmenter run on the expanded hoof ROI (its own default padding)."""

    def __init__(self, seg, pad=None):
        self.seg, self.pad = seg, pad

    def predict(self, image, box):
        roi = crop_expanded_roi(image, box, self.pad if self.pad is not None else self.seg.default_roi_padding)
        rx, ry = roi.origin
        out = []
        for mask, conf in self.seg.segment_scored(roi.image, box.shifted(-rx, -ry)):
            full = np.zeros(image.shape[:2], bool)
            full[ry:ry + mask.shape[0], rx:rx + mask.shape[1]] = mask > 0
            out.append((full, conf))
        return out


class YoloFull:
    def __init__(self, weights, conf):
        from ultralytics import YOLO
        self.model, self.conf = YOLO(str(weights)), conf
        self._cache = {}

    def predict(self, image, box):
        key = id(image)
        if key not in self._cache:
            r = self.model.predict(image, conf=self.conf, verbose=False, retina_masks=True)[0]
            self._cache = {key: [] if r.masks is None else
                           [(m > 0.5, float(c)) for m, c in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy())]}
        return self._cache[key]


class RfDetr:
    """RF-DETR-Seg Medium (432 px). mode = 'full' (whole frame) or 'roi' (hoof ROI with `pad`)."""

    def __init__(self, threshold, mode, pad=0.5):
        from rfdetr import RFDETRSegMedium
        self.model = RFDETRSegMedium(pretrain_weights=str(RFDETR), num_classes=1, trust_checkpoint=True)
        self.threshold, self.mode, self.pad = threshold, mode, pad
        self._cache = {}

    def _run(self, bgr):
        d = self.model.predict(np.ascontiguousarray(bgr[:, :, ::-1]), threshold=self.threshold)
        if d.mask is None or len(d) == 0:
            return []
        return [(m.astype(bool), float(c)) for m, c in zip(d.mask, d.confidence)]

    def predict(self, image, box):
        if self.mode == "full":
            key = id(image)
            if key not in self._cache:
                self._cache = {key: self._run(image)}
            return self._cache[key]
        roi = crop_expanded_roi(image, box, self.pad)
        rx, ry = roi.origin
        out = []
        for m, c in self._run(roi.image):
            full = np.zeros(image.shape[:2], bool)
            full[ry:ry + m.shape[0], rx:rx + m.shape[1]] = m
            out.append((full, c))
        return out


SAM3_WEIGHTS = W / "SAM3" / "sam3.pt"
SAM3_TEXT = "claw"          # fixed a priori (the dataset class is `claw_surface`); not tuned on the test split


class Sam3:
    """SAM 3, zero-shot (no fine-tuning).

    mode 'text_full' / 'text_roi': concept (text) prompt on the whole frame / on the hoof ROI, instance masks returned;
    mode 'box' / 'halves': the hoof box / its two halves as geometric prompts on the ROI, then the same claw split as
    the SAM 2.1 pixel-map source (union inside the box -> split_claws)."""

    def __init__(self, mode, conf=0.25, pad=0.5):
        from ultralytics.models.sam import SAM3Predictor, SAM3SemanticPredictor
        ov = dict(conf=conf, task="segment", mode="predict", model=str(SAM3_WEIGHTS), half=True, save=False, verbose=False)
        self.pred = (SAM3SemanticPredictor if mode.startswith("text") else SAM3Predictor)(overrides=ov)
        self.mode, self.pad = mode, pad
        self._cache = {}

    def _text(self, bgr):
        self.pred.set_image(bgr)
        r = self.pred(text=[SAM3_TEXT])[0]
        if r.masks is None:
            return []
        h, w = bgr.shape[:2]
        return [(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0, float(c))
                for m, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.conf.cpu().numpy())]

    def predict(self, image, box):
        from clawseg.segmentation.pixelmap import split_claws
        if self.mode == "text_full":
            key = id(image)
            if key not in self._cache:
                self._cache = {key: self._text(image)}
            return self._cache[key]
        roi = crop_expanded_roi(image, box, self.pad)
        rx, ry = roi.origin
        rbox = box.shifted(-rx, -ry)
        if self.mode == "text_roi":
            local = self._text(roi.image)
        else:
            prompts = list(rbox.split_halves()) if self.mode == "halves" else [rbox]
            r = self.pred(source=roi.image, bboxes=[list(b.as_tuple()) for b in prompts])[0]
            union = np.zeros(roi.image.shape[:2], bool)
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy() > 0.5:
                    union |= cv2.resize(m.astype(np.uint8), union.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            x1, y1, x2, y2 = (int(round(v)) for v in rbox.as_tuple())
            inside = np.zeros_like(union)
            inside[max(y1, 0):y2, max(x1, 0):x2] = True
            local = [(m > 0, None) for m in split_claws(union & inside, 0.03 * max(rbox.width * rbox.height, 1.0))]
        out = []
        for m, c in local:
            full = np.zeros(image.shape[:2], bool)
            full[ry:ry + m.shape[0], rx:rx + m.shape[1]] = m
            out.append((full, c))
        return out


ABLATION = W / "ablation"


def ablation_unet(path, ctx):
    """A U-Net checkpoint of train_variants.py (2 or 3 classes, with or without image skip), wrapped in
    UnetClawSegmenter so that inference (ROI, box margin, claw split) is identical for all U-Nets."""
    import torch
    from clawseg.segmentation.unet import UnetClawSegmenter
    from clawseg.segmentation.unet_model import IMAGENET_MEAN, IMAGENET_STD
    from train_variants import build_model
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    seg = object.__new__(UnetClawSegmenter)
    seg.settings = UnetClawSegmenter.Settings(weights=Path(path))
    seg._torch = torch
    seg.input_size = int(ckpt["input_size"])
    model = build_model(ckpt["encoder"], len(ckpt["classes"]), bool(ckpt["image_skip"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    seg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg.model = model.eval().to(seg.device)
    seg._mean, seg._std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
    return PipelineModel(seg)


NNUNET = paths.NNUNET
NNUNET_TRAINER = "nnUNetTrainer_100epochs"


def nnunet_model(trainer=NNUNET_TRAINER, fold=0, checkpoint="checkpoint_final.pth", mirroring=True, results="results"):
    """nnU-Net v2 (2D, Dataset501_AngusClaw) wrapped in the pipeline's UnetClawSegmenter, so that the window, box
    margin and claw split are identical to the U-Nets; only the network and its training recipe differ. The window is
    resized to the 320x320 training size, predicted with nnU-Net's default inference (Gaussian tiling, mirroring) from
    checkpoint_final, and the class probabilities are resized back."""
    import os
    for key, sub in (("nnUNet_raw", "raw"), ("nnUNet_preprocessed", "preprocessed"), ("nnUNet_results", "results")):
        os.environ.setdefault(key, str(NNUNET / sub))
    import torch
    from clawseg.segmentation.unet import UnetClawSegmenter
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    folder = NNUNET / results / "Dataset501_AngusClaw" / f"{trainer}__nnUNetPlans__2d"

    class NnUNetSegmenter(UnetClawSegmenter):
        def __init__(self):
            self.settings = UnetClawSegmenter.Settings(weights=folder / f"fold_{fold}" / checkpoint)
            self.input_size = 320
            self.predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirroring,
                                             perform_everything_on_device=True,
                                             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                             verbose=False, verbose_preprocessing=False, allow_tqdm=False)
            self.predictor.initialize_from_trained_model_folder(str(folder), use_folds=(fold,),
                                                                checkpoint_name=checkpoint)

        def predict(self, roi_image):
            h, w = roi_image.shape[:2]
            x = cv2.cvtColor(cv2.resize(roi_image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR),
                             cv2.COLOR_BGR2RGB).astype(np.float32)
            x = x.transpose(2, 0, 1)[:, None]                      # (channels, 1, H, W), as NaturalImage2DIO returns
            _, probs = self.predictor.predict_single_npy_array(x, {"spacing": [999, 1, 1]}, None, None, True)
            probs = np.asarray(probs)[:, 0]                        # (classes, H, W)
            return np.stack([cv2.resize(p.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for p in probs])

    return PipelineModel(NnUNetSegmenter())


def build_ablation_models(ctx, only=None):
    paths = sorted(p for p in ABLATION.glob("*.pt"))
    return {f"abl_{p.stem}": (lambda p=p: ablation_unet(p, ctx)) for p in paths if only is None or f"abl_{p.stem}" in only}


def build_models(ctx, only=None):
    unet = lambda wt: SEGMENTERS.create("unet", {"weights": str(wt)}, ctx)
    sam = lambda method, lora, ft=SAM_LORA: SEGMENTERS.create("pixelmap", {
        "method": method, "sam2_weights": str(SAM_BASE), "sam2_finetuned": str(ft) if lora else None,
        "roi_padding": 0.5, "min_claw_height_px": 0}, ctx)
    yolo_w = W / "YOLO-SEG-claw" / "yolo26s-clawseg2.pt"
    specs = {
        "unet_cnx": lambda: PipelineModel(unet(W / "UNET-claw" / "unet_cnx.pt")),
        "unet_r34": lambda: PipelineModel(unet(W / "UNET-claw" / "unet_resnet34.pt")),
        "yolo_roi": lambda: PipelineModel(SEGMENTERS.create("yolo", {"weights": str(yolo_w), "conf": 0.28, "roi_padding": 0.5}, ctx)),
        "yolo_roi15": lambda: PipelineModel(SEGMENTERS.create("yolo", {"weights": str(yolo_w), "conf": 0.28, "roi_padding": 1.5}, ctx)),
        "yolo_full": lambda: YoloFull(yolo_w, 0.28),
        "rfdetr_full": lambda: RfDetr(0.3, "full"),
        "rfdetr_roi": lambda: RfDetr(0.3, "roi", 0.5),
        "rfdetr_roi15": lambda: RfDetr(0.3, "roi", 1.5),
        # confidence-threshold sensitivity (window 0.5): rerun only when named, e.g. --models yolo_roi_c005
        **{f"yolo_roi_c{t}": (lambda c=c: PipelineModel(SEGMENTERS.create("yolo", {"weights": str(yolo_w), "conf": c, "roi_padding": 0.5}, ctx)))
           for t, c in (("010", 0.10), ("005", 0.05), ("001", 0.01))},
        **{f"rfdetr_roi_c{t}": (lambda c=c: RfDetr(c, "roi", 0.5)) for t, c in (("010", 0.10), ("005", 0.05), ("001", 0.01))},
        # data-matched baselines: retrained on exactly the training set of the primary U-Net
        # (scripts/make_matched_package.py and notebooks/matched_baselines_colab.ipynb)
        "yolo_roi_matched": lambda: PipelineModel(SEGMENTERS.create("yolo", {"weights": str(MATCHED / "yolo26s_matched/best.pt"), "conf": 0.28, "roi_padding": 0.5}, ctx)),
        "yolo_roi_matched_c001": lambda: PipelineModel(SEGMENTERS.create("yolo", {"weights": str(MATCHED / "yolo26s_matched/best.pt"), "conf": 0.01, "roi_padding": 0.5}, ctx)),
        **{f"sam_ftm_{m.split('_')[1]}": (lambda m=m: PipelineModel(sam(m, True, MATCHED / "sam21_matched/best.pt"))) for m in SAM_METHODS},
        **{f"sam_zs_{m.split('_')[1]}": (lambda m=m: PipelineModel(sam(m, False))) for m in SAM_METHODS},
        **{f"sam_lora_{m.split('_')[1]}": (lambda m=m: PipelineModel(sam(m, True))) for m in SAM_METHODS},
        **{f"sam3_{m}": (lambda m=m: Sam3(m)) for m in ("text_full", "text_roi", "box", "halves")},
        "nnunet": lambda: nnunet_model(),
        "nnunet_nomirror": lambda: nnunet_model(mirroring=False),
    }
    # further nnU-Net runs by name: nnunet:<trainer>:<fold>[:nomirror] (local runs) or nnunetc:... (runs trained on
    # Colab, unpacked into nnunet/results_colab), e.g. nnunetc:nnUNetTrainer_100epochs:1:nomirror
    for name in only or []:
        if name.startswith(("nnunet:", "nnunetc:")):
            parts = name.split(":")
            specs[name] = (lambda t=parts[1], fo=int(parts[2]), m=not name.endswith(":nomirror"),
                           r="results_colab" if name.startswith("nnunetc:") else "results": nnunet_model(t, fo, mirroring=m, results=r))
    if only is None:          # the default benchmark run keeps the original configurations
        for k in [k for k in specs if k.startswith(("nnunet", "yolo_roi_matched", "sam_ftm_", "yolo_roi_c", "rfdetr_roi_c"))]:
            specs.pop(k)
    return {k: v for k, v in specs.items() if only is None or k in only}


def select(preds, box: BBox, shape):
    """Shared selection: centre inside the hoof box (+10%), mask NMS 0.5, two largest."""
    mx, my = 0.1 * box.width, 0.1 * box.height
    inside = []
    for m, c in preds:
        ys, xs = np.nonzero(m)
        if xs.size and box.x1 - mx <= xs.mean() <= box.x2 + mx and box.y1 - my <= ys.mean() <= box.y2 + my:
            inside.append((m.astype(np.uint8), c))
    sel = MaskSelector(MaskSelectionSettings(max_claws=2, mask_iou=0.5, drop_cut_edge=False, require_center_in_box=False))
    kept = sel.select_scored(inside, box, (False,) * 4)
    return sorted((m > 0 for m, _ in kept), key=lambda m: np.nonzero(m)[1].mean())


# ----------------------------------------------------------------------------------------------- metrics

def iou(a, b):
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else 1.0


def dice(a, b):
    s = np.count_nonzero(a) + np.count_nonzero(b)
    return float(2 * np.count_nonzero(a & b) / s) if s else 1.0


def inner_band(mask, d):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    return mask & ~cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def boundary_iou(gt, pr, d):
    """Boundary IoU (Cheng et al., 2021): IoU of the inner boundary bands of width d."""
    return iou(inner_band(gt, d), inner_band(pr, d))


def contour(mask):
    return inner_band(mask, 1)


def hd95(gt, pr, penalty):
    if not pr.any():
        return penalty
    cg, cp = contour(gt), contour(pr)
    dg = distance_transform_edt(~cg)   # distance to the GT contour
    dp = distance_transform_edt(~cp)
    return float(np.percentile(np.concatenate([dp[cg], dg[cp]]), 95))


def gap_map(left, right):
    """Interdigital gap between two claws, constructed as for the U-Net labels (unet_data.label_map)."""
    gap = np.zeros_like(left)
    for y in np.nonzero(left.any(axis=1) & right.any(axis=1))[0]:
        xl, xr = np.nonzero(left[y])[0].max(), np.nonzero(right[y])[0].min()
        if xr - xl > 1:
            gap[y, xl + 1:xr] = True
        else:   # touching or overlapping claws: the contact columns separate them
            gap[y, min(xl, xr):max(xl, xr) + 1] = True
    return gap


def evaluate(hoof, pred_full):
    x1, y1, x2, y2 = hoof["window"]
    crop = lambda m: m[y1:y2, x1:x2]
    gts = [crop(m) for m in hoof["claws"]]
    prs = [crop(m) for m in pred_full]
    g_union = np.logical_or.reduce(gts)
    p_union = np.logical_or.reduce(prs) if prs else np.zeros_like(g_union)
    diag = float(np.hypot(*g_union.shape))
    d = max(1, int(round(0.02 * diag)))
    # instance level: Hungarian matching on IoU, unmatched GT claws count as 0
    cost = np.array([[iou(g, p) for p in prs] for g in gts]) if prs else np.zeros((len(gts), 0))
    matched = np.zeros(len(gts))
    if prs:
        r, c = linear_sum_assignment(-cost)
        matched[r] = cost[r, c]
    res = {"iou": iou(g_union, p_union), "dice": dice(g_union, p_union),
           "biou": boundary_iou(g_union, p_union, d), "hd95_px": hd95(g_union, p_union, diag),
           "inst_iou": float(matched.mean()), "n_pred": len(prs), "empty": int(not p_union.any())}
    res["hd95_rel"] = res["hd95_px"] / hoof["height"]
    if len(gts) == 2:
        res["separated"] = int(bool((matched >= 0.5).all()))
        if hoof["side_by_side"]:   # the gap between claws stacked in a side view is not a left-right gap
            g_gap = gap_map(*gts)
            p_gap = gap_map(*prs) if len(prs) == 2 else np.zeros_like(g_gap)
            res["gap_iou"] = iou(g_gap, p_gap) if g_gap.any() else np.nan
        else:
            res["gap_iou"] = np.nan
    else:
        res["separated"] = np.nan
        res["gap_iou"] = np.nan
    return res, prs


# ----------------------------------------------------------------------------------------------- main

METRICS = ("iou", "dice", "biou", "hd95_rel", "hd95_px", "inst_iou", "separated", "gap_iou", "ms")


def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)])
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, x.size)).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def gap_center(masks):
    """Mean x of the interdigital midline (between the two selected claws), or None."""
    if len(masks) != 2:
        return None
    left, right = masks
    rows = np.nonzero(left.any(axis=1) & right.any(axis=1))[0]
    if rows.size == 0:
        return None
    return float(np.mean([(np.nonzero(left[y])[0].max() + np.nonzero(right[y])[0].min()) / 2 for y in rows]))


def run_jitter(hoofs, images, models, out, shift=0.1):
    """Prompt sensitivity: the hoof box is shifted horizontally by ±shift·width. Reports the claw-instance IoU under the
    shifted box and how far the predicted interdigital midline follows the box (0 = follows the image, 1 = the box)."""
    import torch
    rows = []
    for name, make in models.items():
        model = make()
        for hoof in hoofs:
            image, box = images[hoof["path"]], hoof["box"]
            if hasattr(model, "_cache"):
                model._cache = {}
            rec = {"model": name, "hoof": hoof["id"], "subset": hoof["subset"], "n_gt": len(hoof["claws"])}
            centres = []
            for sign in (-1, 1):
                sbox = box.shifted(sign * shift * box.width, 0)
                masks = select(model.predict(image, sbox), sbox, image.shape)
                res, _ = evaluate(hoof, masks)
                rec[f"inst_iou_{'m' if sign < 0 else 'p'}"] = res["inst_iou"]
                rec[f"iou_{'m' if sign < 0 else 'p'}"] = res["iou"]
                centres.append(gap_center(masks))
            rec["follow"] = (centres[1] - centres[0]) / (2 * shift * box.width) if None not in centres and len(hoof["claws"]) == 2 else np.nan
            rows.append(rec)
        print("jitter", name, {k: round(float(np.nanmean([r[k] for r in rows if r["model"] == name])), 3)
                               for k in ("inst_iou_m", "inst_iou_p", "follow")}, flush=True)
        del model
        torch.cuda.empty_cache()
    with open(out / "jitter.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jitter", action="store_true", help="only run the prompt-sensitivity experiment (jitter.csv)")
    ap.add_argument("--ablation", action="store_true", help="evaluate the U-Net ablation checkpoints instead of the benchmark models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=None, help="dataset root (default: ClawSet3); e.g. the external ClawTest set")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats per hoof (median)")
    args = ap.parse_args(argv)
    if args.data:
        global DATA
        DATA = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    hoofs = load_hoofs(args.split)
    print(f"{len(hoofs)} hoofs: " + ", ".join(f"{s} {sum(h['subset'] == s for h in hoofs)}" for s in ("close-up", "video")))
    ctx = BuildContext(base_dir=paths.ROOT, out_dir=out)
    images = {h["path"]: cv2.imread(str(h["path"])) for h in hoofs}
    builder = build_ablation_models if args.ablation else build_models
    if args.jitter:
        run_jitter(hoofs, images, builder(ctx, args.models), out)
        return
    rows, store = [], {}
    for name, make in builder(ctx, args.models).items():
        model = make()
        h0 = hoofs[0]
        model.predict(images[h0["path"]], h0["box"])          # warm-up
        if hasattr(model, "_cache"):
            model._cache = {}
        for hoof in hoofs:
            image = images[hoof["path"]]
            times = []
            for _ in range(args.repeats):
                if hasattr(model, "_cache"):
                    model._cache = {}
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = model.predict(image, hoof["box"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            res, prs = evaluate(hoof, select(preds, hoof["box"], image.shape))
            res["ms"] = 1000 * float(np.median(times))
            rows.append({"model": name, "hoof": hoof["id"], "subset": hoof["subset"], "hoof_h": hoof["height"],
                         "n_gt": len(hoof["claws"]), **res})
            for k, m in enumerate(prs):
                store[f"{name}|{hoof['id']}|{k}"] = np.packbits(m)
                store[f"{name}|{hoof['id']}|{k}|shape"] = np.array(m.shape)
        print(name, {k: round(float(np.nanmean([r[k] for r in rows if r["model"] == name])), 3) for k in METRICS}, flush=True)
        del model
        torch.cuda.empty_cache()

    for hoof in hoofs:   # ground truth in the window frame, for figures
        x1, y1, x2, y2 = hoof["window"]
        for k, m in enumerate(hoof["claws"]):
            store[f"gt|{hoof['id']}|{k}"] = np.packbits(m[y1:y2, x1:x2])
            store[f"gt|{hoof['id']}|{k}|shape"] = np.array(m[y1:y2, x1:x2].shape)
        store[f"window|{hoof['id']}"] = np.array([str(hoof["path"]), *map(str, hoof["window"])])
    np.savez_compressed(out / "masks.npz", **store)

    keys = list(rows[0])
    with open(out / "per_hoof.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    summary, stats = [], {}
    models = list(dict.fromkeys(r["model"] for r in rows))
    for name in models:
        for subset in ("all", "close-up", "video"):
            sel = [r for r in rows if r["model"] == name and (subset == "all" or r["subset"] == subset)]
            rec = {"model": name, "subset": subset, "n": len(sel)}
            for k in METRICS:
                vals = np.array([r[k] for r in sel], float)
                lo, hi = bootstrap_ci(vals)
                rec |= {k: np.nanmean(vals), f"{k}_median": np.nanmedian(vals), f"{k}_lo": lo, f"{k}_hi": hi}
            summary.append(rec)
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)

    ref = "unet_cnx"
    if ref in models and not args.ablation:
        for name in models:
            if name == ref:
                continue
            stats[name] = {}
            for k in ("iou", "biou", "hd95_rel", "inst_iou"):
                a = np.array([r[k] for r in rows if r["model"] == ref], float)
                b = np.array([r[k] for r in rows if r["model"] == name], float)
                diff = a - b
                p = float(wilcoxon(a, b, zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0
                stats[name][k] = {"mean_diff": float(np.mean(diff)), "p_wilcoxon": p}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("done:", out)


if __name__ == "__main__":
    main()
