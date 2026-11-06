"""Two experiments that need no training.

resolution  Controlled downscaling. The close-up test hooves at least 40 px high are shrunk, by resizing the whole image
            with area interpolation, until the hoof is 12, 16, 20, 24, 32 or 40 px high, and JPEG-compressed at quality
            60, 75 or 90 (three replicates per level). Ground-truth claw masks are resized the same way (area, > 0.5).
            Every model gets the ground-truth hoof box of the downscaled image. -> results/resolution/
detector    End-to-end: the hoof box comes from the pipeline's hoof detector (YOLO26m, confidence 0.26) instead of the
            annotation. Each annotated hoof takes the detection with the highest box IoU (>= 0.3); a hoof without such
            a detection counts as a miss (zero overlap). -> results/detector/

    python scripts/downscale_and_detector.py resolution
    python scripts/downscale_and_detector.py detector
    python scripts/downscale_and_detector.py detector --data data/ClawTest --models unet_cnx --out external/detector_per_hoof.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import paths  # noqa: E402
from clawseg.components import BuildContext  # noqa: E402
from clawseg.detection import DETECTORS  # noqa: E402
from clawseg.domain import BBox  # noqa: E402

ROOT = paths.RESULTS
MODELS = ["unet_cnx", "unet_r34", "yolo_roi", "rfdetr_roi", "sam_zs_halves", "sam_lora_halves", "sam3_text_full", "sam3_halves"]
LEVELS = [12, 16, 20, 24, 32, 40]
QUALITIES = [60, 75, 90]
MIN_HEIGHT = 40


def hoof_from_masks(claws, path, subset, side_by_side):
    ys, xs = np.nonzero(np.logical_or.reduce(claws))
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
    h, w = claws[0].shape
    bw, bh = x2 - x1, y2 - y1
    win = (max(0, int(x1 - 0.5 * bw)), max(0, int(y1 - 0.5 * bh)), min(w, int(x2 + 0.5 * bw)), min(h, int(y2 + 0.5 * bh)))
    return {"path": path, "claws": sorted(claws, key=lambda m: np.nonzero(m)[1].mean()),
            "box": BBox.from_xyxy((x1, y1, x2, y2), 1.0), "window": win, "height": bh, "subset": subset,
            "side_by_side": side_by_side}


def run_models(samples, out_csv, ctx, extra_keys):
    rows = []
    builders = {**B.build_models(ctx, [m for m in MODELS if not m.startswith("abl_")]),
                **B.build_ablation_models(ctx, [m for m in MODELS if m.startswith("abl_")])}
    for name, make in builders.items():
        model = make()
        for s in samples:
            if hasattr(model, "_cache"):
                model._cache = {}
            hoof = s["hoof"]
            if hoof["box"] is None:          # no detection for this hoof
                res = {"iou": 0.0, "inst_iou": 0.0, "separated": 0, "n_pred": 0}
            else:
                res, _ = B.evaluate(hoof, B.select(model.predict(s["image"], hoof["box"]), hoof["box"], s["image"].shape))
            rows.append({"model": name, **{k: s[k] for k in extra_keys},
                         **{k: res[k] for k in ("iou", "inst_iou", "separated", "n_pred")}})
        print(name, "mean claw-instance IoU", round(float(np.mean([r["inst_iou"] for r in rows if r["model"] == name])), 3), flush=True)
        del model
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("written", out_csv)


def resolution(ctx):
    samples = []
    for hoof in B.load_hoofs("test"):
        if hoof["subset"] != "close-up" or hoof["height"] < MIN_HEIGHT:
            continue
        image = cv2.imread(str(hoof["path"]))
        for level in LEVELS:
            f = level / hoof["height"]
            size = (max(8, round(image.shape[1] * f)), max(8, round(image.shape[0] * f)))
            small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            claws = [cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.5 for m in hoof["claws"]]
            if any(not c.any() for c in claws):
                continue
            for q in QUALITIES:
                jpg = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
                samples.append({"hoof": hoof_from_masks(claws, hoof["path"], "close-up", hoof["side_by_side"]),
                                "image": jpg, "hoof_id": hoof["id"], "level_px": level, "jpeg_q": q,
                                "native_px": hoof["height"]})
    print(len(samples), "samples from", len({s["hoof_id"] for s in samples}), "hooves")
    run_models(samples, ROOT / (OUT if OUT else "resolution/per_sample.csv"), ctx, ["hoof_id", "level_px", "jpeg_q", "native_px"])


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def detector(ctx):
    det = DETECTORS.create("yolo", {"weights": str(B.W / "YOLO-DET-hoof/yolo26m_v4-best.pt"), "conf": 0.26}, ctx)
    hoofs = B.load_hoofs("test")
    images = {h["path"]: cv2.imread(str(h["path"])) for h in hoofs}
    boxes = {p: det.detect(img) for p, img in images.items()}
    samples, found = [], 0
    for h in hoofs:
        gt = h["box"].as_tuple()
        cand = sorted(((box_iou(gt, b.as_tuple()), b) for b in boxes[h["path"]]), key=lambda t: -t[0])
        best = cand[0] if cand and cand[0][0] >= 0.3 else None
        found += best is not None
        hoof = dict(h, box=best[1] if best else None)
        samples.append({"hoof": hoof, "image": images[h["path"]], "hoof_id": h["id"], "subset": h["subset"],
                        "det_iou": round(best[0], 3) if best else 0.0})
    print(f"detector found {found}/{len(hoofs)} annotated hooves (box IoU >= 0.3)")
    run_models(samples, ROOT / (OUT if OUT else "detector/per_hoof.csv"), ctx, ["hoof_id", "subset", "det_iou"])


OUT = None          # optional output path relative to results/, set with --out

if __name__ == "__main__":
    # optional: resolution_detector.py detector --models m1 m2 ... --out detector_boxscale/per_hoof.csv
    if "--models" in sys.argv:
        MODELS = sys.argv[sys.argv.index("--models") + 1:]
        MODELS = MODELS[:next((i for i, m in enumerate(MODELS) if m.startswith("--")), len(MODELS))]
    if "--out" in sys.argv:
        OUT = sys.argv[sys.argv.index("--out") + 1]
    if "--data" in sys.argv:   # e.g. the external ClawTest set
        B.DATA = Path(sys.argv[sys.argv.index("--data") + 1])
    ctx = BuildContext(base_dir=paths.ROOT, out_dir=ROOT)
    {"resolution": resolution, "detector": detector}[sys.argv[1]](ctx)
