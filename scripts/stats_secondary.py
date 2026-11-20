"""Secondary comparisons of the paper with Holm correction within each family (one family per research question).

The main comparisons (primary U-Net against the seven benchmark models, Holm over the seven per metric and subset) are
computed in stats_internal.py and stats_external.py and are not repeated here. Every test is a two-sided Wilcoxon signed-rank
test on seed- or run-averaged per-hoof values (zero differences split), as in the paper. On the external set every test is
repeated on frame means, and Holm is applied to the hoof-level and the frame-level P values separately.

    python scripts/stats_secondary.py   (after stats_internal.py, which writes results/primary and results/nnunet_avg)

Writes results/secondary_tests.json.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

R = paths.RESULTS


def read(*paths):
    return [r for p in paths for r in (csv.DictReader(open(R / p)) if (R / p).is_file() else [])]


def f(x):
    return float(x) if x not in ("", "nan", None) else np.nan


INT = read("benchmark/per_hoof.csv", "nnunet_avg/per_hoof.csv", "nnunet/per_hoof.csv", "nnunet_nomirror/per_hoof.csv",
           "lowconf_internal/per_hoof.csv", "primary/per_hoof.csv", "ablation/per_hoof.csv")
EXT = read("clawtest/bench/per_hoof.csv", "clawtest/abl/per_hoof.csv", "clawtest/lowconf/per_hoof.csv")
EXT_DET = read("clawtest/detector_per_hoof.csv")

SEEDS = lambda p: [f"{p}_s{s}" for s in range(3)]  # noqa: E731
GROUPS = {
    "primary": SEEDS("abl_cnx_boxscale"), "full": SEEDS("abl_cnx_full"), "nogap": SEEDS("abl_cnx_nogap"),
    "photoval": SEEDS("abl_cnx_photoval"), "fliponly": SEEDS("abl_cnx_fliponly"), "scratch": SEEDS("abl_cnx_scratch"),
    "dlv3p": SEEDS("abl_dlv3p_cnx"), "segformer": SEEDS("abl_segformer_b2"),
    "nnunet_avg_ext": ["nnunet_nomirror"] + [f"nnunetc:nnUNetTrainer_100epochs:{k}:nomirror" for k in range(3)],
}


def ph(rows, name, key="inst_iou", sub=None, hoof="hoof"):
    """Per-hoof value, averaged over the seeds or runs of a group."""
    d = defaultdict(list)
    for r in rows:
        if r["model"] in GROUPS.get(name, [name]) and (sub is None or r.get("subset") == sub):
            d[r[hoof]].append(f(r[key]))
    return {h: float(np.nanmean(v)) for h, v in d.items()}


def frames(d):
    g = defaultdict(list)
    for h, v in d.items():
        g[h.split("#")[0]].append(v)
    return {k: float(np.mean(v)) for k, v in g.items()}


def p(a, b):
    hs = sorted(k for k in set(a) & set(b) if not (np.isnan(a[k]) or np.isnan(b[k])))
    x, y = np.array([a[h] for h in hs]), np.array([b[h] for h in hs])
    return float(wilcoxon(x, y, zero_method="zsplit").pvalue) if np.any(x != y) else 1.0


def holm(ps):
    ps = np.asarray(ps, float)
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj.tolist()


I = lambda name, **k: ph(INT, name, **k)  # noqa: E731,E741
E = lambda name, **k: ph(EXT, name, **k)  # noqa: E731
V = "video"

# family -> {test: (a, b)} ; internal families on hooves, external families on hooves and frame means
INTERNAL = {
    "nnU-Net test-time mirroring": {"mirroring vs none, first run, all": (I("nnunet"), I("nnunet_nomirror"))},
    "encoder (video-selected)": {"ResNet34 vs ConvNeXt, all": (I("unet_r34"), I("unet_cnx"))},
    "whole-box SAM vs primary, IoU, video": {b: (I("unet_primary", key="iou", sub=V), I(b, key="iou", sub=V))
                                             for b in ("sam_zs_box", "sam_lora_box", "sam3_box")},
    "whole-box SAM vs primary, claw-instance IoU, video": {b: (I("unet_primary", sub=V), I(b, sub=V))
                                                           for b in ("sam_zs_box", "sam_lora_box", "sam3_box")},
    "detectors at threshold 0.01 vs primary, video": {b: (I("unet_primary", sub=V), I(b, sub=V))
                                                      for b in ("yolo_roi_c001", "rfdetr_roi_c001")},
    "nnU-Net vs detectors, video": {b: (I("nnunet_avg", sub=V), I(b, sub=V)) for b in ("yolo_roi", "rfdetr_roi")},
    "checkpoint selection": {"photos+video vs photos, video": (I("full", sub=V), I("photoval", sub=V))},
    "augmentation recipe": {"flip-only vs full recipe, video": (I("fliponly", sub=V), I("photoval", sub=V)),
                            "flip-only U-Net vs RF-DETR-Seg, video": (I("fliponly", sub=V), I("rfdetr_roi", sub=V))},
    "gap class": {"two vs three classes, claw-instance IoU, video": (I("nogap", sub=V), I("full", sub=V)),
                  "two vs three classes, gap IoU, video": (I("nogap", key="gap_iou", sub=V), I("full", key="gap_iou", sub=V))},
    "ImageNet pre-training": {"random vs ImageNet, all": (I("scratch"), I("primary")),
                              "random vs ImageNet, video": (I("scratch", sub=V), I("primary", sub=V)),
                              "random encoder vs nnU-Net, all": (I("scratch"), I("nnunet_avg")),
                              "random encoder vs nnU-Net, video": (I("scratch", sub=V), I("nnunet_avg", sub=V))},
    "decoder controls, video": {"DeepLabV3+ vs primary": (I("dlv3p", sub=V), I("primary", sub=V)),
                                "SegFormer-B2 vs primary": (I("segformer", sub=V), I("primary", sub=V))},
}
found = {r["hoof_id"] for r in EXT_DET if f(r["det_iou"]) > 0}
PE = E("primary")
EXTERNAL = {
    "detectors at lower thresholds vs primary": {b: (PE, E(b)) for b in
                                                 ("yolo_roi_c010", "yolo_roi_c005", "yolo_roi_c001",
                                                  "rfdetr_roi_c010", "rfdetr_roi_c005", "rfdetr_roi_c001")},
    "ImageNet pre-training": {"random encoder vs nnU-Net": (E("scratch"), E("nnunet_avg_ext")),
                              "random vs ImageNet encoder (primary)": (E("scratch"), PE)},
    "augmentation recipe": {"flip-only vs full recipe, photo selection": (E("fliponly"), E("photoval"))},
    "encoder (video-selected)": {"ResNet34 vs ConvNeXt": (E("unet_r34"), E("unet_cnx"))},
    "detector boxes": {"primary, detector vs annotated box": (
        {h: PE[h] for h in found}, ph(EXT_DET, "primary", hoof="hoof_id"))},
    "checkpoint selection": {"video-selected (unet_cnx) vs photos, full recipe": (E("unet_cnx"), E("photoval")),
                             "photos+video vs photos, full recipe": (E("full"), E("photoval"))},
}

out = {"internal": {}, "external": {}}
for fam, tests in INTERNAL.items():
    raw = {k: p(a, b) for k, (a, b) in tests.items()}
    out["internal"][fam] = {k: {"p": raw[k], "p_holm": h} for k, h in zip(raw, holm(list(raw.values())))}
for fam, tests in EXTERNAL.items():
    raw = {k: p(a, b) for k, (a, b) in tests.items()}
    rawf = {k: p(frames(a), frames(b)) for k, (a, b) in tests.items()}
    out["external"][fam] = {k: {"p": raw[k], "p_holm": h, "p_frame": rawf[k], "p_frame_holm": hf}
                            for k, h, hf in zip(raw, holm(list(raw.values())), holm(list(rawf.values())))}
(R / "secondary_tests.json").write_text(json.dumps(out, indent=1))
for s in out:
    for fam, tests in out[s].items():
        print(f"[{s}] {fam}")
        for k, v in tests.items():
            print("   " + f"{k:52s} " + "  ".join(f"{m} {x:.3f}" for m, x in v.items()))
