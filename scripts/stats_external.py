"""External test set ClawTest: seed/run averages, image-clustered bootstrap CIs and Holm-corrected tests.

ClawTest = 32 frames of 29 bulls from the WB front-view videos (October 2026), 69 two-claw hooves; no frame of these
videos is in any training or validation set. Evaluated by benchmark/run_clawtest.sh (results/clawtest/{bench,abl,abl2},
detector_per_hoof.csv). Hooves of one frame are not independent, so confidence intervals resample frames, and every
hoof-level Wilcoxon test is repeated on frame means.

    python scripts/stats_external.py

Writes results/clawtest/summary.csv (one row per configuration) and stats.json.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

R = paths.RESULTS / "clawtest"
GROUPS = {
    "unet_primary": [f"abl_cnx_boxscale_s{s}" for s in range(3)],
    "nnunet_avg": ["nnunet_nomirror"] + [f"nnunetc:nnUNetTrainer_100epochs:{k}:nomirror" for k in range(3)],
    **{f"var_{v}": [f"abl_cnx_{v}_s{s}" for s in range(3)] for v in ("full", "nogap", "photoval", "fliponly", "boxscale", "scratch")},
}
BASE = ["yolo_roi", "rfdetr_roi", "sam_zs_halves", "sam_lora_halves", "sam3_text_full", "sam3_halves", "nnunet_avg"]
SINGLE = ["unet_cnx", "unet_r34", "yolo_roi", "rfdetr_roi", "sam_zs_box", "sam_zs_halves", "sam_lora_box", "sam_lora_halves",
          "sam3_text_full", "sam3_box", "sam3_halves"]


def read(p):
    return list(csv.DictReader(open(p))) if Path(p).is_file() else []


def f(x):
    return float(x) if x not in ("", "nan") else np.nan


ROWS = read(R / "bench/per_hoof.csv") + read(R / "abl/per_hoof.csv") + read(R / "abl2/per_hoof.csv")
JIT = read(R / "bench/jitter.csv") + read(R / "abl/jitter.csv")
DET = read(R / "detector_per_hoof.csv")


def members(name):
    return GROUPS.get(name, [name])


def per_hoof(name, key="inst_iou", rows=ROWS, hoof_col="hoof"):
    d = defaultdict(list)
    for r in rows:
        if r["model"] in members(name):
            d[r[hoof_col]].append(f(r[key]))
    return {h: float(np.mean(v)) for h, v in d.items()}


def shifted(name):
    d = defaultdict(list)
    for r in JIT:
        if r["model"] in members(name):
            d[r["hoof"]].append((f(r["inst_iou_m"]) + f(r["inst_iou_p"])) / 2)
    return {h: float(np.mean(v)) for h, v in d.items()}


def frame(h):
    return h.split("#")[0]


def frame_means(d):
    g = defaultdict(list)
    for h, v in d.items():
        g[frame(h)].append(v)
    return {k: float(np.mean(v)) for k, v in g.items()}


def cluster_ci(d, n=10000, seed=0):
    g = defaultdict(list)
    for h, v in d.items():
        g[frame(h)].append(v)
    groups = [np.array(v) for v in g.values()]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def holm(ps):
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def paired_p(a, b):
    hs = sorted(set(a) & set(b))
    x, y = np.array([a[h] for h in hs]), np.array([b[h] for h in hs])
    return float(wilcoxon(x, y, zero_method="zsplit").pvalue) if np.any(x != y) else 1.0


found = {r["hoof_id"] for r in DET if f(r["det_iou"]) > 0}
summary = []
for name in ["unet_primary", "nnunet_avg"] + SINGLE + [k for k in GROUPS if k.startswith("var_")]:
    d = per_hoof(name)
    if not d:
        continue
    seed_means = [float(np.mean([f(r["inst_iou"]) for r in ROWS if r["model"] == m])) for m in members(name)]
    det = per_hoof(name, rows=DET, hoof_col="hoof_id")
    sh = shifted(name)
    follow = [float(np.nanmedian([f(r["follow"]) for r in JIT if r["model"] == m])) for m in members(name)
              if any(r["model"] == m for r in JIT)]
    rec = {"model": name, "n": len(d), "inst_iou": np.mean(list(d.values())), "ci": cluster_ci(d),
           "seed_sd": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else np.nan,
           "iou": np.mean(list(per_hoof(name, "iou").values())), "biou": np.mean(list(per_hoof(name, "biou").values())),
           "separated": np.mean(list(per_hoof(name, "separated").values())),
           "single_region": np.mean([f(r["n_pred"]) == 1 for r in ROWS if r["model"] in members(name)]),
           "shifted": np.mean(list(sh.values())) if sh else np.nan,
           "follow": float(np.median(follow)) if follow else np.nan,
           "det_found": np.mean([det[h] for h in found]) if det else np.nan,
           "gt_found": np.mean([d[h] for h in found]),
           "ms": np.nanmean(list(per_hoof(name, "ms").values()))}
    summary.append(rec)
with open(R / "summary.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=[k for k in summary[0] if k != "ci"] + ["ci_lo", "ci_hi"])
    wr.writeheader()
    for rec in summary:
        wr.writerow({**{k: v for k, v in rec.items() if k != "ci"}, "ci_lo": rec["ci"][0], "ci_hi": rec["ci"][1]})

stats = {"n_hooves": len(per_hoof("unet_primary")), "n_frames": len(frame_means(per_hoof("unet_primary"))),
         "detector_found": len(found)}
for ref in ("unet_primary", "unet_cnx", "var_fliponly"):
    for met, get in (("inst_iou", lambda n: per_hoof(n)), ("shifted", shifted), ("iou", lambda n: per_hoof(n, "iou"))):
        a = get(ref)
        if not a:
            continue
        p_h = [paired_p(a, get(b)) for b in BASE]
        p_f = [paired_p(frame_means(a), frame_means(get(b))) for b in BASE]
        stats[f"{ref}|{met}"] = {b: {"mean": float(np.mean(list(get(b).values()))), "p_holm_hoof": float(ph), "p_holm_frame": float(pf)}
                                 for b, ph, pf in zip(BASE, holm(np.array(p_h)), holm(np.array(p_f)))}
# further uncorrected comparisons used in the text
P = per_hoof("unet_primary")
stats["extra"] = {
    "primary_vs_whole_box_inst": {b: paired_p(P, per_hoof(b)) for b in ("sam_zs_box", "sam_lora_box", "sam3_box")},
    "primary_vs_whole_box_iou": {b: paired_p(per_hoof("unet_primary", "iou"), per_hoof(b, "iou")) for b in ("sam_zs_box", "sam_lora_box", "sam3_box")},
    "scratch_vs_nnunet": paired_p(per_hoof("var_scratch"), per_hoof("nnunet_avg")),
    "scratch_vs_primary": paired_p(per_hoof("var_scratch"), P),
    "fliponly_vs_primary": paired_p(per_hoof("var_fliponly"), P),
    "photoval_vs_primary": paired_p(per_hoof("var_photoval"), P),
    "videoselected_vs_primary": paired_p(per_hoof("unet_cnx"), P),
    "videoselected_vs_photoval": paired_p(per_hoof("unet_cnx"), per_hoof("var_photoval")),
    "full_vs_photoval": paired_p(per_hoof("var_full"), per_hoof("var_photoval")),
    "nogap_vs_full": paired_p(per_hoof("var_nogap"), per_hoof("var_full")),
    "nnunet_vs_detectors": {b: paired_p(per_hoof("nnunet_avg"), per_hoof(b)) for b in ("yolo_roi", "rfdetr_roi")},
    "primary_det_vs_gt": paired_p({h: P[h] for h in found}, per_hoof("unet_primary", rows=DET, hoof_col="hoof_id")),
}
(R / "stats.json").write_text(json.dumps(stats, indent=1))
for rec in summary:
    print(f"{rec['model']:16s} {rec['inst_iou']:.3f} ({rec['ci'][0]:.2f}-{rec['ci'][1]:.2f}) sd {rec['seed_sd']:.3f} iou {rec['iou']:.3f} "
          f"sep {rec['separated']:.2f} single {rec['single_region']:.2f} shift {rec['shifted']:.3f} follow {rec['follow']:.2f} "
          f"det {rec['det_found']:.3f}/gt {rec['gt_found']:.3f} ms {rec['ms']:.0f}")
print(json.dumps({k: v for k, v in stats.items() if k in ("extra", "unet_primary|inst_iou", "unet_primary|shifted", "unet_cnx|inst_iou")}, indent=0)[:3000])
