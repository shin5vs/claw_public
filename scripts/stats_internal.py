"""Builds the results of the primary ConvNeXt-U-Net: the three seeds of cnx_boxscale (trained without any image of the
test recording, checkpoint selected on photograph validation images only, windows from enlarged claw boxes, 40 epochs),
averaged per hoof over the seeds, and compares it with the benchmark models.

    python scripts/stats_internal.py

Writes results/primary/{per_hoof,summary,jitter,resolution,detector}.csv and stats.json (Holm-corrected Wilcoxon tests
of the primary model against the seven benchmark configurations used in the comparisons).

nnU-Net is averaged the same way over its four 100-epoch runs (one local run, results/nnunet_nomirror, and three runs
trained on Colab, results/nnunet_colab; same plans and split, different random initialisation, all evaluated without
mirroring) into results/nnunet_avg/ as model "nnunet_avg".
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

R = paths.RESULTS
OUT = R / "primary"
OUT.mkdir(exist_ok=True)
SEEDS = [f"abl_cnx_boxscale_s{s}" for s in range(3)]
NAME = "unet_primary"
BASE = ["yolo_roi", "rfdetr_roi", "sam_zs_halves", "sam_lora_halves", "sam3_text_full", "sam3_halves", "nnunet_avg"]
METRICS = ["iou", "dice", "biou", "hd95_rel", "hd95_px", "inst_iou", "separated", "gap_iou", "ms", "n_pred"]


def read(p):
    return list(csv.DictReader(open(p)))


def f(x):
    return float(x) if x not in ("", "nan") else np.nan


def seed_average(rows, key_cols, value_cols, seeds=None, name=NAME):
    seeds = SEEDS if seeds is None else seeds
    groups = defaultdict(list)
    for r in rows:
        if r["model"] in seeds:
            groups[tuple(r[k] for k in key_cols)].append(r)
    out = []
    for key, rs in groups.items():
        assert len(rs) == len(seeds), (key, len(rs))
        rec = {"model": name, **dict(zip(key_cols, key))}
        for c in value_cols:
            vals = [f(r[c]) for r in rs]
            rec[c] = np.nanmean(vals) if not all(np.isnan(vals)) else np.nan
        out.append(rec)
    return out


def write(path, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def bootstrap(x, n=10000, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)])
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, x.size)).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(ps):
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def merge(seeds, name, out, src):
    """Seed-averages per hoof and writes per_hoof, summary, jitter, resolution and detector CSVs to out."""
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda key: [r for p in src[key] for r in read(R / p)]  # noqa: E731
    per_hoof = seed_average(rd("per_hoof"), ["hoof", "subset", "hoof_h", "n_gt"], METRICS, seeds, name)
    write(out / "per_hoof.csv", per_hoof)
    summary = []
    for sub in ("all", "close-up", "video"):
        sel = [r for r in per_hoof if sub == "all" or r["subset"] == sub]
        rec = {"model": name, "subset": sub, "n": len(sel)}
        for k in METRICS:
            vals = np.array([r[k] for r in sel], float)
            lo, hi = bootstrap(vals)
            rec |= {k: np.nanmean(vals), f"{k}_lo": lo, f"{k}_hi": hi}
        summary.append(rec)
    write(out / "summary.csv", summary)
    jit = rd("jitter")
    jrows = seed_average(jit, ["hoof", "subset", "n_gt"], ["inst_iou_m", "inst_iou_p", "iou_m", "iou_p", "follow"], seeds, name)
    write(out / "jitter.csv", jrows)
    follow = [float(np.nanmedian([f(r["follow"]) for r in jit if r["model"] == s])) for s in seeds]
    if "resolution" in src:
        write(out / "resolution.csv", seed_average(rd("resolution"), ["hoof_id", "level_px", "jpeg_q", "native_px"],
                                                   ["iou", "inst_iou", "separated", "n_pred"], seeds, name))
    write(out / "detector.csv", seed_average(rd("detector"), ["hoof_id", "subset", "det_iou"],
                                             ["iou", "inst_iou", "separated", "n_pred"], seeds, name))
    return per_hoof, jrows, follow


NN_SEEDS = ["nnunet_nomirror"] + [f"nnunetc:nnUNetTrainer_100epochs:{k}:nomirror" for k in range(3)]
merge(NN_SEEDS, "nnunet_avg", R / "nnunet_avg",
      {"per_hoof": ["nnunet_nomirror/per_hoof.csv", "nnunet_colab/per_hoof.csv"],
       "jitter": ["nnunet_nomirror/jitter.csv", "nnunet_colab/jitter.csv"],
       "resolution": ["nnunet_nomirror/resolution_per_sample.csv", "nnunet_colab/resolution_per_sample.csv"],
       "detector": ["nnunet_nomirror/detector_per_hoof.csv", "nnunet_colab/detector_per_hoof.csv"]})
per_hoof, jrows, follow_seed = merge(SEEDS, NAME, OUT, {"per_hoof": ["ablation/per_hoof.csv"], "jitter": ["ablation/jitter.csv"],
                                                       "resolution": ["ablation/resolution_per_sample.csv"],
                                                       "detector": ["ablation/detector_per_hoof.csv"]})

# decoder controls (DeepLabV3+ on the ConvNeXt-Tiny encoder, SegFormer-B2), seed-averaged the same way into
# results/arch_avg/<name>/ as models dlv3p_avg and segformer_avg (Table 2)
for prefix, cname in (("abl_dlv3p_cnx", "dlv3p_avg"), ("abl_segformer_b2", "segformer_avg")):
    merge([f"{prefix}_s{s}" for s in range(3)], cname, R / "arch_avg" / cname,
          {"per_hoof": ["ablation/per_hoof.csv"], "jitter": ["ablation/jitter.csv"], "detector": ["ablation/detector_per_hoof.csv"],
           "resolution": ["ablation/resolution_per_sample.csv"]})

# tests against the benchmark models
B = read(R / "benchmark/per_hoof.csv") + read(R / "nnunet_avg/per_hoof.csv")
JB = read(R / "benchmark/jitter.csv") + read(R / "nnunet_avg/jitter.csv")
stats = {"follow_median_per_seed": follow_seed}
for sub in ("all", "close-up", "video"):
    for met in ("inst_iou", "iou", "biou", "hd95_rel", "shifted"):
        def vec(model):
            if met == "shifted":
                src = jrows if model == NAME else [r for r in JB if r["model"] == model]
                d = {r["hoof"]: (f(r["inst_iou_m"]) + f(r["inst_iou_p"])) / 2 for r in src if sub == "all" or r["subset"] == sub}
            else:
                src = per_hoof if model == NAME else [r for r in B if r["model"] == model]
                d = {r["hoof"]: f(r[met]) for r in src if sub == "all" or r["subset"] == sub}
            return d
        a = vec(NAME)
        hs = sorted(a)
        ps, means = [], {}
        for b in BASE:
            bb = vec(b)
            x, y = np.array([a[h] for h in hs]), np.array([bb[h] for h in hs])
            ps.append(wilcoxon(x, y, zero_method="zsplit").pvalue if np.any(x != y) else 1.0)
            means[b] = float(np.mean(y))
        stats[f"{sub}|{met}"] = {"primary": float(np.mean([a[h] for h in hs])),
                                 **{b: {"mean": means[b], "p_holm": float(p)} for b, p in zip(BASE, holm(np.array(ps)))}}
(OUT / "stats.json").write_text(json.dumps(stats, indent=1))
for k, v in stats.items():
    if k.startswith("follow"):
        print(k, np.round(v, 3))
        continue
    print(f"{k:20s} primary {v['primary']:.3f} | " + " ".join(f"{b}:{v[b]['mean']:.3f}/p{v[b]['p_holm']:.3f}" for b in BASE))
