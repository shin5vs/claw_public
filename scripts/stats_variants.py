"""U-Net variants (Table 3 and Section 3.6): means and standard deviations over seeds, and the paired Wilcoxon tests
(uncorrected) on seed-averaged per-hoof values reported in the paper.

    python scripts/stats_variants.py

Reads results/ablation/{per_hoof,jitter,detector_per_hoof}.csv (internal test set), results/benchmark/per_hoof.csv,
results/nnunet_avg/per_hoof.csv (written by stats_internal.py) and results/clawtest/abl/per_hoof.csv (external test set).
Writes results/variants.json.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

R = paths.RESULTS
VARIANTS = ["full", "nogap", "photoval", "fliponly", "boxscale", "scratch"]


def read(p):
    return list(csv.DictReader(open(p))) if Path(p).is_file() else []


def f(x):
    return float(x) if x not in ("", "nan") else np.nan


INT = read(R / "ablation/per_hoof.csv")
JIT = read(R / "ablation/jitter.csv")
DET = read(R / "ablation/detector_per_hoof.csv")
EXT = read(R / "clawtest/abl/per_hoof.csv")
BENCH = read(R / "benchmark/per_hoof.csv") + read(R / "nnunet_avg/per_hoof.csv")


def seeds(v):
    return [f"abl_cnx_{v}_s{s}" for s in range(3)]


def per_hoof(rows, models, key="inst_iou", sub=None, hoof="hoof"):
    d = defaultdict(list)
    for r in rows:
        if r["model"] in models and (sub is None or r.get("subset") == sub):
            d[r[hoof]].append(f(r[key]))
    return {h: float(np.nanmean(v)) for h, v in d.items()}


def seed_stat(rows, v, fn):
    x = np.array([fn([r for r in rows if r["model"] == m]) for m in seeds(v)])
    return float(x.mean()), float(x.std(ddof=1))


def test(a, b):
    hs = sorted(set(a) & set(b))
    return float(wilcoxon([a[h] for h in hs], [b[h] for h in hs], zero_method="zsplit").pvalue)


mean = lambda key, sub=None: (lambda rs: np.nanmean([f(r[key]) for r in rs if sub is None or r["subset"] == sub]))  # noqa: E731
table = {}
for v in VARIANTS:
    table[v] = {"photographs": seed_stat(INT, v, mean("inst_iou", "close-up")), "video": seed_stat(INT, v, mean("inst_iou", "video")),
                "shifted": seed_stat(JIT, v, lambda rs: np.mean([(f(r["inst_iou_m"]) + f(r["inst_iou_p"])) / 2 for r in rs])),
                "detector": seed_stat(DET, v, lambda rs: np.mean([f(r["inst_iou"]) for r in rs if r["subset"] == "video" and f(r["det_iou"]) > 0])),
                "external": seed_stat(EXT, v, mean("inst_iou")) if EXT else None,
                "single_regions_per_seed": float(np.mean([sum(r["n_pred"] == "1" for r in INT if r["model"] == m) for m in seeds(v)]))}
V = lambda v, **k: per_hoof(INT, seeds(v), **k)  # noqa: E731
tests = {
    "video: photos+video vs photos selection (full recipe)": test(V("full", sub="video"), V("photoval", sub="video")),
    "video: flip-only vs full recipe (photo selection)": test(V("fliponly", sub="video"), V("photoval", sub="video")),
    "video: flip-only U-Net vs RF-DETR-Seg": test(V("fliponly", sub="video"), per_hoof(BENCH, ["rfdetr_roi"], sub="video")),
    "video: two vs three classes, claw-instance IoU": test(V("nogap", sub="video"), V("full", sub="video")),
    "video: two vs three classes, gap IoU": test(V("nogap", key="gap_iou", sub="video"), V("full", key="gap_iou", sub="video")),
    "all: random vs ImageNet encoder": test(V("scratch"), V("boxscale")),
    "video: random vs ImageNet encoder": test(V("scratch", sub="video"), V("boxscale", sub="video")),
    "all: random encoder vs nnU-Net": test(V("scratch"), per_hoof(BENCH, ["nnunet_avg"])),
    "video: random encoder vs nnU-Net": test(V("scratch", sub="video"), per_hoof(BENCH, ["nnunet_avg"], sub="video")),
}
(R / "variants.json").write_text(json.dumps({"table": table, "tests_uncorrected": tests}, indent=1))
for v, t in table.items():
    print(f"{v:9s} " + " ".join(f"{k} {m:.3f}±{s:.3f}" for k, ms in t.items() if isinstance(ms, tuple) for m, s in [ms])
          + f" single regions/seed {t['single_regions_per_seed']:.1f}")
for k, p in tests.items():
    print(f"P={p:.3f}  {k}")
