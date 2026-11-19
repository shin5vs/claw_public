"""Claw-level analysis of the detection-based models: does a correct claw mask exist among their candidates, and how
confident is it?

For every annotated claw, all raw candidates of YOLO26s-seg and RF-DETR-Seg (window 0.5, confidence >= 0.01, before
the shared selection step) are compared with the claw. Recorded per claw: the best candidate IoU, the confidence of
that candidate, and the number of candidates per hoof. A claw is 'found' when a candidate reaches IoU >= 0.5.

    python scripts/detector_recall.py

Writes results/detector_recall/per_claw.csv and summary.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import paths  # noqa: E402

OUT = paths.RESULTS / "detector_recall"
SETS = {"internal": paths.DATA, "external": paths.EXTERNAL}
MODELS = {"yolo_roi_c001": 0.28, "rfdetr_roi_c001": 0.30}   # model -> operating threshold used in the paper


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    ctx = B.BuildContext(base_dir=paths.ROOT, out_dir=OUT)
    for set_name, root in SETS.items():
        B.DATA = root
        hoofs = B.load_hoofs("test")
        images = {h["path"]: cv2.imread(str(h["path"])) for h in hoofs}
        for name, make in B.build_models(ctx, list(MODELS)).items():
            model = make()
            for hoof in hoofs:
                preds = model.predict(images[hoof["path"]], hoof["box"])
                for k, claw in enumerate(hoof["claws"]):
                    best_iou, best_conf = 0.0, np.nan
                    for m, c in preds:
                        v = B.iou(claw, m)
                        if v > best_iou:
                            best_iou, best_conf = v, c
                    rows.append({"set": set_name, "model": name, "hoof": hoof["id"], "claw": k,
                                 "subset": "photo" if hoof["subset"] == "close-up" else "video",
                                 "hoof_h": hoof["height"], "n_cand": len(preds),
                                 "best_iou": best_iou, "best_conf": best_conf})
            print(set_name, name, "done", flush=True)
            del model
    with open(OUT / "per_claw.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    summary = {}
    for set_name in SETS:
        for name, thr in MODELS.items():
            for subset in ("photo", "video"):
                sel = [r for r in rows if r["set"] == set_name and r["model"] == name and r["subset"] == subset]
                if not sel:
                    continue
                found = [r for r in sel if r["best_iou"] >= 0.5]
                hoof_cands = {r["hoof"]: r["n_cand"] for r in sel}
                summary[f"{set_name}|{name}|{subset}"] = {
                    "n_claws": len(sel),
                    "found_any_conf": len(found) / len(sel),
                    "found_at_operating_thr": sum(r["best_conf"] >= thr for r in found) / len(sel),
                    "median_conf_found": float(np.median([r["best_conf"] for r in found])) if found else None,
                    "median_best_iou": float(np.median([r["best_iou"] for r in sel])),
                    "hoofs_without_candidate": sum(v == 0 for v in hoof_cands.values()) / len(hoof_cands),
                    "median_candidates_per_hoof": float(np.median(list(hoof_cands.values()))),
                }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
