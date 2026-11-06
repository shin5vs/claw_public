"""Builds the Colab package for retraining YOLO26s-seg and fine-tuning SAM 2.1 on exactly the training data of the
primary U-Net (data-matched baselines).

Training set: the ClawSet training split without the frame of the internal test recording (123 images, 38 from video).
Validation: the six validation photographs (as for all benchmark models).

  yolo_matched/  the 156 hoof windows of the U-Net (same windows as train_unet.load_split), YOLO-seg polygons per claw
  sam_matched/   the full training images with one COCO polygon per claw (Roboflow layout), as finetune_sam2.py expects

    python scripts/make_matched_package.py      # writes matched_baselines_package.zip next to the repository
"""
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

REL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REL / "scripts"))
import paths  # noqa: E402
from clawseg.segmentation.unet_data import CLAW, read_yolo_polygons  # noqa: E402

spec = importlib.util.spec_from_file_location("train_unet", REL / "scripts/train_unet.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

DATA = paths.DATA
OUT = REL / "matched_package"
EXCLUDE = "20261007_134710-003"


def keep(split, name):
    return not name.startswith(EXCLUDE) if split == "train" else name.startswith("IMG_")


def yolo_windows():
    n = {}
    for split, ysplit in (("train", "train"), ("valid", "val")):
        (OUT / "yolo_matched/images" / ysplit).mkdir(parents=True)
        (OUT / "yolo_matched/labels" / ysplit).mkdir(parents=True)
        samples = [s for s in T.load_split(DATA, split) if keep(split, s["source"])]
        claws = 0
        for k, s in enumerate(samples):
            h, w = s["labels"].shape
            stem = f"{ysplit}_{k:04d}_{s['source']}"
            cv2.imwrite(str(OUT / "yolo_matched/images" / ysplit / f"{stem}.png"), s["image"])
            lines = []
            n_comp, comp = cv2.connectedComponents((s["labels"] == CLAW).astype(np.uint8), connectivity=4)
            for c in range(1, n_comp):
                cnts, _ = cv2.findContours((comp == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(cnts, key=cv2.contourArea)
                if len(cnt) < 3 or cv2.contourArea(cnt) < 4:
                    continue
                pts = cnt.reshape(-1, 2).astype(float) / [w, h]
                lines.append("0 " + " ".join(f"{v:.6f}" for v in pts.ravel()))
                claws += 1
            (OUT / "yolo_matched/labels" / ysplit / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n[ysplit] = (len(samples), claws)
    (OUT / "yolo_matched/data.yaml").write_text("path: yolo_matched\ntrain: images/train\nval: images/val\nnames:\n  0: claw\n")
    return n


def sam_coco():
    n = {}
    for split in ("train", "valid"):
        d = OUT / "sam_matched" / split
        d.mkdir(parents=True)
        images, anns = [], []
        for img_path in sorted((DATA / split / "images").iterdir()):
            name = img_path.name.split(".rf.")[0]
            lbl = DATA / split / "labels" / (img_path.stem + ".txt")
            if not keep(split, name) or not lbl.is_file():
                continue
            image = cv2.imread(str(img_path))
            h, w = image.shape[:2]
            polys = read_yolo_polygons(lbl, w, h)
            if not polys:
                continue
            shutil.copy(img_path, d / img_path.name)
            iid = len(images) + 1
            images.append({"id": iid, "file_name": img_path.name, "width": w, "height": h})
            for p in polys:
                x, y = p[:, 0], p[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": 1, "iscrowd": 0, "area": float(area),
                             "bbox": [float(x.min()), float(y.min()), float(x.max() - x.min()), float(y.max() - y.min())],
                             "segmentation": [[float(v) for v in p.ravel()]]})
        (d / "_annotations.coco.json").write_text(json.dumps(
            {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "claw"}]}))
        n[split] = (len(images), len(anns))
    return n


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    print("YOLO windows (images, claws):", yolo_windows())
    print("SAM COCO (images, claws):", sam_coco())
    (OUT / "scripts").mkdir()
    for f in ("finetune_sam2.py", "sam2_common.py"):
        shutil.copy(REL / "scripts" / f, OUT / "scripts" / f)
    shutil.copytree(REL / "configs", OUT / "configs")
    zp = OUT.parent / "matched_baselines_package.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT))
    print(zp, round(zp.stat().st_size / 1e6, 1), "MB")
