"""Exports the ClawSet2 hoof windows as an nnU-Net v2 dataset (Dataset501_AngusClaw, 2D natural images).

Same rules as the clean U-Net runs (cnx_photoval): training = hoof windows of the training split without the frame of the
test recording; validation = hoof windows of the photograph (IMG_*) validation images, fixed as fold 0 through
splits_final.json. Windows are the claw box enlarged by 0.5 per side, resized to 320x320 like the U-Net input, with the
three-class labels (0 background, 1 claw, 2 interdigital gap) of the U-Net.

    python scripts/nnunet_prepare.py
then (CPU):  nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
             cp splits_final.json into $nnUNet_preprocessed/Dataset501_AngusClaw/
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
spec = importlib.util.spec_from_file_location("train_unet", Path(__file__).resolve().parent / "train_unet.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

DATA = paths.DATA
NN = paths.NNUNET
NAME = "Dataset501_AngusClaw"
SIZE = 320
EXCLUDE = "20261007_134710-003"


def main():
    raw = NN / "raw" / NAME
    if raw.exists():
        shutil.rmtree(raw)
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir(parents=True)
    train = [s for s in T.load_split(DATA, "train") if not s["source"].startswith(EXCLUDE)]
    valid = [s for s in T.load_split(DATA, "valid") if s["source"].startswith("IMG_")]
    names = {"train": [], "val": []}
    for split, samples in (("train", train), ("val", valid)):
        for k, s in enumerate(samples):
            case = f"claw_{split}_{k:04d}"
            img = cv2.resize(s["image"], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(s["labels"], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(raw / "imagesTr" / f"{case}_0000.png"), img)      # cv2 writes the BGR array as a normal
            cv2.imwrite(str(raw / "labelsTr" / f"{case}.png"), lab)           # RGB PNG; nnU-Net reads it as R, G, B
            names[split].append(case)
    dataset = {"channel_names": {"0": "R", "1": "G", "2": "B"},
               "labels": {"background": 0, "claw": 1, "gap": 2},
               "numTraining": len(train) + len(valid), "file_ending": ".png",
               "name": NAME, "description": "Angus claw hoof windows (ClawSet2), 320x320, three classes"}
    (raw / "dataset.json").write_text(json.dumps(dataset, indent=2))
    (NN / "splits_final.json").write_text(json.dumps([names], indent=1))
    print(f"{len(train)} training and {len(valid)} validation windows written to {raw}")


if __name__ == "__main__":
    main()
