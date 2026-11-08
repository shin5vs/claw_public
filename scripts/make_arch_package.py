"""Builds the Colab package for the decoder controls (DeepLabV3+ on the ConvNeXt-Tiny encoder and SegFormer-B2, trained
with the recipe of the primary U-Net by train_variants.py --arch): the clawseg package, the training scripts and the
ClawSet dataset (train/valid/test; the test split is only read by the source-overlap check of train_variants.py).

    python scripts/make_arch_package.py      # writes arch_package.zip next to the repository
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REL / "scripts"))
import paths  # noqa: E402

ZIP = REL.parent / "arch_package.zip"

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copytree(REL / "clawseg", out / "clawseg", ignore=shutil.ignore_patterns("__pycache__"))
        (out / "scripts").mkdir()
        for f in ("paths.py", "train_unet.py", "train_variants.py"):
            shutil.copy(REL / "scripts" / f, out / "scripts" / f)
        for split in ("train", "valid", "test"):
            shutil.copytree(paths.DATA / split, out / "data/ClawSet" / split)
        with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out))
    print(ZIP, round(ZIP.stat().st_size / 1e6, 1), "MB")
