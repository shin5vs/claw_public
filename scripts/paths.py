"""Locations of data, weights and results. Every path can be overridden by an environment variable.

    CLAWSEG_DATA      dataset root in YOLO-seg format with train/valid/test splits (default: data/ClawSet)
    CLAWSEG_EXTERNAL  external test set in the same format (default: data/ClawTest)
    CLAWSEG_WEIGHTS   model weights (default: weights), layout described in README.md
    CLAWSEG_RESULTS   output folder for results (default: results)
    CLAWSEG_NNUNET    nnU-Net raw/preprocessed/results folders (default: nnunet)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # makes the clawseg package importable from the scripts


def _env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


DATA = _env("CLAWSEG_DATA", ROOT / "data" / "ClawSet")
EXTERNAL = _env("CLAWSEG_EXTERNAL", ROOT / "data" / "ClawTest")
WEIGHTS = _env("CLAWSEG_WEIGHTS", ROOT / "weights")
RESULTS = _env("CLAWSEG_RESULTS", ROOT / "results")
NNUNET = _env("CLAWSEG_NNUNET", ROOT / "nnunet")
