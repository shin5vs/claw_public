"""Safety helpers: file-name sanitising and weight-file checks."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .exceptions import InputValidationError, ModelLoadError

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str, max_len: int = 100, default: str = "image") -> str:
    """Turns arbitrary text (e.g. an animal ID) into an identifier that is safe in file and folder names."""
    return (_UNSAFE_CHARS.sub("_", name).strip("._") or default)[:max_len]


def safe_stem(path: Path, max_len: int = 100) -> str:
    """Turns an input file name into an identifier that is safe in output file names."""
    return safe_name(path.stem, max_len)


def ensure_inside(path: Path, root: Path) -> Path:
    """Before writing: checks that the resolved path stays inside `root` (path traversal protection)."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise InputValidationError(f"The output path leaves the output folder: {path}")
    return resolved


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise ModelLoadError(f"{what} not found: {path}")
    return path


def verify_sha256(path: Path, expected: str | None) -> None:
    """If an expected hash is given, verifies the weight file (protection against tampered checkpoints)."""
    if not expected:
        return
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected.lower():
        raise ModelLoadError(f"SHA-256 mismatch: {path} (expected {expected}, got {digest.hexdigest()})")
