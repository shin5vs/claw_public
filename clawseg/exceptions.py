"""Exception hierarchy. Callers can catch the common base class HoofPipeError."""
from __future__ import annotations


class HoofPipeError(Exception):
    """Base of all errors raised deliberately by this package."""


class ConfigError(HoofPipeError):
    """Invalid or incomplete configuration."""


class InputValidationError(HoofPipeError):
    """The input (image, path) does not meet the safety or format requirements."""


class ModelLoadError(HoofPipeError):
    """A model weight file cannot be loaded."""


class DetectionError(HoofPipeError):
    """Hoof detection failed."""


class SegmentationError(HoofPipeError):
    """Claw segmentation failed."""


class GeometryError(HoofPipeError):
    """A geometric measurement cannot be made on the given masks."""


class OutputError(HoofPipeError):
    """Writing the result failed."""
