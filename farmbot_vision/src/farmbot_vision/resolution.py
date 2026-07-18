"""Typed analysis-resolution presets.

FarmBot Vision only ever processes at one of a small, fixed set of
resolutions. Arbitrary dimensions are never accepted from app options or
from an integration response: everything funnels through :class:`Resolution`
so width, height, pixel count and a stable display label are computed in
exactly one place.

The relative pixel workload figures below are used in documentation and in
the health payload so an operator can reason about CPU/memory cost before
selecting a heavier preset. They are expressed relative to 640 x 480.
"""

from __future__ import annotations

from enum import StrEnum

# Hard ceiling for any processed image the app will accept. Deliberately set
# to the largest supported preset so an integration can never push a larger
# frame through the resize/validation path (see models.VisionImage).
MAX_PROCESSED_WIDTH = 1280
MAX_PROCESSED_HEIGHT = 960

# Native FarmBot full-frame size, documented only so the relative workload is
# explicit. It is intentionally NOT a selectable analysis mode.
NATIVE_FRAME = (2592, 1944)


class AnalysisResolution(StrEnum):
    """The only analysis resolutions a user may select."""

    R640X480 = "640x480"
    R960X720 = "960x720"
    R1280X960 = "1280x960"


#: Default preset for new and migrated installations.
DEFAULT_RESOLUTION = AnalysisResolution.R960X720

_DIMENSIONS: dict[AnalysisResolution, tuple[int, int]] = {
    AnalysisResolution.R640X480: (640, 480),
    AnalysisResolution.R960X720: (960, 720),
    AnalysisResolution.R1280X960: (1280, 960),
}


class Resolution:
    """A resolved analysis preset exposing width, height, pixels and label."""

    __slots__ = ("preset", "width", "height")

    def __init__(self, preset: AnalysisResolution):
        self.preset = preset
        self.width, self.height = _DIMENSIONS[preset]

    @classmethod
    def from_value(cls, value: str | AnalysisResolution) -> Resolution:
        """Resolve an app-option string, rejecting anything not allowlisted."""
        try:
            preset = AnalysisResolution(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in AnalysisResolution)
            raise ValueError(
                f"unsupported analysis_resolution {value!r}; allowed values: {allowed}"
            ) from exc
        return cls(preset)

    @property
    def value(self) -> str:
        return self.preset.value

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def relative_workload(self) -> float:
        """Pixel workload relative to the 640 x 480 baseline."""
        baseline = _DIMENSIONS[AnalysisResolution.R640X480]
        return round(self.pixel_count / (baseline[0] * baseline[1]), 2)

    @property
    def label(self) -> str:
        return f"{self.width} x {self.height} ({self.relative_workload}x)"

    def as_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset.value,
            "width": self.width,
            "height": self.height,
            "pixel_count": self.pixel_count,
            "relative_workload": self.relative_workload,
            "label": self.label,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Resolution({self.preset.value})"


def native_relative_workload() -> float:
    """Relative workload of the native 2592 x 1944 frame vs 640 x 480."""
    baseline = _DIMENSIONS[AnalysisResolution.R640X480]
    return round((NATIVE_FRAME[0] * NATIVE_FRAME[1]) / (baseline[0] * baseline[1]), 1)
