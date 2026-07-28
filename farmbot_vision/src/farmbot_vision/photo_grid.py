"""Plan and persist reliable, calibration-aware FarmBot photo grids."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from .calibration_store import FarmbotCalibrationInput

PHOTO_GRID_OVERLAP = 0.15
PHOTO_GRID_COORDINATE_TOLERANCE_MM = 25.0
PHOTO_GRID_CHUNK_SIZE = 12


class PhotoGridTarget(BaseModel):
    index: int = Field(ge=0)
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    x: float
    y: float
    z: float


class PhotoGridFrame(BaseModel):
    target_index: int = Field(ge=0)
    image_id: int = Field(gt=0)
    x: float
    y: float
    z: float


class PhotoGridRecord(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    config_entry_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "queued"
    message: str = ""
    bed_bounds: dict[str, tuple[float, float]]
    footprint_width_mm: float = Field(gt=0)
    footprint_height_mm: float = Field(gt=0)
    calibration: FarmbotCalibrationInput
    targets: list[PhotoGridTarget] = Field(min_length=1)
    frames: list[PhotoGridFrame] = Field(default_factory=list)


class PhotoGridStore:
    """Atomic storage for the latest grid, including in-progress captures."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> PhotoGridRecord | None:
        if not self.path.exists():
            return None
        try:
            return PhotoGridRecord.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, record: PhotoGridRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False, suffix=".tmp"
        )
        try:
            handle.write(record.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, self.path)


def _capture_centres(
    lower: float,
    upper: float,
    footprint: float,
    gantry_offset: float,
) -> list[float]:
    """Evenly cover one garden axis and return in-bounds gantry positions."""
    if not all(math.isfinite(value) for value in (lower, upper, footprint, gantry_offset)):
        raise ValueError("photo-grid geometry must be finite")
    if upper <= lower:
        raise ValueError("photo-grid axis bounds must have a positive range")
    if footprint <= 0:
        raise ValueError("photo-grid camera footprint must be positive")

    span = upper - lower
    if footprint >= span:
        optical_centres = [(lower + upper) / 2]
    else:
        usable_step = footprint * (1 - PHOTO_GRID_OVERLAP)
        count = max(2, math.ceil((span - footprint) / usable_step) + 1)
        first = lower + footprint / 2
        last = upper - footprint / 2
        step = (last - first) / (count - 1)
        optical_centres = [first + step * index for index in range(count)]
    return [min(upper, max(lower, centre + gantry_offset)) for centre in optical_centres]


def plan_photo_grid(
    calibration: FarmbotCalibrationInput,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z: float,
) -> tuple[list[PhotoGridTarget], float, float]:
    """Create a serpentine grid from FarmBot calibration and motion bounds.

    FarmBot's scale is millimetres per native pixel. Camera rotation expands
    the axis-aligned garden footprint, while calibration offsets translate the
    optical centre relative to the gantry coordinate stored with each photo.
    """
    if not math.isfinite(z):
        raise ValueError("photo-grid Z coordinate must be finite")
    width = calibration.coordinate_scale * calibration.reference_width
    height = calibration.coordinate_scale * calibration.reference_height
    theta = math.radians(calibration.rotation_degrees)
    footprint_x = abs(math.cos(theta)) * width + abs(math.sin(theta)) * height
    footprint_y = abs(math.sin(theta)) * width + abs(math.cos(theta)) * height
    xs = _capture_centres(*x_bounds, footprint_x, calibration.offset_x_mm)
    ys = _capture_centres(*y_bounds, footprint_y, calibration.offset_y_mm)

    targets: list[PhotoGridTarget] = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for column_in_path, x in enumerate(row_xs):
            column = column_in_path if row % 2 == 0 else len(xs) - 1 - column_in_path
            targets.append(
                PhotoGridTarget(
                    index=len(targets),
                    row=row,
                    column=column,
                    x=x,
                    y=y,
                    z=z,
                )
            )
    return targets, footprint_x, footprint_y


def match_verified_frames(
    targets: list[PhotoGridTarget],
    frames: object,
    *,
    tolerance_mm: float = PHOTO_GRID_COORDINATE_TOLERANCE_MM,
) -> list[PhotoGridFrame]:
    """Accept only uploaded frames whose returned coordinates match a target."""
    if not isinstance(frames, list):
        return []
    unmatched = {target.index: target for target in targets}
    matched: list[PhotoGridFrame] = []
    for raw in frames:
        if not isinstance(raw, dict):
            continue
        try:
            image_id = int(raw["image_id"])
            coordinates = tuple(float(raw[key]) for key in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            continue
        candidates = [
            target
            for target in unmatched.values()
            if math.dist(coordinates, (target.x, target.y, target.z)) <= tolerance_mm
        ]
        if not candidates:
            continue
        target = min(
            candidates,
            key=lambda item: math.dist(coordinates, (item.x, item.y, item.z)),
        )
        matched.append(
            PhotoGridFrame(
                target_index=target.index,
                image_id=image_id,
                x=coordinates[0],
                y=coordinates[1],
                z=coordinates[2],
            )
        )
        unmatched.pop(target.index)
    return matched
