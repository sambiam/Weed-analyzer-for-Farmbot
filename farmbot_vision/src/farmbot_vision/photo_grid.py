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
# Grid coordinates are rounded to micrometres so a cell's planned X/Y is
# identical every time it is generated, stored or sent, whichever call carries
# it. FarmBot's own positioning is three orders of magnitude coarser than this,
# so the rounding is purely about coordinate stability.
PHOTO_GRID_COORDINATE_DECIMALS = 3
# Hard contract limit, not a tuning knob: the integration's
# `start_vision_grid_repair` service schema declares
# `vol.Length(min=1, max=12)` on `targets`, so Home Assistant rejects an
# oversized call with HTTP 400 during schema validation -- before the handler
# runs, before any status is reported, and with no partial capture. Raising
# this without first raising that schema (and gating on an advertised
# capability, since the add-on and the integration are versioned and updated
# independently) silently loses every coordinate in the batch. See
# docs/integration-contract.md, "one to twelve targets".
PHOTO_GRID_MAX_TARGETS_PER_CALL = 12
PHOTO_GRID_CHUNK_SIZE = PHOTO_GRID_MAX_TARGETS_PER_CALL

# Integration 2.5.0 and later advertise `continuous_photo_grid_capture` and
# accept a whole bed grid in one call. That is what makes the route continuous:
# the integration runs the lighting, the drive into the bed and the drive back
# to the staging position once around the entire route. Chunking the same route
# into twelve-cell calls gave every chunk its own lighting cycle and its own
# round trip out of the bed, cutting rows in half and adding six long diagonal
# journeys to a 77-cell grid.
PHOTO_GRID_CONTINUOUS_CAPABILITY = "continuous_photo_grid_capture"
PHOTO_GRID_CONTINUOUS_MAX_TARGETS = 256


def photo_grid_chunk_size(capabilities: object) -> int:
    """How many cells one `start_vision_grid_repair` call may carry.

    Falls back to the twelve-target legacy cap whenever the loaded integration
    does not advertise continuous capture, because Home Assistant validates the
    service schema before the handler runs: an oversized call is refused with a
    bare HTTP 400 and captures nothing at all.
    """
    supported = isinstance(capabilities, list | tuple | set | frozenset) and (
        PHOTO_GRID_CONTINUOUS_CAPABILITY in capabilities
    )
    return PHOTO_GRID_CONTINUOUS_MAX_TARGETS if supported else PHOTO_GRID_CHUNK_SIZE


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
    # How many cells of the canonical route each service call carries, decided
    # once from the loaded integration's capabilities. Batches are consecutive
    # slices of `targets`; they never renumber, reorder or regenerate it.
    chunk_size: int = Field(default=PHOTO_GRID_CHUNK_SIZE, ge=1)
    # Whether the loaded integration accepts (and echoes back) a per-target
    # index. Older schemas reject unknown target keys, so this stays off for
    # them and frames fall back to coordinate matching.
    indexed_targets: bool = False


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
        # Multiplying the step by the index (rather than accumulating it) keeps
        # the last centre exactly on `last`, and rounding to micrometres makes
        # every cell's coordinate byte-identical however often it is planned,
        # persisted, reloaded or resent.
        optical_centres = [first + step * index for index in range(count)]
    return [
        round(min(upper, max(lower, centre + gantry_offset)), PHOTO_GRID_COORDINATE_DECIMALS)
        for centre in optical_centres
    ]


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
    """Accept only uploaded frames the integration tied back to a target.

    A frame carrying a ``target_index`` (integration 2.5.0 and later) is
    credited to exactly that cell: the integration already confirmed the
    gantry's position and the uploaded image's coordinates before returning it,
    and re-deciding that here by proximity is what previously left a verified
    cell looking unverified and got it photographed a second time. Frames
    without an index keep the coordinate matching older integrations need.
    """
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
        declared = raw.get("target_index")
        if declared is not None:
            try:
                target = unmatched.pop(int(declared))
            except (TypeError, ValueError, KeyError):
                continue
            matched.append(
                PhotoGridFrame(
                    target_index=target.index,
                    image_id=image_id,
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                )
            )
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
