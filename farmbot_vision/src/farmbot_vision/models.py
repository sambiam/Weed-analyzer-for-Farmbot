"""Typed integration contract and internal domain models."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .resolution import MAX_PROCESSED_HEIGHT, MAX_PROCESSED_WIDTH

# Relative tolerance used when checking that returned resize scales agree with
# the returned pixel dimensions and with each other (isotropic scaling).
_SCALE_TOLERANCE = 0.03


def _is_finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bot(StrictModel):
    config_entry_id: str
    device_id: str
    name: str


class BotList(StrictModel):
    bots: list[Bot]


class InventoryRequest(StrictModel):
    config_entry_id: str
    image_lookback_hours: int = Field(default=72, ge=1, le=720)


class Plant(StrictModel):
    id: int
    name: str
    openfarm_slug: str
    x: float
    y: float
    z: float = 0
    radius: float = Field(ge=0)
    plant_stage: str
    planted_at: datetime | None = None
    spread_curve_id: int | None = None


class WeedPoint(StrictModel):
    """A FarmBot ``Weed`` point returned alongside the plant inventory.

    Weeds are separate from :class:`Plant` in FarmBot (they are map points of
    type ``Weed`` rather than plants), so they are carried on their own list.
    ``name`` is optional because FarmBot weeds are often unnamed. The list
    defaults to empty on :class:`Inventory`, so a companion integration that
    does not yet emit ``weeds`` still validates.
    """

    id: int
    name: str | None = None
    x: float
    y: float
    z: float = 0
    radius: float = Field(default=0, ge=0)


class ImageMeta(StrictModel):
    x: float
    y: float
    z: float = 0
    name: str | None = None


class InventoryImage(StrictModel):
    """An image entry from ``farmbot.get_vision_inventory``.

    The documented contract nests coordinates under ``meta`` and always sends
    ``processed``. At least one companion integration build in the wild
    instead places ``x``/``y``/``z``/``name`` directly on the image object and
    omits ``processed`` entirely. ``_normalize`` accepts both shapes rather
    than rejecting every image in an otherwise-valid inventory response.
    """

    id: int
    created_at: datetime
    processed: bool = True
    meta: ImageMeta

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if not isinstance(data, dict) or "meta" in data:
            return data
        flat_keys = {"x", "y", "z", "name"} & data.keys()
        if not flat_keys:
            return data
        data = dict(data)
        data["meta"] = {key: data.pop(key) for key in flat_keys}
        return data


class CurveData(StrictModel):
    id: int
    name: str
    type: Literal["spread"]
    data: dict[str, float]


class CameraCalibration(StrictModel):
    """Reference (normalized) calibration supplied with the inventory.

    ``pixels_per_mm_*`` are expressed relative to ``reference_width`` x
    ``reference_height`` (the resolution FarmBot calibrated against). To use
    it for a resized processed image the scales must be transformed to the
    processed resolution -- never applied directly (see
    ``calibration.scale_reference_to_processed``).
    """

    available: bool
    pixels_per_mm_x: float | None = Field(default=None, gt=0)
    pixels_per_mm_y: float | None = Field(default=None, gt=0)
    rotation_degrees: float | None = None
    offset_x_mm: float | None = None
    offset_y_mm: float | None = None
    reference_width: int | None = Field(default=None, ge=1)
    reference_height: int | None = Field(default=None, ge=1)
    basis: Literal["reference_image", "native_frame"] | None = None

    @field_validator("basis", mode="before")
    @classmethod
    def _tolerate_unknown_basis(cls, value: object) -> object:
        # ``basis`` here is informational only -- calibration.py never reads
        # it (unlike ProcessedCalibration.basis, which is load-bearing) -- so
        # a value from a companion integration build that doesn't match the
        # two known literals degrades to "unknown" instead of failing the
        # whole inventory response.
        if value not in ("reference_image", "native_frame", None):
            return None
        return value

    @model_validator(mode="after")
    def complete_when_available(self) -> CameraCalibration:
        if self.available and (self.pixels_per_mm_x is None or self.pixels_per_mm_y is None):
            raise ValueError("available calibration requires both pixel scales")
        return self

    @property
    def has_reference_dimensions(self) -> bool:
        return self.reference_width is not None and self.reference_height is not None


class ProcessedCalibration(StrictModel):
    """Calibration that already corresponds to the exact processed pixels.

    Preferred over reference calibration because no transformation is needed:
    the integration computed it for the returned image, so ``basis`` must be
    ``processed_image`` and ``width``/``height`` must match the returned image.
    """

    available: bool
    pixels_per_mm_x: float | None = Field(default=None, gt=0)
    pixels_per_mm_y: float | None = Field(default=None, gt=0)
    rotation_degrees: float = 0.0
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    basis: Literal["processed_image"] | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def complete_when_available(self) -> ProcessedCalibration:
        if not self.available:
            return self
        if not (
            _is_finite_positive(self.pixels_per_mm_x) and _is_finite_positive(self.pixels_per_mm_y)
        ):
            raise ValueError("processed_calibration requires positive finite pixel scales")
        if self.basis != "processed_image":
            raise ValueError("processed_calibration basis must be 'processed_image'")
        if self.width is None or self.height is None:
            raise ValueError("processed_calibration requires width and height")
        return self


class Inventory(StrictModel):
    device_id: str
    generated_at: datetime
    plants: list[Plant]
    images: list[InventoryImage]
    curves: list[CurveData]
    camera_calibration: CameraCalibration
    # FarmBot ``Weed`` points. Optional for backward compatibility: a companion
    # integration that predates the weed contract simply omits it.
    weeds: list[WeedPoint] = Field(default_factory=list)


class VisionImageRequest(StrictModel):
    config_entry_id: str
    image_id: int
    max_width: int = Field(default=960, ge=1, le=MAX_PROCESSED_WIDTH)
    max_height: int = Field(default=720, ge=1, le=MAX_PROCESSED_HEIGHT)


class VisionImageMeta(StrictModel):
    x: float
    y: float
    z: float = 0
    created_at: datetime


class VisionImage(StrictModel):
    """Processed image returned by the integration.

    The ``source_*``/``oriented_*``/``resize_scale_*`` fields are the contract
    v2 additions. When every one of them is present the response is validated
    for dimensional and scaling consistency. When none are present the
    response is treated as a legacy (v1) image via ``full_metadata`` -- callers
    then refuse metric calibration rather than invent missing scaling data.
    """

    image_id: int
    content_type: Literal["image/jpeg"]
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    # Optional: the original bytes are never sent, so this is a format check only.
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    source_width: int | None = Field(default=None, ge=1)
    source_height: int | None = Field(default=None, ge=1)
    oriented_width: int | None = Field(default=None, ge=1)
    oriented_height: int | None = Field(default=None, ge=1)
    width: int = Field(ge=1, le=MAX_PROCESSED_WIDTH)
    height: int = Field(ge=1, le=MAX_PROCESSED_HEIGHT)
    resize_scale_x: float | None = Field(default=None, gt=0)
    resize_scale_y: float | None = Field(default=None, gt=0)
    image_base64: str
    meta: VisionImageMeta
    processed_calibration: ProcessedCalibration | None = None

    @property
    def _v2_fields(self) -> tuple[object, ...]:
        return (
            self.source_width,
            self.source_height,
            self.oriented_width,
            self.oriented_height,
            self.resize_scale_x,
            self.resize_scale_y,
        )

    @property
    def full_metadata(self) -> bool:
        """True when the complete contract-v2 dimension/scale set is present."""
        return all(value is not None for value in self._v2_fields)

    @model_validator(mode="after")
    def _validate_dimensions(self) -> VisionImage:
        present = [value is not None for value in self._v2_fields]
        if not any(present):
            # Legacy v1 image: no scaling metadata to check.
            return self
        if not all(present):
            raise ValueError(
                "incomplete image contract metadata: provide the full v2 dimension "
                "and resize-scale set or none of it"
            )
        # From here every v2 field is present.
        if not (
            _is_finite_positive(self.resize_scale_x) and _is_finite_positive(self.resize_scale_y)
        ):
            raise ValueError("resize scales must be finite and greater than zero")
        # EXIF orientation: oriented dimensions are the source, possibly transposed.
        if {self.source_width, self.source_height} != {self.oriented_width, self.oriented_height}:
            raise ValueError("oriented dimensions must be a rotation of the source dimensions")
        # No unexpected upscaling: processed never larger than the oriented image.
        if self.width > self.oriented_width or self.height > self.oriented_height:
            raise ValueError("processed image is larger than the source (unexpected upscaling)")
        # Scales must agree with the returned dimensions (approx width/oriented_width).
        expected_x = self.width / self.oriented_width
        expected_y = self.height / self.oriented_height
        if abs(self.resize_scale_x - expected_x) > _SCALE_TOLERANCE * expected_x + 1e-6:
            raise ValueError("resize_scale_x is inconsistent with width / oriented_width")
        if abs(self.resize_scale_y - expected_y) > _SCALE_TOLERANCE * expected_y + 1e-6:
            raise ValueError("resize_scale_y is inconsistent with height / oriented_height")
        # Aspect ratio must not be distorted (isotropic scaling).
        larger = max(self.resize_scale_x, self.resize_scale_y)
        if abs(self.resize_scale_x - self.resize_scale_y) > _SCALE_TOLERANCE * larger:
            raise ValueError("aspect ratio distorted: horizontal and vertical scales differ")
        return self


class ApplyRadiusRequest(StrictModel):
    config_entry_id: str
    plant_id: int
    measurement_id: UUID
    expected_current_radius_mm: float = Field(ge=0)
    recommended_radius_mm: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    apply: bool = False
    human_approved: bool = False


class ApplyRemovalRequest(StrictModel):
    config_entry_id: str
    plant_id: int
    measurement_id: UUID
    expected_current_radius_mm: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    apply: bool = False
    human_approved: bool = False


class UpsertCurveRequest(StrictModel):
    config_entry_id: str
    crop_slug: str
    curve_id: int | None = None
    name: str
    data: dict[str, float]
    assign_to_plant_ids: list[int]
    apply: bool = False
    human_approved: bool = False

    @model_validator(mode="after")
    def vision_owned_name(self) -> UpsertCurveRequest:
        if self.curve_id is None and not self.name.startswith("[FarmBot Vision]"):
            raise ValueError("new curves must use the FarmBot Vision prefix")
        return self


class VisionStatus(StrictModel):
    config_entry_id: str
    available: bool
    status: Literal["idle", "running", "warning", "error"]
    job_id: UUID | None = None
    last_completed_at: datetime | None = None
    plants_analysed: int = Field(ge=0)
    recommendations: int = Field(ge=0)
    automatically_applied: int = Field(ge=0)
    uncertain: int = Field(ge=0)
    message: str = Field(max_length=240)
    app_version: str | None = None


class VisionRequestEvent(StrictModel):
    """A request emitted by the companion FarmBot Home Assistant integration.

    An empty ``plant_ids`` list means that all eligible plants should be
    considered. ``device_id`` was added by the companion integration but is
    optional so older event producers remain compatible.
    """

    config_entry_id: str
    device_id: str | None = None
    plant_ids: list[Annotated[int, Field(gt=0, strict=True)]] = Field(default_factory=list)
    # Manual requests specify a mode. Automatic new-photo requests omit it so
    # the app's configured operating mode remains the source of truth.
    mode: Literal["observe", "recommend", "auto_radius"] | None = None
    image_id: int | None = Field(default=None, gt=0, strict=True)


class OperatingMode(StrEnum):
    OBSERVE = "observe"
    RECOMMEND = "recommend"
    AUTO_RADIUS = "auto_radius"


class Decision(StrEnum):
    OBSERVED = "observed"
    RECOMMENDED = "recommended"
    APPLIED = "applied"
    RETAIN = "retain"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"
    REMOVED = "removed"
    REMOVAL_RECOMMENDED = "removal_recommended"


class OriginLocation(StrEnum):
    """Which image corner FarmBot treats as the coordinate origin.

    FarmBot's camera calibration exposes this as ``Origin Location in Image``.
    It encodes the *reflection* between garden axes and pixel axes -- something
    a pure rotation cannot represent -- so a camera mounted rotated or mirrored
    still maps garden coordinates onto the right pixels. ``TOP_LEFT`` is the
    identity (garden +x -> right, garden +y -> down) and is the historical
    behaviour, so it is the default for every calibration that predates this
    field. The two independent sign flips below are applied to the rotated,
    scaled pixel offset from the image centre (see ``vision.garden_to_pixel``).
    """

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"

    @property
    def sign_x(self) -> int:
        return -1 if self in (OriginLocation.TOP_RIGHT, OriginLocation.BOTTOM_RIGHT) else 1

    @property
    def sign_y(self) -> int:
        return -1 if self in (OriginLocation.BOTTOM_LEFT, OriginLocation.BOTTOM_RIGHT) else 1


CalibrationSource = Literal[
    "processed_image",
    "reference_scaled",
    "manual",
    "manual_transformed",
    "farmbot",  # legacy value retained for rows written before contract v2
]


class Calibration(StrictModel):
    """Metric calibration that corresponds to the exact processed pixels.

    ``source`` records how it was obtained (preference order in
    ``calibration.resolve_calibration``). The resolution provenance fields are
    recorded with every measurement so a stored radius can always be traced
    back to the pixels and scaling it was derived from.
    """

    version_id: int | None = None
    source: CalibrationSource
    pixels_per_mm_x: float = Field(gt=0)
    pixels_per_mm_y: float = Field(gt=0)
    rotation_degrees: float = 0
    offset_x_mm: float = 0
    offset_y_mm: float = 0
    # Which image corner is the coordinate origin (garden<->pixel reflection).
    # Defaults to TOP_LEFT so every existing calibration keeps its behaviour.
    origin_location: OriginLocation = OriginLocation.TOP_LEFT
    uncertainty_mm: float = Field(default=10, ge=0)
    # Resolution / transform provenance (contract v2).
    analysis_resolution: str | None = None
    image_id: int | None = None
    processed_width: int | None = None
    processed_height: int | None = None
    source_width: int | None = None
    source_height: int | None = None
    oriented_width: int | None = None
    oriented_height: int | None = None
    resize_scale_x: float | None = None
    resize_scale_y: float | None = None
    calibration_version: str | None = None
    basis: str | None = None
    # Manual calibration provenance (contract v2, Part 5).
    point_a_x: float | None = None
    point_a_y: float | None = None
    point_b_x: float | None = None
    point_b_y: float | None = None
    separation_mm: float | None = None
    transformed_from_id: int | None = None


class PlantSeed(StrictModel):
    plant_id: int
    crop_slug: str
    center_px: tuple[float, float]
    current_radius_mm: float = Field(ge=0)
    planted_at: datetime | None = None


class Measurement(StrictModel):
    measurement_id: UUID
    config_entry_id: str | None = None
    plant_id: int
    crop_slug: str
    image_id: int
    image_timestamp: datetime
    current_radius_mm: float
    typical_canopy_radius_mm: float
    maximum_accepted_canopy_radius_mm: float
    recommended_protection_radius_mm: float
    confidence: float = Field(ge=0, le=1)
    decision: Decision
    reason: str
    ambiguous: bool = False
    calibration_version_id: int | None = None
    transform_json: str = "{}"
    algorithm_version: str
    applied: bool = False
    plant_age_days: int | None = None
    mask_path: str | None = None
    overlay_path: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    vegetation_absent: bool = False
    absent_observations: int = Field(default=0, ge=0)
    safety_margin_mm: float = Field(default=0, ge=0)
    calibration_uncertainty_mm: float = Field(default=0, ge=0)
    # Resolution / calibration provenance (contract v2).
    analysis_resolution: str | None = None
    processed_width: int | None = None
    processed_height: int | None = None
    calibration_source: str | None = None
    calibrated: bool = True
    contract_version: str | None = None


class AnalysisResult(StrictModel):
    measurements: list[Measurement]
    mask: bytes | None = None
    ownership_mask: bytes | None = None
    overlay_jpeg: bytes | None = None
    skipped: dict[int, str] = Field(default_factory=dict)
