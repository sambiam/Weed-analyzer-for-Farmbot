"""Typed integration contract and internal domain models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ImageMeta(StrictModel):
    x: float
    y: float
    z: float = 0
    name: str | None = None


class InventoryImage(StrictModel):
    id: int
    created_at: datetime
    processed: bool
    meta: ImageMeta


class CurveData(StrictModel):
    id: int
    name: str
    type: Literal["spread"]
    data: dict[str, float]


class CameraCalibration(StrictModel):
    available: bool
    pixels_per_mm_x: float | None = Field(default=None, gt=0)
    pixels_per_mm_y: float | None = Field(default=None, gt=0)
    rotation_degrees: float | None = None
    offset_x_mm: float | None = None
    offset_y_mm: float | None = None

    @model_validator(mode="after")
    def complete_when_available(self) -> CameraCalibration:
        if self.available and (self.pixels_per_mm_x is None or self.pixels_per_mm_y is None):
            raise ValueError("available calibration requires both pixel scales")
        return self


class Inventory(StrictModel):
    device_id: str
    generated_at: datetime
    plants: list[Plant]
    images: list[InventoryImage]
    curves: list[CurveData]
    camera_calibration: CameraCalibration


class VisionImageRequest(StrictModel):
    config_entry_id: str
    image_id: int
    max_width: int = Field(default=640, ge=1, le=640)
    max_height: int = Field(default=480, ge=1, le=480)


class VisionImageMeta(StrictModel):
    x: float
    y: float
    z: float = 0
    created_at: datetime


class VisionImage(StrictModel):
    image_id: int
    content_type: Literal["image/jpeg"]
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    width: int = Field(ge=1, le=640)
    height: int = Field(ge=1, le=480)
    image_base64: str
    meta: VisionImageMeta


class ApplyRadiusRequest(StrictModel):
    config_entry_id: str
    plant_id: int
    measurement_id: UUID
    expected_current_radius_mm: float = Field(ge=0)
    recommended_radius_mm: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    apply: bool = False


class UpsertCurveRequest(StrictModel):
    config_entry_id: str
    crop_slug: str
    curve_id: int | None = None
    name: str
    data: dict[str, float]
    assign_to_plant_ids: list[int]
    apply: bool = False

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


class VisionRequestEvent(StrictModel):
    config_entry_id: str
    plant_ids: list[int] = Field(default_factory=list)
    mode: Literal["observe", "recommend", "auto_radius"]


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


class Calibration(StrictModel):
    version_id: int | None = None
    source: Literal["farmbot", "manual"]
    pixels_per_mm_x: float = Field(gt=0)
    pixels_per_mm_y: float = Field(gt=0)
    rotation_degrees: float = 0
    offset_x_mm: float = 0
    offset_y_mm: float = 0
    uncertainty_mm: float = Field(default=10, ge=0)


class PlantSeed(StrictModel):
    plant_id: int
    crop_slug: str
    center_px: tuple[float, float]
    current_radius_mm: float = Field(ge=0)
    planted_at: datetime | None = None


class Measurement(StrictModel):
    measurement_id: UUID
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


class AnalysisResult(StrictModel):
    measurements: list[Measurement]
    mask: bytes | None = None
    overlay_jpeg: bytes | None = None
    skipped: dict[int, str] = Field(default_factory=dict)
