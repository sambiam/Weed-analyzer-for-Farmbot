from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field


class WeedSettings(BaseModel):
    """User-calibrated weed detection settings, disabled by default."""

    enabled: bool = False
    automatic_creation: bool = False
    automatic_radius_adjustment: bool = False
    radius_adjustment_confidence: float = Field(default=0.55, ge=0, le=1)
    automatic_removal: bool = False
    removal_confidence: float = Field(default=0.6, ge=0, le=1)
    removal_min_consecutive_absent: int = Field(default=1, ge=1, le=10)
    minimum_area_mm2: float = Field(default=75, ge=5, le=10_000)
    maximum_area_mm2: float = Field(default=2_500, ge=10, le=100_000)
    plant_exclusion_margin_mm: float = Field(default=35, ge=0, le=500)
    crop_protection_enabled: bool = True
    crop_support_radius_multiplier: float = Field(default=1.2, ge=0.5, le=5)
    crop_support_extra_mm: float = Field(default=25, ge=0, le=500)
    shape_filter_enabled: bool = True
    green_hue_min: int = Field(default=25, ge=0, le=179)
    green_hue_max: int = Field(default=100, ge=0, le=179)
    strong_green_minimum_saturation: int = Field(default=45, ge=0, le=255)
    strong_green_minimum_excess_green: int = Field(default=20, ge=-255, le=510)
    minimum_green_purity: float = Field(default=0.45, ge=0, le=1)
    minimum_solidity: float = Field(default=0.25, ge=0, le=1)
    minimum_circularity: float = Field(default=0.03, ge=0, le=1)
    maximum_aspect_ratio: float = Field(default=7, ge=1, le=50)
    minimum_confidence: float = Field(default=0.70, ge=0, le=1)
    automatic_creation_confidence: float = Field(default=0.90, ge=0, le=1)
    temporal_confirmation_enabled: bool = True
    recommendation_min_observations: int = Field(default=1, ge=1, le=20)
    automatic_min_observations: int = Field(default=3, ge=1, le=20)
    temporal_match_distance_mm: float = Field(default=25, ge=1, le=250)
    temporal_max_gap_hours: int = Field(default=168, ge=1, le=8_760)
    visual_verifier_enabled: bool = False
    visual_verifier_shadow_mode: bool = True
    visual_verifier_required_for_automatic: bool = True
    visual_verifier_minimum_confidence: float = Field(default=0.85, ge=0, le=1)
    # Applied to the three shape gates only while the verifier is scoring. The
    # heuristic's job then is recall -- finding every candidate worth judging --
    # and the verifier decides. Set to 1 to keep the gates identical either way.
    candidate_recall_boost: float = Field(default=0.6, ge=0.1, le=1)
    training_minimum_per_class: int = Field(default=10, ge=2, le=10_000)
    automatic_retraining: bool = False
    retrain_after_label_count: int = Field(default=1, ge=1, le=500)
    candidate_crop_storage_enabled: bool = True
    weed_radius_mm: float = Field(default=15, ge=1, le=250)


class WeedSettingsStore:
    """Small atomic JSON store under /data so settings survive container restarts."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> WeedSettings:
        if not self.path.exists():
            return WeedSettings()
        try:
            return WeedSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return WeedSettings()

    def save(self, values: WeedSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            handle.write(values.model_dump_json(indent=2))
            temp = Path(handle.name)
        os.replace(temp, self.path)
