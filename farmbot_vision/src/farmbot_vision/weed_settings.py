from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field, model_validator


class WeedSettings(BaseModel):
    """User-calibrated weed detection settings, disabled by default."""

    enabled: bool = False
    automatic_creation: bool = False
    automatic_radius_adjustment: bool = False
    radius_min_consecutive_present: int = Field(default=2, ge=1, le=10)
    # A single permissive mask must never be allowed to double the radius of a
    # known weed.  Both limits are measured against the first radius recorded
    # in the previous 24 hours; the smaller allowance wins.
    maximum_radius_growth_mm_per_day: float = Field(default=20, ge=0, le=250)
    maximum_radius_growth_percent_per_day: float = Field(default=40, ge=0, le=200)
    automatic_removal: bool = False
    removal_min_consecutive_absent: int = Field(default=2, ge=1, le=10)
    # A first-true-leaf seedling covers roughly 30 mm2, so the old 75 mm2 floor
    # silently discarded every weed worth catching early. Pixel noise is already
    # rejected by the vegetation mask's own area floor.
    minimum_area_mm2: float = Field(default=20, ge=5, le=10_000)
    # 10,000 mm2 is about a 113 mm circular rosette.  The old 2,500 mm2 default
    # was already smaller than a mature 60 mm dandelion once segmentation and
    # nearby leaf grouping were included.
    maximum_area_mm2: float = Field(default=10_000, ge=10, le=100_000)
    plant_exclusion_margin_mm: float = Field(default=35, ge=0, le=500)
    crop_protection_enabled: bool = True
    crop_support_radius_multiplier: float = Field(default=1.2, ge=0.5, le=5)
    crop_support_extra_mm: float = Field(default=25, ge=0, le=500)
    shape_filter_enabled: bool = True
    green_hue_min: int = Field(default=25, ge=0, le=179)
    green_hue_max: int = Field(default=100, ge=0, le=179)
    strong_green_minimum_saturation: int = Field(default=45, ge=0, le=255)
    strong_green_minimum_excess_green: int = Field(default=20, ge=-255, le=510)
    # Candidate discovery is intentionally more permissive than geometry
    # measurement.  It catches pale/shaded leaves for the verifier, while the
    # stricter extent thresholds below keep soil out of the reported radius.
    candidate_minimum_saturation: int = Field(default=15, ge=0, le=255)
    candidate_minimum_excess_green: int = Field(default=2, ge=-255, le=510)
    candidate_hue_padding: int = Field(default=8, ge=0, le=60)
    candidate_grouping_gap_mm: float = Field(default=18, ge=1, le=75)
    candidate_maximum_span_mm: float = Field(default=240, ge=20, le=500)
    extent_minimum_saturation: int = Field(default=28, ge=0, le=255)
    extent_minimum_excess_green: int = Field(default=10, ge=-255, le=510)
    extent_radial_percentile: float = Field(default=97, ge=80, le=100)
    # Measured against real photographs rather than synthetic discs: a genuine
    # weed leaf reaches only ~0.1-0.45 strong-green purity and ~0.2-0.6 solidity
    # once shadow, glare and soil show through the gaps between its leaves. The
    # previous defaults sat above that range and rejected almost every weed
    # before the verifier ever scored it.
    minimum_green_purity: float = Field(default=0.10, ge=0, le=1)
    minimum_solidity: float = Field(default=0.08, ge=0, le=1)
    minimum_circularity: float = Field(default=0.01, ge=0, le=1)
    # Grass and other narrow-leaved weeds are legitimately long and thin.
    maximum_aspect_ratio: float = Field(default=12, ge=1, le=50)
    minimum_confidence: float = Field(default=0.45, ge=0, le=1)
    automatic_creation_confidence: float = Field(default=0.90, ge=0, le=1)
    temporal_confirmation_enabled: bool = True
    recommendation_min_observations: int = Field(default=1, ge=1, le=20)
    automatic_min_observations: int = Field(default=3, ge=1, le=20)
    temporal_match_distance_mm: float = Field(default=25, ge=1, le=250)
    temporal_max_gap_hours: int = Field(default=168, ge=1, le=8_760)
    visual_verifier_enabled: bool = False
    visual_verifier_shadow_mode: bool = True
    # Three-way verifier triage: below rejection is hidden automatically,
    # between the thresholds remains reviewable, and at/above acceptance may
    # authorise automation when the independent automation guards also pass.
    visual_verifier_rejection_confidence: float = Field(default=0.45, ge=0, le=1)
    visual_verifier_acceptance_confidence: float = Field(default=0.85, ge=0, le=1)
    # The pre-validator below accepts the old one-threshold JSON key and copies
    # it to both bounds, preserving behaviour until these controls are saved.
    # The same local model can provide a second opinion on vegetation newly
    # extending a known plant. It never predicts a radius; it only accepts,
    # rejects, or holds the new boundary evidence before geometry measures it.
    boundary_verifier_enabled: bool = True
    boundary_crop_minimum_confidence: float = Field(default=0.60, ge=0, le=1)
    boundary_noncrop_minimum_confidence: float = Field(default=0.80, ge=0, le=1)
    # Applied to the three shape gates only while the verifier is scoring. The
    # heuristic's job then is recall -- finding every candidate worth judging --
    # and the verifier decides. Set to 1 to keep the gates identical either way.
    candidate_recall_boost: float = Field(default=0.6, ge=0.1, le=1)
    training_minimum_per_class: int = Field(default=10, ge=2, le=10_000)
    automatic_retraining: bool = False
    retrain_after_label_count: int = Field(default=1, ge=1, le=500)
    candidate_crop_storage_enabled: bool = True
    weed_radius_mm: float = Field(default=15, ge=1, le=250)

    @model_validator(mode="before")
    @classmethod
    def migrate_verifier_thresholds(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        values = dict(value)
        legacy = values.get("visual_verifier_minimum_confidence")
        if legacy is not None:
            values.setdefault("visual_verifier_rejection_confidence", legacy)
            values.setdefault("visual_verifier_acceptance_confidence", legacy)
        return values

    @model_validator(mode="after")
    def ordered_confidence_thresholds(self) -> WeedSettings:
        if self.visual_verifier_rejection_confidence > self.visual_verifier_acceptance_confidence:
            raise ValueError("Verifier rejection confidence cannot exceed acceptance confidence")
        if self.candidate_minimum_saturation > self.extent_minimum_saturation:
            raise ValueError(
                "Candidate discovery saturation cannot exceed measured extent saturation"
            )
        if self.candidate_minimum_excess_green > self.extent_minimum_excess_green:
            raise ValueError(
                "Candidate discovery excess green cannot exceed measured extent excess green"
            )
        return self

    @property
    def visual_verifier_minimum_confidence(self) -> float:
        """Compatibility name used before verifier triage gained two thresholds."""

        return self.visual_verifier_acceptance_confidence


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
