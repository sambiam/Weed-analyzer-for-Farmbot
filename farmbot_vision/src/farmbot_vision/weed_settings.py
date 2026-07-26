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
    minimum_area_mm2: float = Field(default=25, ge=5, le=10_000)
    maximum_area_mm2: float = Field(default=2_500, ge=10, le=100_000)
    plant_exclusion_margin_mm: float = Field(default=35, ge=0, le=500)
    minimum_confidence: float = Field(default=0.75, ge=0, le=1)
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
