from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field


class SoilSettings(BaseModel):
    """User-managed soil-height planning and automation controls."""

    clear_soil_margin_mm: float = Field(default=75, ge=0, le=250)
    grid_spacing_mm: float = Field(default=500, ge=50, le=5000)
    grid_maximum_deviation_mm: float = Field(default=100, ge=0, lt=200)
    pair_disagreement_limit_mm: float = Field(default=8, ge=1, le=50)
    scheduled_run_enabled: bool = False
    scheduled_run_time: str = Field(default="03:00", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    automatic_acceptance_enabled: bool = False
    automatic_acceptance_confidence_percent: float = Field(default=90, ge=0, le=100)
    automatic_acceptance_margin_mm: float = Field(default=20, ge=0, le=500)
    automatic_retry_enabled: bool = False
    automatic_retry_delay: float = Field(default=15, gt=0, le=168)
    automatic_retry_unit: str = Field(default="minutes", pattern=r"^(minutes|hours)$")

    @property
    def automatic_retry_delay_seconds(self) -> float:
        multiplier = 3600 if self.automatic_retry_unit == "hours" else 60
        return self.automatic_retry_delay * multiplier


class SoilSettingsStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> SoilSettings:
        if not self.path.exists():
            return SoilSettings()
        try:
            return SoilSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return SoilSettings()

    def save(self, values: SoilSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            handle.write(values.model_dump_json(indent=2))
            temp = Path(handle.name)
        os.replace(temp, self.path)
