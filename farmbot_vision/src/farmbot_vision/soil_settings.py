from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field


class SoilSettings(BaseModel):
    """User-managed clear-soil site planning controls."""

    clear_soil_margin_mm: float = Field(default=75, ge=0, le=250)


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
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
            handle.write(values.model_dump_json(indent=2))
            temp = Path(handle.name)
        os.replace(temp, self.path)
