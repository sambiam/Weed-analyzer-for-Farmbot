"""Persistent thresholds for plant-radius recommendations."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class RadiusChangeSettings(BaseModel):
    """Minimum meaningful plant-radius changes, in millimetres."""

    minimum_radius_increase_mm: float = Field(default=3, ge=0, le=500)
    minimum_radius_reduction_mm: float = Field(default=30, ge=0, le=500)


class RadiusChangeSettingsStore:
    """Small atomic JSON store under /data for settings edited in the app."""

    def __init__(self, path: Path):
        self.path = path

    def load(self, defaults: RadiusChangeSettings | None = None) -> RadiusChangeSettings:
        fallback = defaults or RadiusChangeSettings()
        if not self.path.exists():
            return fallback
        try:
            return RadiusChangeSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return fallback

    def save(self, values: RadiusChangeSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(values.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(self.path)
