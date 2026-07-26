from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field


class CanopyFusionSettings(BaseModel):
    """User-managed controls for calibrated multi-image canopy fusion."""

    enabled: bool = True
    always_fuse_when_available: bool = False
    activation_visible_fraction: float = Field(default=0.92, ge=0, le=1)
    minimum_views: int = Field(default=2, ge=2, le=20)
    maximum_time_gap_hours: float = Field(default=6, gt=0, le=720)
    minimum_view_confidence: float = Field(default=0.35, ge=0, le=1)
    minimum_supporting_views: int = Field(default=2, ge=1, le=10)
    single_view_acceptance_confidence: float = Field(default=0.82, ge=0, le=1)
    source_edge_margin_mm: float = Field(default=20, ge=0, le=250)
    radial_percentile: float = Field(default=97, ge=80, le=100)
    angular_sectors: int = Field(default=72, ge=12, le=360)
    minimum_angular_coverage: float = Field(default=0.70, ge=0, le=1)
    minimum_corroborated_fraction: float = Field(default=0.05, ge=0, le=1)
    maximum_automatic_disagreement_mm: float = Field(default=35, ge=0, le=500)
    automatic_requires_reliable_fusion: bool = True
    maximum_canvas_pixels: int = Field(default=2400, ge=480, le=6000)
    save_diagnostics: bool = True


class CanopyFusionSettingsStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> CanopyFusionSettings:
        if not self.path.exists():
            return CanopyFusionSettings()
        try:
            return CanopyFusionSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return CanopyFusionSettings()

    def save(self, values: CanopyFusionSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            handle.write(values.model_dump_json(indent=2))
            temp = Path(handle.name)
        os.replace(temp, self.path)
