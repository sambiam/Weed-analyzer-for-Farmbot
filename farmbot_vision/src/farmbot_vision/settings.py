from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .models import OperatingMode
from .resolution import DEFAULT_RESOLUTION, AnalysisResolution, Resolution


class Settings(BaseModel):
    selected_config_entry_id: str = ""
    mode: OperatingMode = OperatingMode.OBSERVE
    # Existing installations without this option validate to the new default.
    # Changing it takes effect only after the app is restarted; settings are
    # loaded once at startup (see Settings.load and web.settings).
    analysis_resolution: AnalysisResolution = DEFAULT_RESOLUTION
    schedule_enabled: bool = False
    schedule_time: str = "03:00"
    safety_margin_mm: float = Field(default=20, ge=0)
    calibration_uncertainty_mm: float = Field(default=10, ge=0)
    minimum_auto_confidence: float = Field(default=0.90, ge=0, le=1)
    maximum_daily_radius_growth_mm: float = Field(default=50, gt=0)
    maximum_single_update_percent: float = Field(default=40, gt=0)
    minimum_observations_for_curve: int = Field(default=5, ge=2)
    maximum_system_load_percent: int = Field(default=80, ge=20, le=100)
    minimum_free_memory_mb: int = Field(default=512, ge=128)
    diagnostic_retention_days: int = Field(default=14, ge=0)
    failed_analysis_retention_days: int = Field(default=60, ge=1)
    successful_mask_retention_days: int = Field(default=7, ge=0)
    heartbeat_minutes: int = Field(default=15, ge=5)
    image_lookback_hours: int = Field(default=72, ge=1, le=720)
    max_image_payload_bytes: int = 5 * 1024 * 1024
    data_dir: Path = Path("/data")

    @field_validator("analysis_resolution", mode="before")
    @classmethod
    def _coerce_resolution(cls, value: object) -> object:
        # A blank or missing option (older installs) falls back to the default
        # rather than raising, so upgrades never break on first boot.
        if value in (None, ""):
            return DEFAULT_RESOLUTION
        return value

    @property
    def resolution(self) -> Resolution:
        return Resolution(self.analysis_resolution)

    @property
    def analysis_width(self) -> int:
        return self.resolution.width

    @property
    def analysis_height(self) -> int:
        return self.resolution.height

    @classmethod
    def load(cls) -> Settings:
        data_dir = Path(os.getenv("FARMV_DATA_DIR", "/data"))
        options_file = data_dir / "options.json"
        raw = json.loads(options_file.read_text(encoding="utf-8")) if options_file.exists() else {}
        raw["data_dir"] = data_dir
        return cls.model_validate(raw)
