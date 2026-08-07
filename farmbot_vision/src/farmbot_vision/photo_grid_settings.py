from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field, field_validator

WEEKDAY_LABELS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class PhotoGridScheduleSettings(BaseModel):
    """User-managed automation for the calibrated whole-bed photo grid."""

    enabled: bool = False
    time: str = Field(default="03:00", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    # Python weekday numbering: 0 = Monday … 6 = Sunday.
    days: list[int] = Field(default_factory=list)
    quality_repair_blurry_enabled: bool = True
    quality_repair_washed_out_enabled: bool = True
    quality_repair_close_leaf_enabled: bool = True

    @field_validator("days")
    @classmethod
    def _clean_days(cls, value: list[int]) -> list[int]:
        return sorted({int(day) for day in value if 0 <= int(day) <= 6})

    @property
    def runnable(self) -> bool:
        """A schedule with no selected day would silently never fire."""
        return self.enabled and bool(self.days)

    def due(self, weekday: int, clock: str) -> bool:
        return self.runnable and weekday in self.days and clock == self.time

    def summary(self) -> str:
        if not self.enabled:
            return "Scheduled photo grid is off."
        if not self.days:
            return "Select at least one day for the scheduled photo grid to run."
        if len(self.days) == 7:
            days = "every day"
        else:
            days = "every " + ", ".join(WEEKDAY_LABELS[day] for day in self.days)
        return f"Scheduled photo grid runs {days} at {self.time}."


class PhotoGridScheduleStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> PhotoGridScheduleSettings:
        if not self.path.exists():
            return PhotoGridScheduleSettings()
        try:
            return PhotoGridScheduleSettings.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return PhotoGridScheduleSettings()

    def save(self, values: PhotoGridScheduleSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            handle.write(values.model_dump_json(indent=2))
            temp = Path(handle.name)
        os.replace(temp, self.path)
