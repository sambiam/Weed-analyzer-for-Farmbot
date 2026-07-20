"""Durable, in-app-editable FarmBot calibration store.

The FarmBot camera-calibration inputs a user enters in the app are the master
record and must survive a restart. They are persisted as JSON in the add-on's
persistent ``/data`` volume (not in the add-on options, which the add-on cannot
write back at runtime), keyed by ``config_entry_id`` so a multi-bot install
keeps a calibration per bot.

On startup the app seeds the SQLite active calibration from this file (see
``web.seed_calibration_from_store``); the analysis pipeline continues to read
the active calibration from the database exactly as before, so measurement
provenance and version ids are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from .models import OriginLocation

LOGGER = logging.getLogger(__name__)


class FarmbotCalibrationInput(BaseModel):
    """The exact numbers copied from FarmBot's camera calibration.

    These are stored verbatim (the resize to the analysis resolution happens in
    ``calibration.from_farmbot_calibration`` at save time), so the user can
    always re-read and edit what they originally entered.
    """

    coordinate_scale: float = Field(gt=0)
    reference_width: int = Field(ge=1)
    reference_height: int = Field(ge=1)
    rotation_degrees: float = 0.0
    origin_location: OriginLocation = OriginLocation.TOP_LEFT
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0


class CalibrationStore:
    """Read/write FarmBot calibration inputs keyed by config entry id."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Could not read calibration store %s: %s", self.path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, entry_id: str) -> FarmbotCalibrationInput | None:
        raw = self._read_all().get(entry_id)
        if raw is None:
            return None
        try:
            return FarmbotCalibrationInput.model_validate(raw)
        except ValueError as exc:
            LOGGER.warning("Stored calibration for %s is invalid: %s", entry_id, exc)
            return None

    def save(self, entry_id: str, values: FarmbotCalibrationInput) -> None:
        store = self._read_all()
        store[entry_id] = values.model_dump(mode="json")
        # Atomic replace so a crash mid-write never truncates the store.
        directory = self.path.parent
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
        )
        try:
            json.dump(store, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, self.path)
