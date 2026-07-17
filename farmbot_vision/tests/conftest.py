from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from farmbot_vision.models import Calibration, PlantSeed


@pytest.fixture
def calibration() -> Calibration:
    return Calibration(source="manual", pixels_per_mm_x=1, pixels_per_mm_y=1, uncertainty_mm=10)


@pytest.fixture
def seed() -> PlantSeed:
    return PlantSeed(
        plant_id=1,
        crop_slug="lettuce",
        center_px=(160, 120),
        current_radius_mm=60,
        planted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def jpeg(shapes, size=(240, 320)) -> bytes:
    image = np.zeros((*size, 3), np.uint8)
    for shape in shapes:
        kind, values = shape
        if kind == "circle":
            cv2.circle(image, *values, (20, 210, 30), -1)
        elif kind == "line":
            cv2.line(image, values[0], values[1], (20, 210, 30), values[2])
        elif kind == "rect":
            cv2.rectangle(image, *values, (20, 210, 30), -1)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return data.tobytes()
