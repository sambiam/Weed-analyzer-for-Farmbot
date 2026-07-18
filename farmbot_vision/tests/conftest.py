from __future__ import annotations

import base64
import hashlib
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


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return data.tobytes()


def synthetic_plant(
    width: int, height: int, ppm: float, radius_mm: float, leaf_mm: float = 0.0
) -> np.ndarray:
    """Render a synthetic plant centred in a frame at ``ppm`` px/mm.

    A filled green disc of ``radius_mm`` plus an optional straight leaf out to
    ``leaf_mm`` from the centre. Scaling ppm with the frame reproduces the same
    physical plant at any resolution.
    """
    image = np.zeros((height, width, 3), np.uint8)
    cx, cy = width // 2, height // 2
    cv2.circle(image, (cx, cy), max(1, round(radius_mm * ppm)), (20, 210, 30), -1)
    if leaf_mm > 0:
        thickness = max(2, round(6 * ppm))
        cv2.line(image, (cx, cy), (cx + round(leaf_mm * ppm), cy), (20, 210, 30), thickness)
    return image


def vision_image_dict(
    image: np.ndarray,
    *,
    image_id: int = 1,
    source_wh: tuple[int, int] | None = None,
    oriented_wh: tuple[int, int] | None = None,
    with_v2: bool = True,
    processed_calibration: dict | None = None,
    sha_override: str | None = None,
    resize_override: tuple[float, float] | None = None,
    base64_override: str | None = None,
) -> dict:
    """Build a ``VisionImage``-shaped response dict for a processed array."""
    data = encode_jpeg(image)
    height, width = image.shape[:2]
    oriented_w, oriented_h = oriented_wh or (width, height)
    source_w, source_h = source_wh or (oriented_w, oriented_h)
    payload: dict = {
        "image_id": image_id,
        "content_type": "image/jpeg",
        "sha256": sha_override or hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "image_base64": base64_override or base64.b64encode(data).decode("ascii"),
        "meta": {"x": 0.0, "y": 0.0, "z": 0.0, "created_at": "2026-02-01T00:00:00+00:00"},
    }
    if with_v2:
        scale_x, scale_y = resize_override or (width / oriented_w, height / oriented_h)
        payload.update(
            {
                "source_width": source_w,
                "source_height": source_h,
                "oriented_width": oriented_w,
                "oriented_height": oriented_h,
                "resize_scale_x": scale_x,
                "resize_scale_y": scale_y,
            }
        )
    if processed_calibration is not None:
        payload["processed_calibration"] = processed_calibration
    return payload
