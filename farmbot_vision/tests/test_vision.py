from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
from conftest import jpeg
from farmbot_vision.models import Decision, PlantSeed
from farmbot_vision.vision import ClassicalVisionEngine, decode_jpeg, register_translation

NOW = datetime(2026, 2, 1, tzinfo=UTC)


def analyse(shapes, seed, calibration, previous=None, seeds=None):
    return ClassicalVisionEngine().analyse(
        jpeg(shapes), 9, NOW, seeds or [seed], calibration, previous or {}
    )


def test_circular_plant_without_weeds(seed, calibration):
    result = analyse([("circle", ((160, 120), 35))], seed, calibration)
    measurement = result.measurements[0]
    assert 33 <= measurement.maximum_accepted_canopy_radius_mm <= 37
    assert measurement.recommended_protection_radius_mm >= 63
    assert measurement.confidence > 0.7


def test_largest_genuine_leaf_is_not_excluded(seed, calibration):
    result = analyse(
        [("circle", ((160, 120), 30)), ("line", ((160, 120), (250, 120), 10))], seed, calibration
    )
    measurement = result.measurements[0]
    assert measurement.maximum_accepted_canopy_radius_mm > 85
    assert measurement.maximum_accepted_canopy_radius_mm > measurement.typical_canopy_radius_mm


def test_isolated_weed_does_not_inflate_radius(seed, calibration):
    result = analyse([("circle", ((160, 120), 30)), ("circle", ((285, 40), 10))], seed, calibration)
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35


def test_weed_close_to_crop_is_conservative(seed, calibration):
    result = analyse([("circle", ((160, 120), 25)), ("circle", ((235, 120), 9))], seed, calibration)
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35
    assert result.measurements[0].decision == Decision.UNCERTAIN


def test_overlapping_crops_are_uncertain(calibration):
    seeds = [
        PlantSeed(plant_id=1, crop_slug="lettuce", center_px=(135, 120), current_radius_mm=50),
        PlantSeed(plant_id=2, crop_slug="lettuce", center_px=(185, 120), current_radius_mm=50),
    ]
    result = analyse(
        [("circle", ((135, 120), 40)), ("circle", ((185, 120), 40))],
        seeds[0],
        calibration,
        seeds=seeds,
    )
    assert len(result.measurements) == 2
    assert all(item.ambiguous for item in result.measurements)


def test_disconnected_leaf_accepted_from_previous_mask(seed, calibration):
    previous = np.zeros((240, 320), np.uint8)
    cv2.circle(previous, (235, 120), 12, 255, -1)
    result = analyse(
        [("circle", ((160, 120), 25)), ("circle", ((235, 120), 12))],
        seed,
        calibration,
        {1: previous},
    )
    assert result.measurements[0].maximum_accepted_canopy_radius_mm > 80
    assert not result.measurements[0].ambiguous


def test_sudden_disconnected_region_is_uncertain(seed, calibration):
    result = analyse(
        [("circle", ((160, 120), 25)), ("circle", ((235, 120), 12))], seed, calibration
    )
    assert result.measurements[0].ambiguous
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35


def test_green_irrigation_line_and_noise_are_rejected(seed, calibration):
    shapes = [("circle", ((160, 120), 25)), ("rect", ((5, 20), (310, 24)))]
    result = analyse(shapes, seed, calibration)
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35


def test_camera_translation_registration():
    previous = np.zeros((120, 160), np.uint8)
    cv2.circle(previous, (70, 60), 15, 255, -1)
    transform = np.float32([[1, 0, 6], [0, 1, -4]])
    current = cv2.warpAffine(previous, transform, (160, 120))
    dx, dy, response = register_translation(previous, current)
    assert abs(dx - 6) < 0.5
    assert abs(dy + 4) < 0.5
    assert response > 0.5


def test_decode_keeps_processed_resolution_and_caps_at_ceiling():
    # The integration already resized to the processed size; the app keeps it.
    processed = np.zeros((960, 1280, 3), np.uint8)
    cv2.circle(processed, (640, 480), 200, (0, 255, 0), -1)
    ok, encoded = cv2.imencode(".jpg", processed)
    assert ok
    assert decode_jpeg(encoded.tobytes()).shape[:2] == (960, 1280)

    # Anything above the 1280x960 ceiling is defensively downscaled.
    oversized = np.zeros((1200, 1600, 3), np.uint8)
    ok, encoded = cv2.imencode(".jpg", oversized)
    assert ok
    decoded = decode_jpeg(encoded.tobytes())
    assert decoded.shape[1] <= 1280 and decoded.shape[0] <= 960
