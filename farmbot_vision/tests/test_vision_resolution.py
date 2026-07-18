from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
from conftest import encode_jpeg, synthetic_plant
from farmbot_vision.models import Calibration, PlantSeed
from farmbot_vision.vision import ClassicalVisionEngine, resize_prior_mask

NOW = datetime(2026, 2, 1, tzinfo=UTC)

# Physical plant: 40 mm disc with a genuine leaf extending to 90 mm.
RADIUS_MM = 40.0
LEAF_MM = 90.0

RESOLUTIONS = [
    (640, 480, 1.0),
    (960, 720, 1.5),
    (1280, 960, 2.0),
]


def _calibration(ppm: float) -> Calibration:
    return Calibration(
        source="processed_image",
        pixels_per_mm_x=ppm,
        pixels_per_mm_y=ppm,
        uncertainty_mm=10,
        analysis_resolution=f"{int(640 * ppm)}x{int(480 * ppm)}",
        processed_width=round(640 * ppm),
        processed_height=round(480 * ppm),
    )


def _analyse_plant(width: int, height: int, ppm: float):
    image = synthetic_plant(width, height, ppm, RADIUS_MM, LEAF_MM)
    seed = PlantSeed(
        plant_id=1, crop_slug="lettuce", center_px=(width / 2, height / 2), current_radius_mm=20
    )
    return ClassicalVisionEngine().analyse(
        encode_jpeg(image), 1, NOW, [seed], _calibration(ppm), {}
    )


@pytest.mark.parametrize(("width", "height", "ppm"), RESOLUTIONS)
def test_long_genuine_leaf_included_at_all_resolutions(width, height, ppm):
    result = _analyse_plant(width, height, ppm)
    assert result.measurements, f"no measurement at {width}x{height}"
    maximum = result.measurements[0].maximum_accepted_canopy_radius_mm
    # The 90 mm leaf must be retained (not clipped to the ~40 mm disc).
    assert maximum > 80, f"long leaf excluded at {width}x{height}: {maximum}"


def test_physical_radius_stable_across_resolutions():
    # Regression: the same physical plant measured at all three resolutions
    # yields a millimetre radius within a documented tolerance.
    maxima = []
    for width, height, ppm in RESOLUTIONS:
        result = _analyse_plant(width, height, ppm)
        maxima.append(result.measurements[0].maximum_accepted_canopy_radius_mm)
    spread = max(maxima) - min(maxima)
    assert spread < 12.0, f"radius varied too much across resolutions: {maxima}"
    for value in maxima:
        assert abs(value - LEAF_MM) < 15.0, f"radius {value} far from {LEAF_MM}"


@pytest.mark.parametrize(("width", "height", "ppm"), RESOLUTIONS)
def test_overlay_and_mask_match_processed_dimensions(width, height, ppm):
    result = _analyse_plant(width, height, ppm)
    overlay = cv2.imdecode(np.frombuffer(result.overlay_jpeg, np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(result.mask, np.uint8), cv2.IMREAD_UNCHANGED)
    assert overlay.shape[:2] == (height, width)
    assert mask.shape[:2] == (height, width)


def test_isolated_weed_stays_separate_at_high_resolution():
    ppm = 2.0
    width, height = 1280, 960
    image = synthetic_plant(width, height, ppm, RADIUS_MM, 0)
    # A small isolated weed far from the plant centre.
    cv2.circle(image, (width - 120, 120), round(8 * ppm), (20, 210, 30), -1)
    seed = PlantSeed(
        plant_id=1, crop_slug="lettuce", center_px=(width / 2, height / 2), current_radius_mm=20
    )
    result = ClassicalVisionEngine().analyse(
        encode_jpeg(image), 1, NOW, [seed], _calibration(ppm), {}
    )
    # The weed must not inflate the plant radius near the ~40 mm disc.
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 60


def test_overlapping_plants_remain_uncertain():
    ppm = 1.5
    width, height = 960, 720
    image = np.zeros((height, width, 3), np.uint8)
    cv2.circle(image, (width // 2 - 30, height // 2), round(40 * ppm), (20, 210, 30), -1)
    cv2.circle(image, (width // 2 + 30, height // 2), round(40 * ppm), (20, 210, 30), -1)
    seeds = [
        PlantSeed(
            plant_id=1,
            crop_slug="lettuce",
            center_px=(width / 2 - 30, height / 2),
            current_radius_mm=40,
        ),
        PlantSeed(
            plant_id=2,
            crop_slug="lettuce",
            center_px=(width / 2 + 30, height / 2),
            current_radius_mm=40,
        ),
    ]
    result = ClassicalVisionEngine().analyse(
        encode_jpeg(image), 1, NOW, seeds, _calibration(ppm), {}
    )
    assert any(m.ambiguous for m in result.measurements)


def test_temporal_mask_from_another_resolution_is_resized_safely():
    # A 640x480 prior reused for a 960x720 frame is rescaled, not stretched.
    prior = np.zeros((480, 640), np.uint8)
    cv2.circle(prior, (500, 240), 20, 255, -1)
    fitted = resize_prior_mask(prior, (720, 960))
    assert fitted.shape == (720, 960)
    assert fitted.max() == 255


def test_temporal_mask_with_wrong_aspect_is_rejected():
    prior = np.zeros((480, 640), np.uint8)
    assert resize_prior_mask(prior, (960, 960)) is None


def test_uncalibrated_diagnostic_only_produces_no_measurements():
    image = synthetic_plant(960, 720, 1.5, RADIUS_MM, LEAF_MM)
    result = ClassicalVisionEngine().diagnostic_only(encode_jpeg(image))
    assert result.measurements == []
    assert result.overlay_jpeg is not None
    overlay = cv2.imdecode(np.frombuffer(result.overlay_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert overlay.shape[:2] == (720, 960)
