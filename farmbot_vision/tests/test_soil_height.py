from __future__ import annotations

import cv2
import numpy as np
import pytest

from farmbot_vision.soil_height import (
    SoilFrame,
    SoilHeightError,
    analyse_soil_height,
    estimate_triplet,
    fit_calibration,
)


def _encode(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return data.tobytes()


def _triplet(
    normalized_disparity: float,
    *,
    z_offset: float = 0,
    camera_z: float = 0,
    green_cover: bool = False,
    vertical_shift_px: float = 0,
    brightness_step: float = 0,
) -> list[SoilFrame]:
    rng = np.random.default_rng(47)
    texture = rng.integers(25, 230, (480, 640), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    color = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    if green_cover:
        cv2.rectangle(color, (0, 0), (639, 479), (20, 210, 30), -1)
    frames = []
    for image_id, offset in enumerate((-15.0, 0.0, 15.0), start=1):
        shift = normalized_disparity * (offset + 15)
        view_index = (offset + 15) / 15
        image = cv2.warpAffine(
            color,
            np.float32([[1, 0, shift], [0, 1, vertical_shift_px * view_index]]),
            (640, 480),
            borderMode=cv2.BORDER_REFLECT,
        )
        if brightness_step:
            image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_step * view_index)
        frames.append(
            SoilFrame(
                image_id=image_id + round(z_offset) * 10,
                jpeg=_encode(image),
                x=100,
                y=200 + offset,
                z=camera_z,
                lateral_offset_mm=offset,
                z_offset_mm=z_offset,
                processed_width=640,
                processed_height=480,
                source_width=640,
                source_height=480,
            )
        )
    return frames


def test_three_pair_stereo_recovers_normalized_disparity():
    estimates = estimate_triplet(_triplet(1.2))
    assert len(estimates) == 3
    assert all(estimate.passes_geometry for estimate in estimates)
    assert np.median([item.normalized_disparity for item in estimates]) == pytest.approx(
        1.2, abs=0.03
    )


def test_guided_calibration_and_measurement_recover_soil_z():
    # f / distance gives disparity normalized by baseline.
    frames = []
    for offset, distance in ((0, 300), (25, 275), (50, 250)):
        frames.extend(_triplet(600 / distance, z_offset=offset, camera_z=-offset))
    calibration = fit_calibration(
        config_entry_id="bot",
        point_id=70,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=300,
        z_direction=-1,
        frames=frames,
    )
    assert calibration.residual_mm <= 5

    result = analyse_soil_height(_triplet(600 / 285), calibration)
    assert result.valid
    assert result.proposed_z_mm == pytest.approx(-285, abs=2)
    assert result.uncertainty_mm <= 10
    assert {"triptych.jpg", "disparity.jpg", "valid-mask.png", "soil-plane.jpg"} <= set(
        result.artifacts
    )
    assert "rectification-overlay.jpg" in result.artifacts


def test_vertical_misalignment_and_brightness_changes_are_rectified():
    estimates = estimate_triplet(_triplet(1.4, vertical_shift_px=2.5, brightness_step=-18))
    passing = [estimate for estimate in estimates if estimate.passes_geometry]
    assert len(passing) >= 2
    assert np.median([item.normalized_disparity for item in passing]) == pytest.approx(
        1.4, abs=0.08
    )


def test_low_texture_or_vegetation_fails_closed():
    result = analyse_soil_height(
        _triplet(1.8, green_cover=True),
        fit_calibration(
            config_entry_id="bot",
            point_id=70,
            capture_z=0,
            baseline_mm=15,
            reference_distance_mm=300,
            z_direction=-1,
            frames=sum(
                (
                    _triplet(600 / distance, z_offset=offset, camera_z=-offset)
                    for offset, distance in ((0, 300), (25, 275), (50, 250))
                ),
                [],
            ),
        ),
    )
    assert not result.valid
    assert result.proposed_z_mm is None
    assert "triptych.jpg" in result.artifacts


def test_calibration_rejects_non_monotonic_disparity():
    frames = (
        _triplet(2.0, z_offset=0)
        + _triplet(1.8, z_offset=25, camera_z=-25)
        + _triplet(2.4, z_offset=50, camera_z=-50)
    )
    with pytest.raises(SoilHeightError, match="not monotonic"):
        fit_calibration(
            config_entry_id="bot",
            point_id=70,
            capture_z=0,
            baseline_mm=15,
            reference_distance_mm=300,
            z_direction=-1,
            frames=frames,
        )


def test_measurement_rejects_changed_source_geometry():
    calibration_frames = sum(
        (
            _triplet(600 / distance, z_offset=offset, camera_z=-offset)
            for offset, distance in ((0, 300), (25, 275), (50, 250))
        ),
        [],
    )
    calibration = fit_calibration(
        config_entry_id="bot",
        point_id=70,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=300,
        z_direction=-1,
        frames=calibration_frames,
    )
    changed = _triplet(2.0)
    for frame in changed:
        frame.source_width = 1280
        frame.source_height = 960
    result = analyse_soil_height(changed, calibration)
    assert not result.valid
    assert "recalibration" in result.reason
