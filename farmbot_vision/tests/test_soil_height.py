from __future__ import annotations

import cv2
import numpy as np
import pytest

import farmbot_vision.soil_height as soil_height
from farmbot_vision.soil_height import (
    PairEstimate,
    SoilCalibrationQualityError,
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


def _minimal_frame(z_offset: float, image_id: int) -> SoilFrame:
    return SoilFrame(
        image_id=image_id,
        jpeg=b"",
        x=100,
        y=200,
        z=-z_offset,
        lateral_offset_mm=0,
        z_offset_mm=z_offset,
        processed_width=640,
        processed_height=480,
        source_width=640,
        source_height=480,
    )


def _fake_pair(normalized_disparity: float, coverage: float, support: float) -> PairEstimate:
    return PairEstimate(
        normalized_disparity=normalized_disparity,
        disparity_px=normalized_disparity * 15,
        baseline_mm=15,
        valid_coverage=coverage,
        plane_support=support,
        lr_error_px=0.5,
        plane_mad_px=0.5,
        disparity_map=np.zeros((4, 4)),
        valid_mask=np.zeros((4, 4), dtype=bool),
        plane_mask=np.zeros((4, 4), dtype=bool),
        rectification_overlay=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def _patch_marginal_z0(monkeypatch):
    """Only the Z=0 level has fewer than two pairs passing quality gates."""
    by_z = {
        0: [_fake_pair(2.0, 0.3, 0.6), _fake_pair(2.0, 0.05, 0.6), _fake_pair(2.0, 0.3, 0.2)],
        25: [_fake_pair(600 / 275, 0.3, 0.6)] * 3,
        50: [_fake_pair(600 / 250, 0.3, 0.6)] * 3,
    }
    monkeypatch.setattr(soil_height, "estimate_triplet", lambda group: by_z[group[0].z_offset_mm])
    return [_minimal_frame(z, i) for i, z in enumerate((0, 25, 50), start=1)]


def test_fit_calibration_reports_which_gates_failed_at_which_level(monkeypatch):
    frames = _patch_marginal_z0(monkeypatch)
    with pytest.raises(SoilCalibrationQualityError, match=r"at 0 mm \(1/3 pairs passed") as exc:
        fit_calibration(
            config_entry_id="bot",
            point_id=70,
            capture_z=0,
            baseline_mm=15,
            reference_distance_mm=300,
            z_direction=-1,
            frames=frames,
        )
    assert "coverage 0.05 < 0.15" in str(exc.value)
    assert "plane support 0.20 < 0.50" in str(exc.value)


def test_fit_calibration_force_overrides_the_gate_and_records_a_warning(monkeypatch):
    frames = _patch_marginal_z0(monkeypatch)
    calibration = fit_calibration(
        config_entry_id="bot",
        point_id=70,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=300,
        z_direction=-1,
        frames=frames,
        force=True,
    )
    assert calibration.quality_override is True
    assert len(calibration.quality_warnings) == 1
    assert "at 0 mm" in calibration.quality_warnings[0]
    assert "accepted by override" in calibration.quality_warnings[0]
    assert calibration.residual_mm <= 5
