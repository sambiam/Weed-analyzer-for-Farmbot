from __future__ import annotations

import numpy as np
import pytest
from conftest import vision_image_dict
from farmbot_vision.calibration import (
    resolve_calibration,
    scale_reference_to_processed,
    transform_manual_calibration,
)
from farmbot_vision.models import Calibration, CameraCalibration, VisionImage
from farmbot_vision.resolution import AnalysisResolution, Resolution


def _image(width: int, height: int, **kwargs) -> VisionImage:
    frame = np.zeros((height, width, 3), np.uint8)
    return VisionImage.model_validate(
        vision_image_dict(frame, source_wh=(2592, 1944), oriented_wh=(2592, 1944), **kwargs)
    )


RES_960 = Resolution(AnalysisResolution.R960X720)


def test_processed_calibration_is_preferred():
    processed = {
        "available": True,
        "pixels_per_mm_x": 0.455,
        "pixels_per_mm_y": 0.455,
        "rotation_degrees": 0.0,
        "offset_x_mm": 0.0,
        "offset_y_mm": 0.0,
        "basis": "processed_image",
        "width": 960,
        "height": 720,
    }
    image = _image(960, 720, processed_calibration=processed)
    reference = CameraCalibration(
        available=True,
        pixels_per_mm_x=1.2,
        pixels_per_mm_y=1.2,
        reference_width=2592,
        reference_height=1944,
        basis="native_frame",
    )
    outcome = resolve_calibration(image, reference, None, RES_960, 10)
    assert outcome.source == "processed_image"
    assert outcome.calibration.pixels_per_mm_x == pytest.approx(0.455)


def test_reference_calibration_is_rescaled():
    image = _image(960, 720)
    reference = CameraCalibration(
        available=True,
        pixels_per_mm_x=1.2,
        pixels_per_mm_y=1.2,
        reference_width=2592,
        reference_height=1944,
        basis="native_frame",
    )
    scaled = scale_reference_to_processed(reference, image, RES_960, 10)
    assert scaled is not None
    # 1.2 * 960 / 2592
    assert scaled.pixels_per_mm_x == pytest.approx(1.2 * 960 / 2592)
    assert scaled.source == "reference_scaled"


def test_reference_without_dimensions_is_refused():
    image = _image(960, 720)
    reference = CameraCalibration(available=True, pixels_per_mm_x=1.2, pixels_per_mm_y=1.2)
    assert scale_reference_to_processed(reference, image, RES_960, 10) is None
    outcome = resolve_calibration(image, reference, None, RES_960, 10)
    assert outcome.calibration is None
    assert any("could not be scaled" in w for w in outcome.warnings)


def test_native_scale_never_applied_directly():
    # A 2592-wide reference scale must not survive unchanged onto 960 pixels.
    image = _image(960, 720)
    reference = CameraCalibration(
        available=True,
        pixels_per_mm_x=2.0,
        pixels_per_mm_y=2.0,
        reference_width=2592,
        reference_height=1944,
        basis="native_frame",
    )
    scaled = scale_reference_to_processed(reference, image, RES_960, 10)
    assert scaled.pixels_per_mm_x < 2.0


def test_invalid_processed_calibration_is_rejected():
    # basis says processed but dimensions disagree with the returned image.
    processed = {
        "available": True,
        "pixels_per_mm_x": 0.455,
        "pixels_per_mm_y": 0.455,
        "basis": "processed_image",
        "width": 640,
        "height": 480,
    }
    image = _image(960, 720, processed_calibration=processed)
    with pytest.raises(ValueError, match="dimensions do not match"):
        resolve_calibration(image, CameraCalibration(available=False), None, RES_960, 10)


def test_processed_calibration_requires_positive_scales():
    with pytest.raises(ValueError):
        # available with zero scale is rejected at model level (gt=0)
        from farmbot_vision.models import ProcessedCalibration

        ProcessedCalibration(
            available=True,
            pixels_per_mm_x=0,
            pixels_per_mm_y=0.4,
            basis="processed_image",
            width=960,
            height=720,
        )


def test_manual_calibration_resolution_mismatch_is_detected():
    manual = Calibration(
        source="manual",
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        processed_width=640,
        processed_height=480,
    )
    image = _image(1280, 960)
    outcome = resolve_calibration(image, CameraCalibration(available=False), manual, RES_960, 10)
    # 640x480 manual -> 1280x960 image: scale is fully known, so it transforms.
    assert outcome.calibration is not None
    assert outcome.source == "manual_transformed"
    assert any("transformed" in w for w in outcome.warnings)


def test_manual_calibration_transforms_when_scales_known():
    manual = Calibration(
        source="manual",
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        processed_width=640,
        processed_height=480,
        point_a_x=100,
        point_a_y=100,
        point_b_x=200,
        point_b_y=100,
    )
    transformed = transform_manual_calibration(manual, 1280, 960)
    assert transformed.pixels_per_mm_x == pytest.approx(2.0)
    assert transformed.point_b_x == pytest.approx(400)
    assert transformed.transformed_from_id == manual.version_id


def test_manual_without_recorded_resolution_is_not_reused():
    manual = Calibration(source="manual", pixels_per_mm_x=1.0, pixels_per_mm_y=1.0)
    image = _image(960, 720)
    outcome = resolve_calibration(image, CameraCalibration(available=False), manual, RES_960, 10)
    assert outcome.calibration is None
    assert any("recalibration" in w for w in outcome.warnings)


def test_exif_oriented_dimensions_are_handled():
    # Source is portrait 1944x2592; EXIF orients to landscape 2592x1944.
    frame = np.zeros((720, 960, 3), np.uint8)
    image = VisionImage.model_validate(
        vision_image_dict(frame, source_wh=(1944, 2592), oriented_wh=(2592, 1944))
    )
    reference = CameraCalibration(
        available=True,
        pixels_per_mm_x=1.2,
        pixels_per_mm_y=1.2,
        reference_width=2592,
        reference_height=1944,
        basis="native_frame",
    )
    scaled = scale_reference_to_processed(reference, image, RES_960, 10)
    assert scaled is not None
    assert scaled.pixels_per_mm_x == pytest.approx(1.2 * 960 / 2592)
