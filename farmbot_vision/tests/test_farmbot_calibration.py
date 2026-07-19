from __future__ import annotations

import pytest

from farmbot_vision.calibration import from_farmbot_calibration
from farmbot_vision.database import Database
from farmbot_vision.models import Calibration, OriginLocation
from farmbot_vision.vision import garden_to_pixel


def _cal(origin: OriginLocation, ppm: float = 2.0, rotation: float = 0.0) -> Calibration:
    return Calibration(
        source="manual",
        pixels_per_mm_x=ppm,
        pixels_per_mm_y=ppm,
        rotation_degrees=rotation,
        origin_location=origin,
        processed_width=960,
        processed_height=720,
    )


def test_farmbot_scale_is_rescaled_to_processed_resolution():
    # FarmBot's 0.242 mm/px at native 2592x1944 must not be applied directly
    # to the resized 960x720 frame; it is inverted and rescaled by the ratio.
    cal = from_farmbot_calibration(
        coordinate_scale_mm_per_px=0.242,
        reference_width=2592,
        reference_height=1944,
        processed_width=960,
        processed_height=720,
        rotation_degrees=-31.9,
        offset_x_mm=0,
        offset_y_mm=0,
        origin_location=OriginLocation.TOP_LEFT,
        uncertainty_mm=10,
        analysis_resolution="960x720",
    )
    reference_ppm = 1 / 0.242
    assert cal.pixels_per_mm_x == pytest.approx(reference_ppm * 960 / 2592)
    assert cal.pixels_per_mm_y == pytest.approx(reference_ppm * 720 / 1944)
    # 4:3 preserving resize -> isotropic scale.
    assert cal.pixels_per_mm_x == pytest.approx(cal.pixels_per_mm_y)
    assert cal.rotation_degrees == -31.9
    assert cal.processed_width == 960
    assert cal.calibration_version == "farmbot@2592x1944"


def test_farmbot_scale_rejects_nonpositive_inputs():
    with pytest.raises(ValueError, match="positive"):
        from_farmbot_calibration(
            coordinate_scale_mm_per_px=0,
            reference_width=2592,
            reference_height=1944,
            processed_width=960,
            processed_height=720,
            rotation_degrees=0,
            offset_x_mm=0,
            offset_y_mm=0,
            origin_location=OriginLocation.TOP_LEFT,
            uncertainty_mm=10,
            analysis_resolution="960x720",
        )


def test_top_left_origin_matches_legacy_behaviour():
    # A plant 100 mm east of the image centre lands right of centre.
    cal = _cal(OriginLocation.TOP_LEFT)
    px, py = garden_to_pixel(1100, 1000, 1000, 1000, 960, 720, cal)
    assert px == pytest.approx(960 / 2 + 100 * 2)
    assert py == pytest.approx(720 / 2)


def test_top_right_origin_flips_x():
    cal = _cal(OriginLocation.TOP_RIGHT)
    px, _ = garden_to_pixel(1100, 1000, 1000, 1000, 960, 720, cal)
    assert px == pytest.approx(960 / 2 - 100 * 2)


def test_bottom_left_origin_flips_y():
    cal = _cal(OriginLocation.BOTTOM_LEFT)
    _, py = garden_to_pixel(1000, 1050, 1000, 1000, 960, 720, cal)
    assert py == pytest.approx(720 / 2 - 50 * 2)


def test_bottom_right_origin_flips_both():
    cal = _cal(OriginLocation.BOTTOM_RIGHT)
    px, py = garden_to_pixel(1100, 1050, 1000, 1000, 960, 720, cal)
    assert px == pytest.approx(960 / 2 - 100 * 2)
    assert py == pytest.approx(720 / 2 - 50 * 2)


def test_default_origin_is_top_left():
    cal = Calibration(source="manual", pixels_per_mm_x=1.0, pixels_per_mm_y=1.0)
    assert cal.origin_location == OriginLocation.TOP_LEFT


def test_origin_round_trips_through_database(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    database.save_calibration(
        "bot",
        _cal(OriginLocation.BOTTOM_RIGHT),
    )
    active = database.active_calibration("bot")
    assert active is not None
    assert active.origin_location == OriginLocation.BOTTOM_RIGHT
