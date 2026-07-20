from __future__ import annotations

import pytest

from farmbot_vision.calibration_store import CalibrationStore, FarmbotCalibrationInput
from farmbot_vision.models import OriginLocation


def _input() -> FarmbotCalibrationInput:
    return FarmbotCalibrationInput(
        coordinate_scale=0.242,
        reference_width=2592,
        reference_height=1944,
        rotation_degrees=-31.9,
        origin_location=OriginLocation.TOP_RIGHT,
        offset_x_mm=1.5,
        offset_y_mm=-2.0,
    )


def test_round_trips_per_entry(tmp_path):
    store = CalibrationStore(tmp_path / "cal.json")
    store.save("botA", _input())
    got = store.get("botA")
    assert got is not None
    assert got.coordinate_scale == 0.242
    assert got.origin_location == OriginLocation.TOP_RIGHT
    assert got.offset_x_mm == 1.5
    assert store.get("botB") is None


def test_survives_a_new_store_instance(tmp_path):
    path = tmp_path / "cal.json"
    CalibrationStore(path).save("bot", _input())
    # A fresh instance (as after a restart) reads the same values from disk.
    reopened = CalibrationStore(path).get("bot")
    assert reopened is not None
    assert reopened.rotation_degrees == -31.9


def test_saving_a_second_bot_keeps_the_first(tmp_path):
    store = CalibrationStore(tmp_path / "cal.json")
    store.save("botA", _input())
    store.save("botB", _input().model_copy(update={"coordinate_scale": 0.3}))
    assert store.get("botA").coordinate_scale == 0.242
    assert store.get("botB").coordinate_scale == 0.3


def test_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        FarmbotCalibrationInput(coordinate_scale=0, reference_width=2592, reference_height=1944)


def test_missing_file_returns_none(tmp_path):
    assert CalibrationStore(tmp_path / "absent.json").get("bot") is None
