from __future__ import annotations

import json

import pytest
from farmbot_vision.resolution import (
    DEFAULT_RESOLUTION,
    AnalysisResolution,
    Resolution,
    native_relative_workload,
)
from farmbot_vision.settings import Settings


def test_default_resolution_is_960x720():
    assert Settings().analysis_resolution == AnalysisResolution.R960X720
    assert DEFAULT_RESOLUTION == AnalysisResolution.R960X720
    assert (Settings().analysis_width, Settings().analysis_height) == (960, 720)


@pytest.mark.parametrize(
    ("value", "dims"),
    [("640x480", (640, 480)), ("960x720", (960, 720)), ("1280x960", (1280, 960))],
)
def test_supported_presets_are_accepted(value, dims):
    settings = Settings(analysis_resolution=value)
    assert (settings.analysis_width, settings.analysis_height) == dims
    assert settings.resolution.pixel_count == dims[0] * dims[1]


def test_unsupported_presets_are_rejected():
    with pytest.raises(ValueError):
        Settings(analysis_resolution="2592x1944")
    with pytest.raises(ValueError):
        Settings(analysis_resolution="800x600")
    with pytest.raises(ValueError):
        Resolution.from_value("1024x768")


def test_missing_option_migrates_to_default(tmp_path, monkeypatch):
    # An older installation whose options.json predates analysis_resolution.
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"mode": "observe", "safety_margin_mm": 15}))
    monkeypatch.setenv("FARMV_DATA_DIR", str(tmp_path))
    settings = Settings.load()
    assert settings.analysis_resolution == AnalysisResolution.R960X720


def test_blank_option_migrates_to_default():
    assert Settings(analysis_resolution="").analysis_resolution == AnalysisResolution.R960X720


def test_relative_workloads_are_documented():
    assert Resolution(AnalysisResolution.R640X480).relative_workload == 1.0
    assert Resolution(AnalysisResolution.R960X720).relative_workload == 2.25
    assert Resolution(AnalysisResolution.R1280X960).relative_workload == 4.0
    assert native_relative_workload() == pytest.approx(16.4, abs=0.1)


def test_resolution_label_and_dict():
    resolution = Resolution(AnalysisResolution.R960X720)
    assert resolution.label == "960 x 720 (2.25x)"
    payload = resolution.as_dict()
    assert payload["width"] == 960 and payload["pixel_count"] == 691200
