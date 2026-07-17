from datetime import UTC, datetime
from time import perf_counter

from conftest import jpeg
from farmbot_vision.vision import ClassicalVisionEngine


def test_640x480_analysis_benchmark(seed, calibration):
    data = jpeg([("circle", ((320, 240), 100))], size=(480, 640))
    scaled_seed = seed.model_copy(update={"center_px": (320, 240), "current_radius_mm": 120})
    started = perf_counter()
    result = ClassicalVisionEngine().analyse(data, 1, datetime.now(UTC), [scaled_seed], calibration)
    assert result.measurements
    assert perf_counter() - started < 3.0
