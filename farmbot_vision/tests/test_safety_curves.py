from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from farmbot_vision.curves import fit_monotonic_curve, pava
from farmbot_vision.models import Decision, Measurement, OperatingMode
from farmbot_vision.safety import decide
from farmbot_vision.settings import Settings


def measurement(current=100, recommendation=120, confidence=0.95, ambiguous=False):
    return Measurement(
        measurement_id=uuid4(),
        plant_id=1,
        crop_slug="lettuce",
        image_id=2,
        image_timestamp=datetime.now(UTC),
        current_radius_mm=current,
        typical_canopy_radius_mm=80,
        maximum_accepted_canopy_radius_mm=90,
        recommended_protection_radius_mm=recommendation,
        confidence=confidence,
        decision=Decision.OBSERVED,
        reason="test",
        ambiguous=ambiguous,
        algorithm_version="test",
    )


def test_missing_calibration_prevents_job(tmp_path):
    from farmbot_vision.database import Database

    assert Database(tmp_path / "db.sqlite").active_calibration("bot") is None


def test_shrink_is_always_retained():
    result = decide(
        measurement(current=100, recommendation=80), OperatingMode.AUTO_RADIUS, Settings()
    )
    assert result.decision == Decision.RETAIN


def test_auto_radius_requires_confidence():
    result = decide(measurement(confidence=0.5), OperatingMode.AUTO_RADIUS, Settings())
    assert result.decision == Decision.UNCERTAIN


def test_monotonic_curve_fitting():
    curve = fit_monotonic_curve([(1, 20), (4, 18), (7, 30), (10, 28)])
    values = list(curve.values())
    assert values == sorted(values)


def test_radius_becomes_farmbot_diameter():
    curve = fit_monotonic_curve([(1, 25)], quantile=1)
    assert curve["0"] == 50


def test_curve_has_at_most_ten_control_points():
    curve = fit_monotonic_curve([(day, day + 10) for day in range(60)], bin_days=1)
    assert len(curve) <= 10


def test_pava_preserves_length_and_monotonicity():
    fitted = pava([1, 4, 3, 2, 8])
    assert len(fitted) == 5
    assert fitted == sorted(fitted)
