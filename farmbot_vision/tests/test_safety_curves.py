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
        plant_age_days=31,
    )


def test_missing_calibration_prevents_job(tmp_path):
    from farmbot_vision.database import Database

    assert Database(tmp_path / "db.sqlite").active_calibration("bot") is None


def test_small_shrink_is_held_below_the_new_default_threshold():
    result = decide(
        measurement(current=100, recommendation=95), OperatingMode.AUTO_RADIUS, Settings()
    )
    assert result.decision == Decision.UNCERTAIN
    assert result.confidence == 0.05


def test_small_radius_changes_are_retained_at_low_confidence():
    increase = decide(
        measurement(current=100, recommendation=102), OperatingMode.RECOMMEND, Settings()
    )
    reduction = decide(
        measurement(current=100, recommendation=75), OperatingMode.RECOMMEND, Settings()
    )

    assert increase.confidence == 0.05
    assert reduction.confidence == 0.05


def test_large_shrink_is_reviewable_but_not_automatically_applied():
    result = decide(
        measurement(current=100, recommendation=50), OperatingMode.RECOMMEND, Settings()
    )
    assert result.decision == Decision.RECOMMENDED
    assert result.confidence < Settings().minimum_auto_confidence
    automatic = decide(
        measurement(current=100, recommendation=50), OperatingMode.AUTO_RADIUS, Settings()
    )
    assert automatic.decision == Decision.UNCERTAIN
    assert automatic.confidence < Settings().minimum_auto_confidence


def test_shrink_limit_is_configurable():
    result = decide(
        measurement(current=200, recommendation=165),
        OperatingMode.AUTO_RADIUS,
        Settings(maximum_automatic_radius_reduction_percent=20),
    )
    assert result.decision == Decision.APPLIED


def test_auto_radius_requires_confidence():
    result = decide(measurement(confidence=0.5), OperatingMode.AUTO_RADIUS, Settings())
    assert result.decision == Decision.UNCERTAIN


def test_large_growth_is_reviewable_but_not_automatically_applied():
    large = measurement(current=100, recommendation=190)
    assert decide(large, OperatingMode.RECOMMEND, Settings()).decision == Decision.RECOMMENDED
    assert decide(large, OperatingMode.AUTO_RADIUS, Settings()).decision == Decision.UNCERTAIN


def test_absence_requires_enabled_detection_prior_canopy_and_streak():
    absent = measurement()
    absent = absent.model_copy(
        update={
            "vegetation_absent": True,
            "absent_observations": 1,
            "recommended_protection_radius_mm": 0,
            "maximum_accepted_canopy_radius_mm": 0,
        }
    )
    enabled = Settings(removal_detection_enabled=True, removal_min_consecutive_absent=2)

    assert decide(absent, OperatingMode.RECOMMEND, Settings()).decision == Decision.OBSERVED
    assert (
        decide(absent, OperatingMode.RECOMMEND, enabled, previously_observed_canopy=True).decision
        == Decision.OBSERVED
    )
    confirmed = absent.model_copy(update={"absent_observations": 2})
    assert (
        decide(
            confirmed, OperatingMode.RECOMMEND, enabled, previously_observed_canopy=False
        ).decision
        == Decision.OBSERVED
    )
    assert (
        decide(
            confirmed, OperatingMode.RECOMMEND, enabled, previously_observed_canopy=True
        ).decision
        == Decision.REMOVAL_RECOMMENDED
    )


def test_removal_is_not_recommended_for_seedlings_or_small_plant_points():
    settings = Settings(removal_detection_enabled=True, removal_min_consecutive_absent=1)
    absent = measurement(current=100).model_copy(
        update={"vegetation_absent": True, "absent_observations": 1}
    )

    unknown_age = absent.model_copy(update={"plant_age_days": None})
    exactly_thirty_days = absent.model_copy(update={"plant_age_days": 30})
    exactly_fifty_mm = absent.model_copy(update={"plant_age_days": 31, "current_radius_mm": 50})

    for ineligible in (unknown_age, exactly_thirty_days, exactly_fifty_mm):
        result = decide(
            ineligible,
            OperatingMode.RECOMMEND,
            settings,
            previously_observed_canopy=True,
        )
        assert result.decision == Decision.OBSERVED

    eligible = absent.model_copy(update={"plant_age_days": 31, "current_radius_mm": 51})
    assert (
        decide(
            eligible,
            OperatingMode.RECOMMEND,
            settings,
            previously_observed_canopy=True,
        ).decision
        == Decision.REMOVAL_RECOMMENDED
    )


def test_confirmed_absence_auto_archives_only_when_enabled():
    absent = measurement().model_copy(update={"vegetation_absent": True, "absent_observations": 2})
    manual = Settings(removal_detection_enabled=True, removal_min_consecutive_absent=2)
    automatic = Settings(
        removal_detection_enabled=True,
        removal_min_consecutive_absent=2,
        removal_auto_apply=True,
    )

    assert (
        decide(absent, OperatingMode.AUTO_RADIUS, manual, previously_observed_canopy=True).decision
        == Decision.REMOVAL_RECOMMENDED
    )
    assert (
        decide(
            absent, OperatingMode.AUTO_RADIUS, automatic, previously_observed_canopy=True
        ).decision
        == Decision.REMOVED
    )


def test_confident_empty_center_can_remove_on_first_observation():
    absent = measurement(confidence=0.96).model_copy(
        update={
            "vegetation_absent": True,
            "center_misaligned": True,
            "absent_observations": 1,
            "recommended_protection_radius_mm": 0,
        }
    )
    settings = Settings(
        removal_detection_enabled=True,
        removal_min_consecutive_absent=2,
        removal_auto_apply=True,
    )
    result = decide(absent, OperatingMode.AUTO_RADIUS, settings, previously_observed_canopy=True)
    assert result.decision == Decision.REMOVED


def test_automatic_decision_threshold_applies_to_removal_too():
    absent = measurement(confidence=0.8).model_copy(
        update={
            "vegetation_absent": True,
            "center_misaligned": True,
            "absent_observations": 2,
            "recommended_protection_radius_mm": 0,
        }
    )
    settings = Settings(
        removal_detection_enabled=True,
        removal_auto_apply=True,
        minimum_auto_confidence=0.9,
    )
    result = decide(absent, OperatingMode.AUTO_RADIUS, settings, previously_observed_canopy=True)
    assert result.decision == Decision.REMOVAL_RECOMMENDED


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
