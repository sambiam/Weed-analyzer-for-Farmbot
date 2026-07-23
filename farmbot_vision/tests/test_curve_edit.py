from __future__ import annotations

from farmbot_vision.curve_edit import (
    MAX_CURVE_CONTROL_POINTS,
    propose_curve_point,
    reasonableness_gate,
)


def test_append_measurement_beyond_existing_max_day() -> None:
    edit = propose_curve_point({"1": 5, "10": 25, "80": 91}, 101, 100)

    assert edit.data == {"1": 5.0, "10": 25.0, "80": 91.0, "101": 100.0}
    assert edit.max_day_changed is True
    assert edit.verdict == "ok"
    assert edit.downsampled is False


def test_day_81_spike_is_flagged_with_conflicting_previous_point() -> None:
    existing = {"1": 5, "10": 25, "80": 91}
    verdict = reasonableness_gate(
        existing,
        81,
        400,
        max_daily_growth_mm=50,
        maximum_plant_radius_mm=300,
    )
    edit = propose_curve_point(
        existing,
        81,
        400,
        max_daily_growth_mm=50,
        maximum_plant_radius_mm=300,
    )

    assert verdict.verdict == "flagged"
    assert verdict.conflict_day == 80
    assert verdict.conflict_old_diameter == 91
    assert "daily growth" in (verdict.reason or "")
    assert edit.data["81"] == 400
    assert edit.verdict == "flagged"
    assert edit.conflict_day == 80


def test_monotonicity_flags_a_decrease_from_previous_and_an_excess_before_next() -> None:
    existing = {"0": 10, "10": 30, "20": 50}

    lower_conflict = reasonableness_gate(
        existing, 15, 20, max_daily_growth_mm=50, maximum_plant_radius_mm=100
    )
    upper_conflict = reasonableness_gate(
        existing, 15, 60, max_daily_growth_mm=50, maximum_plant_radius_mm=100
    )

    assert (lower_conflict.verdict, lower_conflict.conflict_day) == ("flagged", 10)
    assert (upper_conflict.verdict, upper_conflict.conflict_day) == ("flagged", 20)


def test_absolute_radius_limit_is_flagged_without_a_control_point_conflict() -> None:
    verdict = reasonableness_gate(
        {"0": 10}, 1, 201, max_daily_growth_mm=500, maximum_plant_radius_mm=100
    )

    assert verdict.verdict == "flagged"
    assert verdict.conflict_day is None
    assert "maximum plant radius" in (verdict.reason or "")


def test_downsampling_keeps_endpoints_and_new_interior_measurement() -> None:
    existing = {str(day): float(day * 2) for day in range(12)}
    edit = propose_curve_point(existing, 5, 123)

    assert len(edit.data) == MAX_CURVE_CONTROL_POINTS
    assert edit.downsampled is True
    assert edit.warnings
    assert edit.data["0"] == 0
    assert edit.data["5"] == 123
    assert edit.data["11"] == 22


def test_replacement_does_not_claim_to_extend_max_day() -> None:
    edit = propose_curve_point({"0": 10, "10": 20}, 10, 25)

    assert edit.data == {"0": 10.0, "10": 25.0}
    assert edit.max_day_changed is False


def test_edited_value_is_revalidated_and_can_clear_a_flag() -> None:
    existing = {"80": 91}
    flagged = reasonableness_gate(
        existing, 81, 400, max_daily_growth_mm=50, maximum_plant_radius_mm=300
    )
    edited = reasonableness_gate(
        existing, 81, 100, max_daily_growth_mm=50, maximum_plant_radius_mm=300
    )

    assert flagged.verdict == "flagged"
    assert edited.verdict == "ok"


def test_empty_curve_is_a_valid_first_control_point() -> None:
    edit = propose_curve_point({}, 0, 20)

    assert edit.data == {"0": 20.0}
    assert edit.max_day_changed is True
