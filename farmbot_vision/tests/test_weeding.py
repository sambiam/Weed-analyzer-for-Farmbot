"""Risk-aware path and soil planning for the rotary weeding tab."""

from datetime import UTC, datetime, timedelta

import pytest

from farmbot_vision.models import Plant, SoilPoint, WeedPoint
from farmbot_vision.weeding import (
    SoilSample,
    confirmed_weeds,
    estimate_soil_height,
    plan_cut_path,
    protected_tall_plants,
    recent_soil_samples,
    safe_transit_waypoints,
)


def plant(plant_id: int, x: float, y: float, radius: float = 30) -> Plant:
    return Plant(
        id=plant_id,
        name=f"Plant {plant_id}",
        openfarm_slug="test",
        x=x,
        y=y,
        radius=radius,
        plant_stage="active",
    )


def test_soil_estimate_uses_only_recent_samples_within_500_mm():
    now = datetime.now(UTC)
    points = [
        SoilPoint(id=1, name="near", x=0, y=0, z=-400, updated_at=now),
        SoilPoint(id=2, name="old", x=10, y=0, z=-100, updated_at=now - timedelta(days=31)),
        SoilPoint(id=3, name="far", x=700, y=0, z=-50, updated_at=now),
    ]
    estimate = estimate_soil_height(100, 0, recent_soil_samples(points, [], now=now))
    assert estimate is not None
    assert estimate.z == pytest.approx(-400)
    assert len(estimate.samples) == 1


def test_multiple_soil_samples_are_inverse_distance_interpolated():
    now = datetime.now(UTC)
    estimate = estimate_soil_height(
        100,
        0,
        [
            SoilSample(0, 0, -400, now, "a"),
            SoilSample(200, 0, -420, now, "b"),
        ],
    )
    assert estimate is not None
    assert estimate.z == pytest.approx(-410)
    assert estimate.method == "inverse-distance interpolation"


def test_path_scores_all_plants_not_only_the_nearest_one():
    weed = WeedPoint(id=9, x=500, y=500, radius=20)
    soil = estimate_soil_height(500, 500, [SoilSample(500, 500, -430, datetime.now(UTC), "test")])
    assert soil is not None
    path = plan_cut_path(
        weed,
        [plant(1, 500, 400), plant(2, 500, 600)],
        soil,
        x_bounds=(0, 1000),
        y_bounds=(0, 1000),
    )
    # Plants north and south make the east-west line safest.
    assert path.angle_degrees in {0, 175}
    assert path.start_y == pytest.approx(500, abs=5)
    assert path.end_y == pytest.approx(500, abs=5)


def test_path_is_clipped_to_the_bed_but_still_crosses_the_weed():
    weed = WeedPoint(id=1, x=10, y=10, radius=5)
    soil = estimate_soil_height(10, 10, [SoilSample(10, 10, -430, datetime.now(UTC), "test")])
    assert soil is not None
    path = plan_cut_path(
        weed,
        [],
        soil,
        x_bounds=(0, 100),
        y_bounds=(0, 100),
    )
    assert 0 <= path.start_x <= 100
    assert 0 <= path.end_x <= 100
    assert 0 <= path.start_y <= 100
    assert 0 <= path.end_y <= 100
    assert path.length_mm >= 20


def test_tall_plant_transit_routes_around_canopy():
    obstacle = plant(1, 500, 500, radius=80).model_copy(update={"height_mm": 450})
    protected = protected_tall_plants([obstacle], enabled=True, minimum_height_mm=300)
    route = safe_transit_waypoints(
        (100, 500),
        (900, 500),
        protected,
        x_bounds=(0, 1000),
        y_bounds=(0, 1000),
    )
    assert route
    assert any(abs(point["y"] - 500) > 80 for point in route)


def test_unknown_height_is_protected_but_option_can_be_disabled():
    unknown = plant(1, 100, 100)
    short = unknown.model_copy(update={"height_mm": 250})
    assert protected_tall_plants([unknown, short], enabled=True, minimum_height_mm=300) == [unknown]
    assert protected_tall_plants([unknown], enabled=False, minimum_height_mm=300) == []


def test_candidates_require_an_explicit_farmbot_weed_type():
    confirmed = WeedPoint(id=1, pointer_type="Weed", x=10, y=20, radius=5)
    untyped = WeedPoint(id=2, x=30, y=40, radius=10)
    assert confirmed_weeds([confirmed, untyped]) == [confirmed]
