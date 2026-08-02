from __future__ import annotations

from datetime import UTC, datetime, timedelta

from farmbot_vision.models import (
    CameraCalibration,
    Inventory,
    Plant,
    SoilMotionState,
    SoilPoint,
    SoilPointInventory,
    WeedPoint,
)
from farmbot_vision.soil_sites import plan_safe_soil_sites

NOW = datetime(2026, 7, 26, tzinfo=UTC)


def _soil(*points: SoilPoint) -> SoilPointInventory:
    return SoilPointInventory(
        device_id="42",
        generated_at=NOW,
        points=list(points),
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 0, "y": 0, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
        ),
    )


def _garden(*, plants: list[Plant] | None = None, weeds: list[WeedPoint] | None = None):
    return Inventory(
        device_id="42",
        generated_at=NOW,
        plants=plants or [],
        weeds=weeds or [],
        images=[],
        curves=[],
        camera_calibration=CameraCalibration(available=False),
    )


def _point(point_id: int, x: float, y: float, age_days: int = 15) -> SoilPoint:
    return SoilPoint(
        id=point_id,
        name=f"Soil {point_id}",
        x=x,
        y=y,
        z=-300,
        updated_at=NOW - timedelta(days=age_days),
    )


def test_safe_site_relocates_stale_point_beyond_plant_clearance():
    plant = Plant(
        id=1,
        name="Tomato",
        openfarm_slug="tomato",
        x=500,
        y=500,
        radius=50,
        plant_stage="planted",
    )
    sites = plan_safe_soil_sites(
        _soil(_point(70, 500, 500)),
        _garden(plants=[plant]),
        [],
        [],
        [],
        baseline_mm=15,
        now=NOW,
    )
    assert len(sites) == 1
    assert sites[0].point_id == 70
    assert sites[0].relocation_distance_mm < 200
    assert sites[0].relocation_distance_mm >= 155
    assert sites[0].clearance_mm >= 0


def test_fresh_unknown_and_too_distant_points_are_not_replaced():
    fresh = _point(70, 200, 200, age_days=10)
    unknown = _point(71, 400, 400)
    unknown.updated_at = None
    blocking_weed = WeedPoint(id=9, x=700, y=700, radius=500)
    blocked = _point(72, 700, 700)
    sites = plan_safe_soil_sites(
        _soil(fresh, unknown, blocked),
        _garden(weeds=[blocking_weed]),
        [],
        [],
        [],
        baseline_mm=15,
        now=NOW,
    )
    assert sites == []


def test_pending_vision_weed_is_an_obstacle():
    sites = plan_safe_soil_sites(
        _soil(_point(70, 300, 300)),
        _garden(),
        [],
        [{"x": 300, "y": 300, "radius_mm": 20}],
        [],
        baseline_mm=15,
        now=NOW,
    )
    assert len(sites) == 1
    assert sites[0].relocation_distance_mm >= 125


def test_latest_vision_canopy_is_an_obstacle():
    sites = plan_safe_soil_sites(
        _soil(_point(70, 300, 300)),
        _garden(),
        [{"x": 300, "y": 300, "radius_mm": 40}],
        [],
        [],
        baseline_mm=15,
        now=NOW,
    )
    assert len(sites) == 1
    assert sites[0].relocation_distance_mm >= 145


def test_reduced_clear_soil_margin_exposes_a_closer_candidate():
    plant = Plant(
        id=1, name="Tomato", openfarm_slug="tomato", x=300, y=300,
        radius=40, plant_stage="planted",
    )
    conservative = plan_safe_soil_sites(
        _soil(_point(70, 300, 300)), _garden(plants=[plant]), [], [], [],
        baseline_mm=15, clear_soil_margin_mm=75, now=NOW,
    )
    relaxed = plan_safe_soil_sites(
        _soil(_point(70, 300, 300)), _garden(plants=[plant]), [], [], [],
        baseline_mm=15, clear_soil_margin_mm=10, now=NOW,
    )
    assert relaxed[0].relocation_distance_mm < conservative[0].relocation_distance_mm
