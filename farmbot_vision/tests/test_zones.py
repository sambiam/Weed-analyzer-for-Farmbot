from __future__ import annotations

import pytest

from farmbot_vision.zones import (
    Zone,
    ZoneAspect,
    ZoneKind,
    ZoneSet,
    ZoneShape,
    ZoneStore,
    evaluate,
    plant_center_allowed,
    plant_radius_allowed,
    weeds_allowed,
)


def _bed(**updates) -> Zone:
    values = {
        "name": "Bed",
        "kind": ZoneKind.BOUNDARY,
        "shape": ZoneShape.RECTANGLE,
        "min_x": 0,
        "min_y": 0,
        "max_x": 1000,
        "max_y": 1000,
        "allow_weeds": True,
        "allow_plant_centers": True,
        "allow_plant_radius": True,
    }
    values.update(updates)
    return Zone(**values)


def test_no_zones_allows_every_aspect():
    for aspect in ZoneAspect:
        verdict = evaluate([], aspect, 5_000, 5_000, 100)
        assert verdict.allowed
        assert "no zone restricts" in verdict.reason


def test_boundary_contains_and_excludes_points():
    zones = [_bed()]
    assert weeds_allowed(zones, 500, 500).allowed
    outside = weeds_allowed(zones, 1_500, 500)
    assert not outside.allowed
    assert "outside every boundary" in outside.reason


def test_boundary_may_forbid_a_single_aspect():
    zones = [_bed(allow_weeds=False)]
    assert not weeds_allowed(zones, 500, 500).allowed
    assert plant_center_allowed(zones, 500, 500).allowed
    assert plant_radius_allowed(zones, 500, 500, 50).allowed


def test_exclusion_zone_blocks_and_leaves_the_rest_of_the_garden_alone():
    zones = [
        Zone(
            name="Path",
            kind=ZoneKind.EXCLUSION,
            shape=ZoneShape.CIRCLE,
            center_x=200,
            center_y=200,
            radius_mm=100,
        )
    ]
    blocked = weeds_allowed(zones, 210, 210)
    assert not blocked.allowed
    assert blocked.zone_name == "Path"
    assert weeds_allowed(zones, 800, 800).allowed


def test_exclusion_allowance_overrides_a_surrounding_exclusion():
    zones = [
        Zone(
            name="Whole path",
            kind=ZoneKind.EXCLUSION,
            shape=ZoneShape.RECTANGLE,
            min_x=0,
            min_y=0,
            max_x=1000,
            max_y=200,
        ),
        Zone(
            name="Weeding strip",
            kind=ZoneKind.EXCLUSION,
            shape=ZoneShape.RECTANGLE,
            min_x=400,
            min_y=0,
            max_x=600,
            max_y=200,
            allow_weeds=True,
        ),
    ]
    permitted = weeds_allowed(zones, 500, 100)
    assert permitted.allowed
    assert permitted.zone_name == "Weeding strip"
    assert not weeds_allowed(zones, 100, 100).allowed


def test_radius_may_extend_past_the_boundary_but_must_avoid_exclusions():
    zones = [_bed()]
    assert plant_radius_allowed(zones, 500, 500, 200).allowed
    # The disc may reach past the boundary edge; only its centre must be inside.
    assert plant_center_allowed(zones, 950, 500).allowed
    assert plant_radius_allowed(zones, 950, 500, 100).allowed
    # The centre itself still has to fall inside the boundary.
    outside_center = plant_radius_allowed(zones, 1_050, 500, 100)
    assert not outside_center.allowed
    assert "is outside" in outside_center.reason

    zones.append(
        Zone(
            name="Post",
            kind=ZoneKind.EXCLUSION,
            shape=ZoneShape.CIRCLE,
            center_x=700,
            center_y=500,
            radius_mm=50,
        )
    )
    # The centre is clear of the post, but the protection radius reaches it.
    assert plant_center_allowed(zones, 600, 500).allowed
    overlapping = plant_radius_allowed(zones, 600, 500, 80)
    assert not overlapping.allowed
    assert overlapping.zone_name == "Post"
    assert plant_radius_allowed(zones, 300, 500, 80).allowed


def test_polygon_zone_uses_point_in_polygon_and_edges_count_as_inside():
    zone = Zone(
        name="Triangle",
        kind=ZoneKind.BOUNDARY,
        shape=ZoneShape.POLYGON,
        points=[(0, 0), (1000, 0), (0, 1000)],
        allow_weeds=True,
    )
    assert weeds_allowed([zone], 100, 100).allowed
    assert weeds_allowed([zone], 0, 500).allowed  # on an edge
    assert not weeds_allowed([zone], 900, 900).allowed


def test_disabled_zones_are_ignored():
    zones = [_bed(enabled=False, allow_weeds=False)]
    assert weeds_allowed(zones, 5_000, 5_000).allowed


def test_multiple_boundaries_allow_any_containing_one():
    zones = [_bed(name="Bed 1"), _bed(name="Bed 2", min_x=2_000, max_x=3_000)]
    assert weeds_allowed(zones, 2_500, 500).zone_name == "Bed 2"
    assert not weeds_allowed(zones, 1_500, 500).allowed


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        _bed(max_x=0)
    with pytest.raises(ValueError):
        Zone(name="c", kind=ZoneKind.EXCLUSION, shape=ZoneShape.CIRCLE, radius_mm=0)
    with pytest.raises(ValueError):
        Zone(
            name="p",
            kind=ZoneKind.EXCLUSION,
            shape=ZoneShape.POLYGON,
            points=[(0, 0), (1, 1)],
        )


def test_rectangle_corners_are_normalized_in_any_order():
    zone = _bed(min_x=1_000, max_x=0, min_y=800, max_y=200)
    assert (zone.min_x, zone.max_x, zone.min_y, zone.max_y) == (0, 1_000, 200, 800)
    assert zone.contains_point(500, 500)


def test_store_round_trips_and_edits_zones(tmp_path):
    store = ZoneStore(tmp_path / "zones.json")
    assert store.zones() == []

    zone = store.add(_bed(name="Bed 1"))
    assert [z.name for z in store.zones()] == ["Bed 1"]

    updated = store.update(zone.zone_id, allow_weeds=False, enabled=False)
    assert updated is not None
    assert updated.allow_weeds is False
    assert store.zones()[0].enabled is False
    assert store.update("missing", allow_weeds=True) is None

    assert store.delete(zone.zone_id) is True
    assert store.delete(zone.zone_id) is False
    assert store.zones() == []


def test_store_survives_a_corrupt_file_without_blocking_analysis(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text("{not json", encoding="utf-8")
    assert ZoneStore(path).load() == ZoneSet()
