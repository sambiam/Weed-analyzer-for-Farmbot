"""Select clear-soil capture sites and assign them to stale FarmBot points."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from .models import Inventory, SoilPointInventory, SoilSite
from .zones import Zone, ZoneAspect, evaluate

STALE_AFTER = timedelta(days=14)
MAX_RELOCATION_MM = 200.0
GRID_SPACING_MM = 25.0
DEFAULT_SOIL_CLEARANCE_MM = 75.0


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _candidate_offsets() -> list[tuple[float, float, float]]:
    limit = math.ceil(MAX_RELOCATION_MM / GRID_SPACING_MM)
    offsets = []
    for ix in range(-limit, limit + 1):
        for iy in range(-limit, limit + 1):
            dx, dy = ix * GRID_SPACING_MM, iy * GRID_SPACING_MM
            distance = math.hypot(dx, dy)
            if distance < MAX_RELOCATION_MM:
                offsets.append((distance, dx, dy))
    return sorted(offsets, key=lambda item: (item[0], abs(item[1]) + abs(item[2])))


_OFFSETS = _candidate_offsets()


def _triplet_fits(y: float, baseline_mm: float, y_min: float, y_max: float) -> bool:
    return (
        (y - baseline_mm >= y_min and y + baseline_mm <= y_max)
        or y + 2 * baseline_mm <= y_max
        or y - 2 * baseline_mm >= y_min
    )


def plan_safe_soil_sites(
    soil: SoilPointInventory,
    garden: Inventory,
    vision_plants: list[dict],
    vision_weeds: list[dict],
    zones: list[Zone],
    *,
    baseline_mm: float,
    clear_soil_margin_mm: float = DEFAULT_SOIL_CLEARANCE_MM,
    now: datetime | None = None,
) -> list[SoilSite]:
    """Return one nearest clear site for each eligible stale soil point.

    The configured soil patch margin plus the worst-case one-sided stereo baseline must not
    overlap any active FarmBot or Vision plant, FarmBot weed, Vision weed, or
    forbidden zone. Points without a trustworthy ``updated_at`` are not
    silently relocated.
    """

    now = _utc(now or datetime.now(UTC))
    cutoff = now - STALE_AFTER
    x_bounds = soil.motion.axis_bounds.get("x")
    y_bounds = soil.motion.axis_bounds.get("y")
    if x_bounds is None or y_bounds is None:
        return []
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    if not math.isfinite(clear_soil_margin_mm) or clear_soil_margin_mm < 0:
        raise ValueError("clear-soil margin must be a finite non-negative number")
    safety_margin = clear_soil_margin_mm + 2 * baseline_mm

    obstacles: list[tuple[float, float, float]] = [
        (plant.x, plant.y, max(0.0, plant.radius) + safety_margin) for plant in garden.plants
    ]
    obstacles.extend(
        (weed.x, weed.y, max(0.0, weed.radius) + safety_margin) for weed in garden.weeds
    )
    for detection in [*vision_plants, *vision_weeds]:
        try:
            obstacles.append(
                (
                    float(detection["x"]),
                    float(detection["y"]),
                    max(0.0, float(detection.get("radius_mm") or 0)) + safety_margin,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    stale = [
        point
        for point in soil.points
        if point.updated_at is not None and _utc(point.updated_at) < cutoff
    ]
    options: list[tuple[float, int, float, float, float]] = []
    for point in stale:
        for distance, dx, dy in _OFFSETS:
            x, y = point.x + dx, point.y + dy
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            if not _triplet_fits(y, baseline_mm, y_min, y_max):
                continue
            if not evaluate(zones, ZoneAspect.CENTERS, x, y, safety_margin).allowed:
                continue
            clearance = min(
                (
                    math.hypot(x - obstacle_x, y - obstacle_y) - obstacle_radius
                    for obstacle_x, obstacle_y, obstacle_radius in obstacles
                ),
                default=10_000.0,
            )
            if clearance < 0:
                continue
            options.append((distance, point.id, x, y, clearance))

    by_id = {point.id: point for point in stale}
    assigned_points: set[int] = set()
    assigned_sites: set[tuple[float, float]] = set()
    result = []
    for distance, point_id, x, y, clearance in sorted(options):
        key = (round(x, 3), round(y, 3))
        if point_id in assigned_points or key in assigned_sites:
            continue
        point = by_id[point_id]
        result.append(
            SoilSite(
                point_id=point.id,
                point_name=point.name,
                expected_x=point.x,
                expected_y=point.y,
                expected_z=point.z,
                point_updated_at=point.updated_at,
                capture_x=x,
                capture_y=y,
                relocation_distance_mm=distance,
                clearance_mm=clearance,
            )
        )
        assigned_points.add(point_id)
        assigned_sites.add(key)
    return sorted(result, key=lambda site: (site.capture_x, site.capture_y, site.point_id))
