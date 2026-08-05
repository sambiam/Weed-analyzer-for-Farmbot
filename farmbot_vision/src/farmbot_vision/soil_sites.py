"""Select clear-soil capture sites and assign them to stale FarmBot points."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from .models import Inventory, SoilGridPlan, SoilGridPoint, SoilPoint, SoilPointInventory, SoilSite
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


def _obstacles(
    garden: Inventory,
    vision_plants: list[dict],
    vision_weeds: list[dict],
) -> list[tuple[float, float, float]]:
    obstacles = [(plant.x, plant.y, max(0.0, plant.radius)) for plant in garden.plants]
    obstacles.extend((weed.x, weed.y, max(0.0, weed.radius)) for weed in garden.weeds)
    for detection in [*vision_plants, *vision_weeds]:
        try:
            obstacles.append(
                (
                    float(detection["x"]),
                    float(detection["y"]),
                    max(0.0, float(detection.get("radius_mm") or 0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return obstacles


def _available_clear_soil_margin(
    x: float,
    y: float,
    obstacles: list[tuple[float, float, float]],
    zones: list[Zone],
    baseline_mm: float,
) -> float:
    """Return the largest clear-soil margin supported at one coordinate."""

    stereo_reach = 2 * baseline_mm
    obstacle_limit = min(
        (
            math.hypot(x - obstacle_x, y - obstacle_y) - obstacle_radius - stereo_reach
            for obstacle_x, obstacle_y, obstacle_radius in obstacles
        ),
        default=10_000.0,
    )
    if obstacle_limit < 0 or not evaluate(zones, ZoneAspect.CENTERS, x, y, stereo_reach).allowed:
        return -1.0
    if not zones:
        return obstacle_limit
    low, high = 0.0, min(10_000.0, obstacle_limit)
    if evaluate(zones, ZoneAspect.CENTERS, x, y, high + stereo_reach).allowed:
        return high
    for _ in range(24):
        middle = (low + high) / 2
        if evaluate(zones, ZoneAspect.CENTERS, x, y, middle + stereo_reach).allowed:
            low = middle
        else:
            high = middle
    return low


def _axis_grid(spacing_mm: float, minimum: float, maximum: float) -> list[float]:
    """Start at half a spacing from zero, as FarmBot grid sequences do."""

    if maximum < spacing_mm / 2:
        return []
    count = math.floor((maximum - spacing_mm / 2) / spacing_mm) + 1
    return [
        spacing_mm / 2 + index * spacing_mm
        for index in range(count)
        if spacing_mm / 2 + index * spacing_mm >= minimum
    ]


def _nearest_anchor(
    grid_x: float,
    grid_y: float,
    points: list[SoilPoint],
    assigned: set[int],
) -> SoilPoint | None:
    choices = [point for point in points if point.id not in assigned]
    if not choices:
        return None
    point = min(
        choices,
        key=lambda item: (math.hypot(item.x - grid_x, item.y - grid_y), item.id),
    )
    return point if math.hypot(point.x - grid_x, point.y - grid_y) < MAX_RELOCATION_MM else None


def plan_soil_measurement_grid(
    soil: SoilPointInventory,
    garden: Inventory,
    vision_plants: list[dict],
    vision_weeds: list[dict],
    zones: list[Zone],
    *,
    spacing_mm: float,
    maximum_deviation_mm: float,
    baseline_mm: float,
    clear_soil_margin_mm: float = DEFAULT_SOIL_CLEARANCE_MM,
    now: datetime | None = None,
) -> SoilGridPlan:
    """Plan every half-spacing-offset grid point and explain every omission.

    Existing FarmBot soil points are the update records because the companion
    contract deliberately cannot create points. Each record is assigned once,
    then the nearest sampled clear location within the user's deviation is used.
    """

    values = (spacing_mm, maximum_deviation_mm, baseline_mm, clear_soil_margin_mm)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("soil grid settings must be finite numbers")
    if not 50 <= spacing_mm <= 5000:
        raise ValueError("soil grid spacing must be from 50 to 5000 mm")
    if not 0 <= maximum_deviation_mm < MAX_RELOCATION_MM:
        raise ValueError("maximum grid deviation must be from 0 up to, but not including, 200 mm")
    if baseline_mm <= 0 or clear_soil_margin_mm < 0:
        raise ValueError("baseline and clear-soil margin must be positive")
    x_bounds = soil.motion.axis_bounds.get("x")
    y_bounds = soil.motion.axis_bounds.get("y")
    if x_bounds is None or y_bounds is None:
        return SoilGridPlan(
            spacing_mm=spacing_mm,
            maximum_deviation_mm=maximum_deviation_mm,
            clear_soil_margin_mm=clear_soil_margin_mm,
            points=[],
        )
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    grid_x_values = _axis_grid(spacing_mm, x_min, x_max)
    grid_y_values = _axis_grid(spacing_mm, y_min, y_max)
    if len(grid_x_values) * len(grid_y_values) > 10_000:
        raise ValueError("soil grid is too large; increase spacing to keep it under 10,000 points")
    now = _utc(now or datetime.now(UTC))
    cutoff = now - STALE_AFTER
    obstacles = _obstacles(garden, vision_plants, vision_weeds)
    offsets = _OFFSETS
    assigned: set[int] = set()
    used_sites: set[tuple[float, float]] = set()
    result: list[SoilGridPoint] = []

    for grid_y in grid_y_values:
        for grid_x in grid_x_values:
            anchor = _nearest_anchor(grid_x, grid_y, soil.points, assigned)
            if anchor is None:
                result.append(
                    SoilGridPoint(
                        grid_x=grid_x,
                        grid_y=grid_y,
                        status="skipped",
                        explanation=(
                            "No unassigned FarmBot soil-height point is within the 200 mm "
                            "integration limit. Create a soil-height point near this grid location."
                        ),
                    )
                )
                continue
            assigned.add(anchor.id)
            common = {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "point_id": anchor.id,
                "point_name": anchor.name,
                "expected_x": anchor.x,
                "expected_y": anchor.y,
                "expected_z": anchor.z,
                "point_updated_at": anchor.updated_at,
            }
            if anchor.updated_at is None:
                result.append(
                    SoilGridPoint(
                        **common,
                        status="skipped",
                        explanation=(
                            "The nearest soil-height point has no trustworthy update date and "
                            "cannot be safely replaced."
                        ),
                    )
                )
                continue
            if _utc(anchor.updated_at) >= cutoff:
                result.append(
                    SoilGridPoint(
                        **common,
                        status="skipped",
                        explanation=(
                            "The nearest soil-height point is less than 14 days old and is not "
                            "eligible for replacement yet."
                        ),
                    )
                )
                continue

            candidates: list[tuple[float, float, float, float]] = []
            for distance, dx, dy in offsets:
                x, y = grid_x + dx, grid_y + dy
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    continue
                if not _triplet_fits(y, baseline_mm, y_min, y_max):
                    continue
                if math.hypot(x - anchor.x, y - anchor.y) >= MAX_RELOCATION_MM:
                    continue
                capacity = _available_clear_soil_margin(x, y, obstacles, zones, baseline_mm)
                candidates.append((distance, x, y, capacity))

            allowed = [
                item
                for item in candidates
                if item[0] <= maximum_deviation_mm + 1e-6
                and item[3] + 1e-6 >= clear_soil_margin_mm
                and (round(item[1], 3), round(item[2], 3)) not in used_sites
            ]
            if allowed:
                distance, capture_x, capture_y, capacity = allowed[0]
                used_sites.add((round(capture_x, 3), round(capture_y, 3)))
                result.append(
                    SoilGridPoint(
                        **common,
                        capture_x=capture_x,
                        capture_y=capture_y,
                        deviation_mm=distance,
                        clearance_mm=max(0.0, capacity - clear_soil_margin_mm),
                        status="clear" if distance < 0.001 else "replaced",
                        explanation=(
                            "Nominal grid point has clear soil."
                            if distance < 0.001
                            else f"Replaced by the nearest clear point {distance:.0f} mm away."
                        ),
                    )
                )
                continue

            nearest_clear = next(
                (
                    item
                    for item in candidates
                    if item[3] + 1e-6 >= clear_soil_margin_mm
                    and (round(item[1], 3), round(item[2], 3)) not in used_sites
                ),
                None,
            )
            if nearest_clear is not None:
                explanation = (
                    f"Nearest clear point is {nearest_clear[0]:.0f} mm away; increase maximum "
                    f"deviation to at least {math.ceil(nearest_clear[0])} mm to include it."
                )
            else:
                margin_candidates = [
                    item
                    for item in candidates
                    if item[0] <= maximum_deviation_mm + 1e-6 and item[3] >= 0
                ]
                best_margin = max((item[3] for item in margin_candidates), default=-1)
                if best_margin >= 0:
                    explanation = (
                        f"Reduce clear-soil margin to {math.floor(best_margin):.0f} mm or less "
                        "to include the best nearby location."
                    )
                else:
                    explanation = (
                        "No clear-soil location is available within the deviation and the "
                        "200 mm integration limit, even with a zero clear-soil margin."
                    )
            result.append(SoilGridPoint(**common, status="skipped", explanation=explanation))

    return SoilGridPlan(
        spacing_mm=spacing_mm,
        maximum_deviation_mm=maximum_deviation_mm,
        clear_soil_margin_mm=clear_soil_margin_mm,
        points=result,
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
