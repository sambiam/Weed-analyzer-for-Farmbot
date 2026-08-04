"""Risk-aware straight-line planning and recent soil-height interpolation."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .models import Plant, SoilPoint, WeedPoint

SOIL_SEARCH_RADIUS_MM = 500.0
DEFAULT_SOIL_MAX_AGE_DAYS = 30.0
PATH_ANGLE_STEP_DEGREES = 5
MIN_PATH_LENGTH_MM = 100.0
PATH_OVERHANG_MM = 30.0
MAX_PATH_LENGTH_MM = 220.0
PLANT_MARGIN_MM = 25.0
TRANSIT_MARGIN_MM = 40.0
TRANSIT_CIRCLE_POINTS = 16


@dataclass(frozen=True, slots=True)
class SoilSample:
    x: float
    y: float
    z: float
    measured_at: datetime
    source: str


@dataclass(frozen=True, slots=True)
class SoilEstimate:
    z: float
    samples: tuple[SoilSample, ...]
    nearest_distance_mm: float
    spread_mm: float
    method: str


@dataclass(frozen=True, slots=True)
class CutPath:
    weed_id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    angle_degrees: float
    length_mm: float
    minimum_plant_clearance_mm: float
    soil_z: float
    soil_method: str


def confirmed_weeds(points: Iterable[WeedPoint]) -> list[WeedPoint]:
    """Keep only inventory records explicitly typed as FarmBot Weed points."""
    return [point for point in points if point.pointer_type == "Weed"]


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def recent_soil_samples(
    points: Iterable[SoilPoint],
    measurements: Iterable[dict],
    *,
    now: datetime | None = None,
    max_age_days: float | None = DEFAULT_SOIL_MAX_AGE_DAYS,
) -> list[SoilSample]:
    """Merge FarmBot points and valid in-app captures without double counting.

    Applied measurements use their measured clear-soil coordinate. A newer
    valid measurement is useful to weeding even when the user has not chosen
    to write it back to the FarmBot soil point yet.
    """
    cutoff = (
        _utc(now or datetime.now(UTC)) - timedelta(days=max_age_days)
        if max_age_days is not None
        else None
    )
    samples: list[SoilSample] = []
    measured_point_ids: set[int] = set()
    for row in measurements:
        if row.get("status") not in {"valid", "applied"} or row.get("proposed_z_mm") is None:
            continue
        try:
            created = _utc(datetime.fromisoformat(str(row["created_at"])))
            x = float(
                row.get("capture_x") if row.get("capture_x") is not None else row["expected_x"]
            )
            y = float(
                row.get("capture_y") if row.get("capture_y") is not None else row["expected_y"]
            )
            z = float(row["proposed_z_mm"])
        except (KeyError, TypeError, ValueError):
            continue
        if (cutoff is not None and created < cutoff) or not all(
            math.isfinite(value) for value in (x, y, z)
        ):
            continue
        samples.append(SoilSample(x, y, z, created, "vision measurement"))
        measured_point_ids.add(int(row["point_id"]))
    for point in points:
        if point.id in measured_point_ids or point.updated_at is None:
            continue
        updated = _utc(point.updated_at)
        if cutoff is None or updated >= cutoff:
            samples.append(SoilSample(point.x, point.y, point.z, updated, "FarmBot soil point"))
    return samples


def estimate_soil_height(x: float, y: float, samples: Iterable[SoilSample]) -> SoilEstimate | None:
    """Inverse-distance interpolate recent samples within 500 mm.

    Squared inverse-distance weighting gives nearby measurements authority
    while avoiding an unstable plane extrapolation when all points lie on one
    side of a weed. Exact-coordinate samples win outright.
    """
    nearby = []
    for sample in samples:
        distance = math.hypot(sample.x - x, sample.y - y)
        if distance <= SOIL_SEARCH_RADIUS_MM:
            nearby.append((distance, sample))
    if not nearby:
        return None
    nearby.sort(key=lambda item: item[0])
    if nearby[0][0] < 0.5:
        chosen = (nearby[0][1],)
        return SoilEstimate(chosen[0].z, chosen, nearby[0][0], 0.0, "exact recent sample")
    weights = [1.0 / max(distance, 1.0) ** 2 for distance, _ in nearby]
    total = sum(weights)
    value = (
        sum(weight * sample.z for weight, (_, sample) in zip(weights, nearby, strict=True)) / total
    )
    spread = math.sqrt(
        sum(
            weight * (sample.z - value) ** 2
            for weight, (_, sample) in zip(weights, nearby, strict=True)
        )
        / total
    )
    return SoilEstimate(
        value,
        tuple(sample for _, sample in nearby),
        nearby[0][0],
        spread,
        "inverse-distance interpolation" if len(nearby) > 1 else "nearest recent sample",
    )


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def protected_tall_plants(
    plants: Iterable[Plant], *, enabled: bool, minimum_height_mm: float
) -> list[Plant]:
    """Return plants a mounted tool must route around.

    Unknown heights are protected conservatively: absence of height data must
    never make a mounted cutting tool assume that a plant is short.
    """
    if not enabled:
        return []
    return [
        plant for plant in plants if plant.height_mm is None or plant.height_mm > minimum_height_mm
    ]


def safe_transit_waypoints(
    start: tuple[float, float],
    end: tuple[float, float],
    plants: Iterable[Plant],
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    margin_mm: float = TRANSIT_MARGIN_MM,
    endpoint_margin_mm: float | None = None,
) -> list[dict[str, float]]:
    """Find the shortest visible route around protected plant circles.

    Transit normally uses the larger mounted-tool margin. A cut endpoint may
    legitimately sit inside that conservative travel buffer while still
    satisfying the cutting clearance used by ``plan_cut_path``. For a plant
    whose buffer contains either endpoint, reduce only that plant's routing
    circle to ``endpoint_margin_mm``; all other plants keep the full transit
    margin. Endpoints inside the reduced cutting clearance remain forbidden.
    """
    circles = []
    for plant in plants:
        radius = plant.radius + margin_mm
        if endpoint_margin_mm is not None and (
            math.hypot(start[0] - plant.x, start[1] - plant.y) < radius
            or math.hypot(end[0] - plant.x, end[1] - plant.y) < radius
        ):
            radius = plant.radius + endpoint_margin_mm
        circles.append((plant.x, plant.y, radius))

    def inside_bounds(point: tuple[float, float]) -> bool:
        return x_bounds[0] <= point[0] <= x_bounds[1] and y_bounds[0] <= point[1] <= y_bounds[1]

    def segment_clear(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return all(
            _point_segment_distance(cx, cy, a[0], a[1], b[0], b[1]) >= radius - 1e-6
            for cx, cy, radius in circles
        )

    if not inside_bounds(start) or not inside_bounds(end):
        raise ValueError("mounted-tool transit endpoint is outside the bed")
    if any(math.hypot(start[0] - cx, start[1] - cy) < radius for cx, cy, radius in circles):
        raise ValueError("mounted-tool transit starts inside a tall plant clearance")
    if any(math.hypot(end[0] - cx, end[1] - cy) < radius for cx, cy, radius in circles):
        raise ValueError("weed approach lies inside a tall plant clearance")
    if segment_clear(start, end):
        return []

    nodes = [start, end]
    for cx, cy, radius in circles:
        # Keep polygon chords outside the actual circle, not just their nodes.
        waypoint_radius = radius / math.cos(math.pi / TRANSIT_CIRCLE_POINTS) + 2.0
        for index in range(TRANSIT_CIRCLE_POINTS):
            angle = math.tau * index / TRANSIT_CIRCLE_POINTS
            point = (cx + math.cos(angle) * waypoint_radius, cy + math.sin(angle) * waypoint_radius)
            if inside_bounds(point) and all(
                math.hypot(point[0] - ox, point[1] - oy) >= other_radius - 1e-6
                for ox, oy, other_radius in circles
            ):
                nodes.append(point)

    graph: list[list[tuple[float, int]]] = [[] for _ in nodes]
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if segment_clear(nodes[left], nodes[right]):
                distance = math.dist(nodes[left], nodes[right])
                graph[left].append((distance, right))
                graph[right].append((distance, left))
    distances = [math.inf] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == 1:
            break
        for edge, neighbour in graph[node]:
            candidate = distance + edge
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    if not math.isfinite(distances[1]):
        raise ValueError("no in-bounds mounted-tool route avoids tall plants")
    route = []
    node: int | None = 1
    while node is not None:
        route.append(node)
        node = previous[node]
    route.reverse()
    return [{"x": nodes[index][0], "y": nodes[index][1]} for index in route[1:-1]]


def _bounded_segment(
    weed: WeedPoint,
    angle_radians: float,
    length_mm: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    ux, uy = math.cos(angle_radians), math.sin(angle_radians)
    half = length_mm / 2
    # Scale both halves uniformly to remain inside the configured bed while
    # keeping the weed at the centre of the cut.
    scale = 1.0
    for coordinate, direction, bounds in (
        (weed.x, ux, x_bounds),
        (weed.y, uy, y_bounds),
    ):
        if abs(direction) < 1e-9:
            continue
        for sign in (-1.0, 1.0):
            target = coordinate + sign * direction * half
            if target < bounds[0]:
                scale = min(scale, (coordinate - bounds[0]) / (half * abs(direction)))
            elif target > bounds[1]:
                scale = min(scale, (bounds[1] - coordinate) / (half * abs(direction)))
    if scale * length_mm < max(20.0, weed.radius * 2):
        return None
    half *= max(0.0, scale)
    return (
        min(x_bounds[1], max(x_bounds[0], weed.x - ux * half)),
        min(y_bounds[1], max(y_bounds[0], weed.y - uy * half)),
        min(x_bounds[1], max(x_bounds[0], weed.x + ux * half)),
        min(y_bounds[1], max(y_bounds[0], weed.y + uy * half)),
    )


def plan_cut_path(
    weed: WeedPoint,
    plants: Iterable[Plant],
    soil: SoilEstimate,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    plant_margin_mm: float = PLANT_MARGIN_MM,
) -> CutPath:
    """Choose the lowest-risk straight line through a weed.

    Every 5-degree candidate is scored against every plant protection circle,
    rather than only the nearest plant. Maximising the worst clearance avoids
    a line that is perpendicular to one plant but points directly at another;
    summed near-plant penalties break ties smoothly.
    """
    plants = list(plants)
    desired = min(
        MAX_PATH_LENGTH_MM, max(MIN_PATH_LENGTH_MM, weed.radius * 2 + 2 * PATH_OVERHANG_MM)
    )
    best: tuple[tuple[float, float, float], tuple[float, float, float, float], float] | None = None
    for degrees in range(0, 180, PATH_ANGLE_STEP_DEGREES):
        segment = _bounded_segment(weed, math.radians(degrees), desired, x_bounds, y_bounds)
        if segment is None:
            continue
        ax, ay, bx, by = segment
        clearances = [
            _point_segment_distance(plant.x, plant.y, ax, ay, bx, by)
            - plant.radius
            - plant_margin_mm
            for plant in plants
        ]
        minimum = min(clearances, default=10_000.0)
        penalty = sum(max(0.0, 150.0 - clearance) ** 2 for clearance in clearances)
        actual_length = math.hypot(bx - ax, by - ay)
        score = (minimum, -penalty, actual_length)
        if minimum >= 0 and (best is None or score > best[0]):
            best = (score, segment, float(degrees))
    if best is None:
        # A full cut can be blocked even though the centre remains safely
        # reachable. Approach from the open side and stop at the weed centre;
        # this avoids sending the cutter through the crop-facing half.
        partial_length = max(20.0, weed.radius + PATH_OVERHANG_MM)
        partial_best: (
            tuple[tuple[float, float, float], tuple[float, float, float, float], float] | None
        ) = None
        for degrees in range(0, 360, PATH_ANGLE_STEP_DEGREES):
            angle = math.radians(degrees)
            ux, uy = math.cos(angle), math.sin(angle)
            available = partial_length
            for coordinate, direction, bounds in (
                (weed.x, ux, x_bounds),
                (weed.y, uy, y_bounds),
            ):
                if abs(direction) < 1e-9:
                    continue
                distance = (
                    (coordinate - bounds[0]) / direction
                    if direction > 0
                    else (bounds[1] - coordinate) / -direction
                )
                available = min(available, distance)
            if available < 20.0:
                continue
            ax, ay = weed.x - ux * available, weed.y - uy * available
            bx, by = weed.x, weed.y
            clearances = [
                _point_segment_distance(plant.x, plant.y, ax, ay, bx, by)
                - plant.radius
                - plant_margin_mm
                for plant in plants
            ]
            minimum = min(clearances, default=10_000.0)
            if minimum < 0:
                continue
            penalty = sum(max(0.0, 150.0 - clearance) ** 2 for clearance in clearances)
            score = (minimum, -penalty, available)
            if partial_best is None or score > partial_best[0]:
                partial_best = (score, (ax, ay, bx, by), float(degrees % 180))
        if partial_best is None:
            raise ValueError(f"weed {weed.id} cannot be reached without entering plant clearance")
        best = partial_best
    score, (ax, ay, bx, by), degrees = best
    return CutPath(
        weed_id=weed.id,
        start_x=ax,
        start_y=ay,
        end_x=bx,
        end_y=by,
        angle_degrees=degrees,
        length_mm=math.hypot(bx - ax, by - ay),
        minimum_plant_clearance_mm=score[0],
        soil_z=soil.z,
        soil_method=soil.method,
    )


def nearest_neighbour_order(paths: Iterable[CutPath], x: float, y: float) -> list[CutPath]:
    """Order cuts to reduce safe-height travel without changing their geometry."""
    remaining, ordered = list(paths), []
    while remaining:
        chosen = min(
            remaining,
            key=lambda path: min(
                math.hypot(path.start_x - x, path.start_y - y),
                math.hypot(path.end_x - x, path.end_y - y),
            ),
        )
        remaining.remove(chosen)
        # Orient this first pass from the closer endpoint. Recovery alternates
        # endpoints inside the integration.
        if math.hypot(chosen.end_x - x, chosen.end_y - y) < math.hypot(
            chosen.start_x - x, chosen.start_y - y
        ):
            chosen = CutPath(
                chosen.weed_id,
                chosen.end_x,
                chosen.end_y,
                chosen.start_x,
                chosen.start_y,
                chosen.angle_degrees,
                chosen.length_mm,
                chosen.minimum_plant_clearance_mm,
                chosen.soil_z,
                chosen.soil_method,
            )
        ordered.append(chosen)
        x, y = chosen.end_x, chosen.end_y
    return ordered
