"""Garden boundaries and exclusion zones.

A zone is an area of the garden, in FarmBot garden millimetres, that either
contains what is allowed (a **boundary**) or marks what must be avoided (an
**exclusion zone**). Each zone states, independently, whether three kinds of
placement are permitted inside it:

* ``weeds`` -- whether a detected weed may be created there;
* ``centers`` -- whether a plant centre may be moved there;
* ``radius`` -- whether a plant's protection radius may extend into it.

Decisions follow one fixed precedence, so overlapping zones are predictable:

1. an exclusion zone that *allows* the aspect and contains the geometry is an
   explicit exception and wins outright;
2. an exclusion zone that *forbids* the aspect and is touched by the geometry
   denies it;
3. a boundary that *forbids* the aspect and is touched by the geometry denies
   it (a hole inside an otherwise permitted boundary);
4. if any boundary allows the aspect, the geometry must fit inside one of them;
5. otherwise -- no zones defined for that aspect -- placement is allowed, so an
   empty configuration keeps the previous behaviour.

"Touched" and "fits inside" are point tests for weeds and plant centres. For
the radius aspect, a boundary only ever tests the centre point -- a plant's
protection disc is free to extend past a boundary's edge, since it is the
plant (and the weed) that must stay inside the growing area, not its
clearance disc. Exclusion zones are different: they mark real hazards, so the
full protection disc must not overlap a forbidding exclusion zone, and must
fit entirely inside an exclusion zone that explicitly allows it.
"""

from __future__ import annotations

import math
import os
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ZoneKind(str, Enum):
    BOUNDARY = "boundary"
    EXCLUSION = "exclusion"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ZoneShape(str, Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ZoneAspect(str, Enum):
    WEEDS = "weeds"
    CENTERS = "centers"
    RADIUS = "radius"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


ASPECT_LABELS = {
    ZoneAspect.WEEDS: "weeds",
    ZoneAspect.CENTERS: "plant centres",
    ZoneAspect.RADIUS: "plant protection radius",
}

_ASPECT_FIELDS = {
    ZoneAspect.WEEDS: "allow_weeds",
    ZoneAspect.CENTERS: "allow_plant_centers",
    ZoneAspect.RADIUS: "allow_plant_radius",
}


class Zone(BaseModel):
    """One boundary or exclusion zone in garden coordinates (millimetres)."""

    zone_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=60)
    kind: ZoneKind
    shape: ZoneShape
    enabled: bool = True
    allow_weeds: bool = False
    allow_plant_centers: bool = False
    allow_plant_radius: bool = False
    # Rectangle: opposite corners, normalized so min <= max.
    min_x: float = 0
    min_y: float = 0
    max_x: float = 0
    max_y: float = 0
    # Circle.
    center_x: float = 0
    center_y: float = 0
    radius_mm: float = 0
    # Polygon: at least three vertices.
    points: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_geometry(self) -> Zone:
        if self.shape is ZoneShape.RECTANGLE:
            min_x, max_x = sorted((self.min_x, self.max_x))
            min_y, max_y = sorted((self.min_y, self.max_y))
            if max_x - min_x <= 0 or max_y - min_y <= 0:
                raise ValueError("a rectangular zone needs a non-zero width and height")
            self.min_x, self.max_x = min_x, max_x
            self.min_y, self.max_y = min_y, max_y
        elif self.shape is ZoneShape.CIRCLE:
            if self.radius_mm <= 0:
                raise ValueError("a circular zone needs a radius above zero")
        elif len(self.points) < 3:
            raise ValueError("a polygon zone needs at least three points")
        return self

    def allows(self, aspect: ZoneAspect) -> bool:
        return bool(getattr(self, _ASPECT_FIELDS[ZoneAspect(aspect)]))

    def describe_geometry(self) -> str:
        if self.shape is ZoneShape.RECTANGLE:
            return f"rectangle X {self.min_x:g}…{self.max_x:g}, Y {self.min_y:g}…{self.max_y:g} mm"
        if self.shape is ZoneShape.CIRCLE:
            return (
                f"circle at ({self.center_x:g}, {self.center_y:g}) mm, radius {self.radius_mm:g} mm"
            )
        return f"polygon with {len(self.points)} points"

    def contains_point(self, x: float, y: float) -> bool:
        if self.shape is ZoneShape.RECTANGLE:
            return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
        if self.shape is ZoneShape.CIRCLE:
            return math.hypot(x - self.center_x, y - self.center_y) <= self.radius_mm
        return _point_in_polygon(self.points, x, y)

    def distance_to_edge(self, x: float, y: float) -> float:
        """Shortest distance from a point to this zone's border, always >= 0."""
        if self.shape is ZoneShape.CIRCLE:
            return abs(math.hypot(x - self.center_x, y - self.center_y) - self.radius_mm)
        if self.shape is ZoneShape.RECTANGLE:
            corners = [
                (self.min_x, self.min_y),
                (self.max_x, self.min_y),
                (self.max_x, self.max_y),
                (self.min_x, self.max_y),
            ]
            edges = list(zip(corners, corners[1:] + corners[:1], strict=True))
        else:
            edges = list(zip(self.points, self.points[1:] + self.points[:1], strict=True))
        return min(_distance_to_segment(x, y, start, end) for start, end in edges)

    def contains_disc(self, x: float, y: float, radius_mm: float) -> bool:
        """True when a disc of ``radius_mm`` around (x, y) fits entirely inside."""
        if radius_mm <= 0:
            return self.contains_point(x, y)
        return self.contains_point(x, y) and self.distance_to_edge(x, y) >= radius_mm

    def overlaps_disc(self, x: float, y: float, radius_mm: float) -> bool:
        """True when any part of a disc around (x, y) falls inside this zone."""
        if radius_mm <= 0:
            return self.contains_point(x, y)
        return self.contains_point(x, y) or self.distance_to_edge(x, y) <= radius_mm


class ZoneSet(BaseModel):
    zones: list[Zone] = Field(default_factory=list)


class ZoneVerdict(BaseModel):
    """Why a placement was allowed or refused, for logs and review messages."""

    allowed: bool
    reason: str
    zone_name: str | None = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.allowed


def _point_in_polygon(points: list[tuple[float, float]], x: float, y: float) -> bool:
    # Ray casting, with points on an edge counted as inside so a zone's border
    # never falls between two zones.
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        if _distance_to_segment(x, y, start, end) <= 1e-9:
            return True
    inside = False
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x < crossing_x:
                inside = not inside
    return inside


def _distance_to_segment(
    x: float, y: float, start: tuple[float, float], end: tuple[float, float]
) -> float:
    (x1, y1), (x2, y2) = start, end
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(x - x1, y - y1)
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_squared))
    return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))


def evaluate(
    zones: list[Zone],
    aspect: ZoneAspect,
    x: float,
    y: float,
    radius_mm: float = 0.0,
) -> ZoneVerdict:
    """Decide whether a placement is permitted by the configured zones.

    ``radius_mm`` is the protection disc for the radius aspect; pass 0 (the
    default) for point placements such as weeds and plant centres.
    """
    aspect = ZoneAspect(aspect)
    label = ASPECT_LABELS[aspect]
    active = [zone for zone in zones if zone.enabled]
    exclusions = [zone for zone in active if zone.kind is ZoneKind.EXCLUSION]
    boundaries = [zone for zone in active if zone.kind is ZoneKind.BOUNDARY]

    for zone in exclusions:
        if zone.allows(aspect) and zone.contains_disc(x, y, radius_mm):
            return ZoneVerdict(
                allowed=True,
                reason=f'exclusion zone "{zone.name}" explicitly permits {label} here',
                zone_name=zone.name,
            )
    for zone in exclusions:
        if not zone.allows(aspect) and zone.overlaps_disc(x, y, radius_mm):
            return ZoneVerdict(
                allowed=False,
                reason=f'exclusion zone "{zone.name}" does not permit {label}',
                zone_name=zone.name,
            )
    # A boundary marks the growing area, not a hazard: only the centre point
    # needs to stay inside one, so a protection radius may extend past its
    # edge. Exclusion zones mark real hazards, so they keep the full disc
    # test above.
    boundary_radius_mm = 0.0 if aspect is ZoneAspect.RADIUS else radius_mm
    for zone in boundaries:
        if not zone.allows(aspect) and zone.overlaps_disc(x, y, boundary_radius_mm):
            return ZoneVerdict(
                allowed=False,
                reason=f'boundary "{zone.name}" does not permit {label}',
                zone_name=zone.name,
            )
    allowing = [zone for zone in boundaries if zone.allows(aspect)]
    if allowing:
        for zone in allowing:
            if zone.contains_disc(x, y, boundary_radius_mm):
                return ZoneVerdict(
                    allowed=True,
                    reason=f'inside boundary "{zone.name}"',
                    zone_name=zone.name,
                )
        fit = "does not fit inside" if boundary_radius_mm > 0 else "is outside"
        return ZoneVerdict(
            allowed=False,
            reason=f"the position {fit} every boundary that permits {label}",
        )
    return ZoneVerdict(allowed=True, reason=f"no zone restricts {label} here")


def weeds_allowed(zones: list[Zone], x: float, y: float) -> ZoneVerdict:
    return evaluate(zones, ZoneAspect.WEEDS, x, y)


def plant_center_allowed(zones: list[Zone], x: float, y: float) -> ZoneVerdict:
    return evaluate(zones, ZoneAspect.CENTERS, x, y)


def plant_radius_allowed(zones: list[Zone], x: float, y: float, radius_mm: float) -> ZoneVerdict:
    return evaluate(zones, ZoneAspect.RADIUS, x, y, radius_mm)


class ZoneStore:
    """Atomic JSON store under /data so zones survive container restarts."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> ZoneSet:
        if not self.path.exists():
            return ZoneSet()
        try:
            return ZoneSet.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ZoneSet()

    def zones(self) -> list[Zone]:
        return self.load().zones

    def save(self, values: ZoneSet) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            handle.write(values.model_dump_json(indent=2))
            temp = Path(handle.name)
        os.replace(temp, self.path)

    def add(self, zone: Zone) -> Zone:
        current = self.load()
        current.zones.append(zone)
        self.save(current)
        return zone

    def update(self, zone_id: str, **changes) -> Zone | None:
        current = self.load()
        for index, zone in enumerate(current.zones):
            if zone.zone_id == zone_id:
                updated = zone.model_copy(update=changes)
                # Re-validate so an edit can never persist invalid geometry.
                updated = Zone.model_validate(updated.model_dump())
                current.zones[index] = updated
                self.save(current)
                return updated
        return None

    def delete(self, zone_id: str) -> bool:
        current = self.load()
        remaining = [zone for zone in current.zones if zone.zone_id != zone_id]
        if len(remaining) == len(current.zones):
            return False
        self.save(ZoneSet(zones=remaining))
        return True
