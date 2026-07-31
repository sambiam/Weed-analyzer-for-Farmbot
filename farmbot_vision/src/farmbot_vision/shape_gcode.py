"""Turn a shape into a raw FarmBot G-code program (experimental).

This is the planning half of the experimental Draw shape feature: it decides
*where* the gantry goes, in FarmBot bed coordinates, and writes that out as
G-code text. The companion integration owns everything after that -- it
re-validates the program against live axis bounds and firmware config, converts
the feed rate into the per-axis speeds the Farmduino wants, and sends it. That
split matches the rest of the app: the app plans coordinates, the integration
moves the bot.

Two FarmBot firmware facts drive the shape of the output:

- ``G01`` is not implemented. Only ``G00`` moves.
- ``G00`` is documented as "move to location at given speed for axis (don't
  have to be a straight line)". There is no coordinated interpolation, so a
  circle cannot be an arc and is not even reliably a polygon of long chords --
  it has to be many short segments. ``segments_for_tolerance`` picks how many
  from a sagitta (chord-deviation) budget, which is the honest way to state
  "how round is round enough" for a given radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Deliberately narrow. Everything here has to survive the integration's own
# allowlist, and a shape the user cannot describe in a sentence is not worth
# the extra failure modes on an experimental path.
SHAPES = {
    "circle": "Circle",
    "triangle": "Triangle (3 sides)",
    "square": "Square (4 sides)",
    "pentagon": "Pentagon (5 sides)",
    "hexagon": "Hexagon (6 sides)",
    "octagon": "Octagon (8 sides)",
    "polygon": "Regular polygon",
}
_POLYGON_SIDES = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}

MIN_CIRCUMRADIUS_MM = 5.0
MAX_CIRCUMRADIUS_MM = 2000.0
MIN_SIDES = 3
MAX_SIDES = 24
MIN_SEGMENTS = 8
MAX_SEGMENTS = 720
MIN_FEED_MM_PER_MIN = 1.0
MAX_FEED_MM_PER_MIN = 3000.0
# How far a chord may sag from the true circle, in millimetres. 0.5 mm is well
# under FarmBot's own positioning repeatability, so a tighter budget would just
# buy segments the hardware cannot honour.
DEFAULT_CHORD_TOLERANCE_MM = 0.5


class ShapeError(ValueError):
    """A shape could not be planned; the message is safe to show the user."""


@dataclass(frozen=True)
class ShapePlan:
    """A planned path plus the numbers worth showing before anything moves."""

    shape: str
    sides: int | None
    points: list[tuple[float, float]]
    center: tuple[float, float]
    circumradius_mm: float
    draw_z: float
    travel_z: float
    feed_mm_per_min: float
    lines: list[str]

    @property
    def perimeter_mm(self) -> float:
        total = 0.0
        for (x1, y1), (x2, y2) in zip(self.points, self.points[1:], strict=False):
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    def extent(self) -> tuple[float, float, float, float]:
        xs = [x for x, _ in self.points]
        ys = [y for _, y in self.points]
        return min(xs), min(ys), max(xs), max(ys)


def segments_for_tolerance(
    radius_mm: float, tolerance_mm: float = DEFAULT_CHORD_TOLERANCE_MM
) -> int:
    """Segment count whose chords stay within ``tolerance_mm`` of the circle.

    The sagitta of a chord subtending angle ``t`` is ``r * (1 - cos(t / 2))``.
    Solving for ``t`` at the tolerance and dividing into a full turn gives the
    count. Clamped to keep a huge circle from producing a program longer than
    the integration will accept.
    """
    if radius_mm <= 0:
        raise ShapeError("Radius must be positive")
    if tolerance_mm <= 0:
        raise ShapeError("Chord tolerance must be positive")
    if tolerance_mm >= radius_mm:
        return MIN_SEGMENTS
    angle = 2 * math.acos(1 - tolerance_mm / radius_mm)
    return max(MIN_SEGMENTS, min(MAX_SEGMENTS, math.ceil(2 * math.pi / angle)))


def resolve_sides(shape: str, sides: int | None) -> int | None:
    """Sides for a named shape; ``None`` for a circle."""
    if shape == "circle":
        return None
    if shape in _POLYGON_SIDES:
        return _POLYGON_SIDES[shape]
    if shape != "polygon":
        raise ShapeError(f"Unknown shape {shape!r}")
    if sides is None:
        raise ShapeError("A regular polygon needs a side count")
    if not MIN_SIDES <= int(sides) <= MAX_SIDES:
        raise ShapeError(f"Sides must be between {MIN_SIDES} and {MAX_SIDES}")
    return int(sides)


def shape_points(
    *,
    shape: str,
    center_x: float,
    center_y: float,
    circumradius_mm: float,
    sides: int | None = None,
    rotation_deg: float = 0.0,
    segments: int | None = None,
    chord_tolerance_mm: float = DEFAULT_CHORD_TOLERANCE_MM,
) -> list[tuple[float, float]]:
    """Vertices of the closed path, first point repeated at the end.

    Every shape is inscribed in the circumradius, so a circle and a hexagon
    asked for the same radius touch the same bounding circle -- which is what
    makes 'centre plus circumradius' one control for all of them.
    """
    if not all(math.isfinite(value) for value in (center_x, center_y, circumradius_mm)):
        raise ShapeError("Centre and radius must be finite numbers")
    if not MIN_CIRCUMRADIUS_MM <= circumradius_mm <= MAX_CIRCUMRADIUS_MM:
        raise ShapeError(
            f"Circumradius must be between {MIN_CIRCUMRADIUS_MM:g} and {MAX_CIRCUMRADIUS_MM:g} mm"
        )
    if not math.isfinite(rotation_deg):
        raise ShapeError("Rotation must be a finite number")

    resolved_sides = resolve_sides(shape, sides)
    if resolved_sides is None:
        count = (
            segments
            if segments is not None
            else segments_for_tolerance(circumradius_mm, chord_tolerance_mm)
        )
        if not MIN_SEGMENTS <= int(count) <= MAX_SEGMENTS:
            raise ShapeError(f"Segments must be between {MIN_SEGMENTS} and {MAX_SEGMENTS}")
        count = int(count)
    else:
        count = resolved_sides

    start = math.radians(rotation_deg)
    points = []
    for index in range(count):
        angle = start + 2 * math.pi * index / count
        points.append(
            (
                center_x + circumradius_mm * math.cos(angle),
                center_y + circumradius_mm * math.sin(angle),
            )
        )
    points.append(points[0])  # close the path
    return points


def generate_program(
    *,
    shape: str,
    center_x: float,
    center_y: float,
    circumradius_mm: float,
    sides: int | None = None,
    rotation_deg: float = 0.0,
    segments: int | None = None,
    chord_tolerance_mm: float = DEFAULT_CHORD_TOLERANCE_MM,
    draw_z: float = 0.0,
    travel_z: float = 0.0,
    feed_mm_per_min: float = 400.0,
    travel_feed_mm_per_min: float | None = None,
) -> ShapePlan:
    """Plan a shape and render it as a G-code program.

    The program always approaches at ``travel_z``, descends to ``draw_z`` only
    once it is over the start point, and retracts before it is done -- so a
    mistake in the centre coordinates drags the tool through the air rather
    than through the bed.
    """
    if not math.isfinite(draw_z) or not math.isfinite(travel_z):
        raise ShapeError("Draw and travel heights must be finite numbers")
    if not MIN_FEED_MM_PER_MIN <= feed_mm_per_min <= MAX_FEED_MM_PER_MIN:
        raise ShapeError(
            f"Feed rate must be between {MIN_FEED_MM_PER_MIN:g} and {MAX_FEED_MM_PER_MIN:g} mm/min"
        )
    travel_feed = travel_feed_mm_per_min or feed_mm_per_min
    if not MIN_FEED_MM_PER_MIN <= travel_feed <= MAX_FEED_MM_PER_MIN:
        raise ShapeError(
            f"Travel feed rate must be between {MIN_FEED_MM_PER_MIN:g} and "
            f"{MAX_FEED_MM_PER_MIN:g} mm/min"
        )

    points = shape_points(
        shape=shape,
        center_x=center_x,
        center_y=center_y,
        circumradius_mm=circumradius_mm,
        sides=sides,
        rotation_deg=rotation_deg,
        segments=segments,
        chord_tolerance_mm=chord_tolerance_mm,
    )
    resolved_sides = resolve_sides(shape, sides)
    first_x, first_y = points[0]

    label = SHAPES.get(shape, shape)
    if resolved_sides is not None:
        label = f"{label} ({resolved_sides} sides)" if shape == "polygon" else label
    lines = [
        "; FarmBot Vision experimental shape",
        f"; shape={shape} centre=({center_x:.1f}, {center_y:.1f}) "
        f"circumradius={circumradius_mm:.1f}mm",
        f"; segments={len(points) - 1} draw_z={draw_z:.1f} travel_z={travel_z:.1f}",
        "; G01 is not implemented by the FarmBot firmware and G00 is not",
        "; guaranteed to travel in a straight line, so curves are short G00 chords.",
        "G21",
        "G90",
        f"G00 Z{travel_z:.3f} F{travel_feed:.0f}",
        f"G00 X{first_x:.3f} Y{first_y:.3f}",
        f"G00 Z{draw_z:.3f} F{feed_mm_per_min:.0f}",
    ]
    lines += [f"G00 X{x:.3f} Y{y:.3f}" for x, y in points[1:]]
    lines.append(f"G00 Z{travel_z:.3f} F{travel_feed:.0f}")

    return ShapePlan(
        shape=shape,
        sides=resolved_sides,
        points=points,
        center=(center_x, center_y),
        circumradius_mm=circumradius_mm,
        draw_z=draw_z,
        travel_z=travel_z,
        feed_mm_per_min=feed_mm_per_min,
        lines=lines,
    )


def program_lines(text: str) -> list[str]:
    """Split an edited G-code textarea into lines the service will accept.

    Blank lines are dropped rather than sent: they are meaningless to the
    firmware and every one of them counts against the integration's line cap.
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    return [line for line in lines if line]
