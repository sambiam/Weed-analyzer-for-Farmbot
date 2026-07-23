"""Pure, conservative edits for individual FarmBot spread curves.

FarmBot stores spread-curve values as *diameters*.  This module deliberately
does not know about measurements, databases, or Home Assistant: callers pass
the already-converted diameter and decide whether an ``ok`` edit is written or
a ``flagged`` edit is presented for review.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal

import numpy as np

MAX_CURVE_CONTROL_POINTS = 10
CurveVerdict = Literal["ok", "flagged"]


def radius_mm_to_diameter_mm(radius_mm: float) -> float:
    """Convert a FarmBot point radius into a spread-curve diameter."""
    return 2 * _diameter(radius_mm, field="radius_mm")


@dataclass(frozen=True, slots=True)
class Verdict:
    """Result of validating one measured diameter against a curve.

    ``conflict_*`` identifies the existing point that caused a monotonic or
    growth-rate warning.  It is absent for an absolute-limit warning because
    there is no single conflicting control point in that case.
    """

    verdict: CurveVerdict
    reason: str | None = None
    conflict_day: int | None = None
    conflict_old_diameter: float | None = None

    @property
    def status(self) -> CurveVerdict:
        """Alias used by service/UI callers that expose a ``status`` field."""

        return self.verdict

    @property
    def is_ok(self) -> bool:
        return self.verdict == "ok"


@dataclass(frozen=True, slots=True)
class ProposedEdit:
    """A proposed curve data payload plus its review metadata."""

    data: dict[str, float]
    max_day_changed: bool
    verdict: CurveVerdict = "ok"
    reason: str | None = None
    conflict_day: int | None = None
    conflict_old_diameter: float | None = None
    downsampled: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> CurveVerdict:
        """Alias for consumers that use status-oriented result objects."""

        return self.verdict

    def with_verdict(self, verdict: Verdict) -> ProposedEdit:
        """Attach a gate result while retaining structural-edit warnings."""

        return replace(
            self,
            verdict=verdict.verdict,
            reason=verdict.reason,
            conflict_day=verdict.conflict_day,
            conflict_old_diameter=verdict.conflict_old_diameter,
        )


def _day(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("curve days must be non-negative integers")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid curve day: {value!r}") from error
    if str(parsed) != str(value) or parsed < 0:
        raise ValueError(f"invalid curve day: {value!r}")
    return parsed


def _diameter(value: object, *, field: str = "diameter") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite non-negative number") from error
    if not isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return parsed


def _normalise_curve(existing: Mapping[str, float]) -> dict[int, float]:
    normalised: dict[int, float] = {}
    for raw_day, raw_diameter in existing.items():
        day = _day(raw_day)
        if day in normalised:
            raise ValueError(f"duplicate curve day after normalisation: {day}")
        normalised[day] = _diameter(raw_diameter, field=f"diameter at day {day}")
    return normalised


def _limit_control_points(points: dict[int, float], new_day: int) -> tuple[dict[int, float], bool]:
    """Select evenly spaced interior points while preserving required anchors."""

    ordered_days = sorted(points)
    if len(ordered_days) <= MAX_CURVE_CONTROL_POINTS:
        return points, False

    # The first/last point preserve the curve's original extent; the newly
    # measured point must survive even when it is an interior replacement.
    mandatory_indices = {0, len(ordered_days) - 1, ordered_days.index(new_day)}
    remaining = MAX_CURVE_CONTROL_POINTS - len(mandatory_indices)
    candidate_indices = [
        index for index in range(len(ordered_days)) if index not in mandatory_indices
    ]
    if remaining > 0:
        # Keep the same linspace-style selection convention as curves.py,
        # applied to the removable interior points.
        chosen_positions = np.linspace(0, len(candidate_indices) - 1, remaining).round().astype(int)
        mandatory_indices.update(candidate_indices[position] for position in chosen_positions)

    selected_days = [ordered_days[index] for index in sorted(mandatory_indices)]
    return {day: points[day] for day in selected_days}, True


def reasonableness_gate(
    existing: Mapping[str, float],
    day: int,
    new_diameter: float,
    *,
    max_daily_growth_mm: float,
    maximum_plant_radius_mm: float,
) -> Verdict:
    """Validate a candidate diameter without modifying the curve.

    The point at ``day`` is treated as a replacement, so it is excluded while
    finding adjacent controls.  A lower-day conflict is preferred because it
    is also the point used for the daily-growth calculation.
    """

    target_day = _day(day)
    target_diameter = _diameter(new_diameter, field="new_diameter")
    daily_limit = _diameter(max_daily_growth_mm, field="max_daily_growth_mm")
    radius_limit = _diameter(maximum_plant_radius_mm, field="maximum_plant_radius_mm")
    if daily_limit <= 0 or radius_limit <= 0:
        raise ValueError("growth and maximum-radius limits must be greater than zero")

    points = _normalise_curve(existing)
    lower_days = [point_day for point_day in points if point_day < target_day]
    higher_days = [point_day for point_day in points if point_day > target_day]
    previous_day = max(lower_days) if lower_days else None
    next_day = min(higher_days) if higher_days else None

    if previous_day is not None and target_diameter < points[previous_day]:
        return Verdict(
            "flagged",
            "measured diameter would make the curve decrease after the previous control point",
            previous_day,
            points[previous_day],
        )
    if next_day is not None and target_diameter > points[next_day]:
        return Verdict(
            "flagged",
            "measured diameter would make the curve decrease before the next control point",
            next_day,
            points[next_day],
        )
    if previous_day is not None:
        daily_growth = (target_diameter - points[previous_day]) / max(1, target_day - previous_day)
        if daily_growth > 2 * daily_limit:
            return Verdict(
                "flagged",
                "measured diameter exceeds the maximum daily growth rate",
                previous_day,
                points[previous_day],
            )
    if target_diameter > 2 * radius_limit:
        return Verdict(
            "flagged",
            "measured diameter exceeds the maximum plant radius limit",
        )
    return Verdict("ok")


def propose_curve_point(
    existing: Mapping[str, float],
    day: int,
    new_diameter: float,
    *,
    max_daily_growth_mm: float | None = None,
    maximum_plant_radius_mm: float | None = None,
) -> ProposedEdit:
    """Insert or replace one diameter point and retain at most ten controls.

    Passing both optional limits also attaches the reasonableness result to
    the edit.  Keeping the gate optional preserves a small structural-only
    API for callers that need to revalidate an edited UI value separately.
    """

    target_day = _day(day)
    target_diameter = _diameter(new_diameter, field="new_diameter")
    points = _normalise_curve(existing)
    old_max_day = max(points) if points else None
    points[target_day] = target_diameter
    limited, downsampled = _limit_control_points(points, target_day)
    warnings = ("curve was downsampled to ten control points",) if downsampled else ()
    edit = ProposedEdit(
        data={str(point_day): limited[point_day] for point_day in sorted(limited)},
        max_day_changed=old_max_day is None or target_day > old_max_day,
        downsampled=downsampled,
        warnings=warnings,
    )

    if (max_daily_growth_mm is None) != (maximum_plant_radius_mm is None):
        raise ValueError("provide both reasonableness limits or neither")
    if max_daily_growth_mm is not None and maximum_plant_radius_mm is not None:
        edit = edit.with_verdict(
            reasonableness_gate(
                existing,
                target_day,
                target_diameter,
                max_daily_growth_mm=max_daily_growth_mm,
                maximum_plant_radius_mm=maximum_plant_radius_mm,
            )
        )
    return edit
