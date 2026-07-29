"""Shared plant-evidence selection and calibrated image geometry.

The measurement pipeline deliberately keeps three sets separate:

* candidates: images examined for the plant;
* useful: images containing target-owned canopy pixels;
* used: the useful image(s) that determine the final estimate.

Boundary coverage is the fraction of 72 equal angular sectors whose expected
outer-canopy point is inside at least one selected calibrated image footprint.
This is an observable geometric quantity, unlike foreground area, and makes
the "approximately 50%" partial-view rule deterministic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

BOUNDARY_SECTOR_COUNT = 72
MINIMUM_PARTIAL_BOUNDARY_COVERAGE = 0.50
COMPLETE_BOUNDARY_COVERAGE = 0.98
MAXIMUM_EVIDENCE_AGE_HOURS = 6.0


def value(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def measurement_id(item: object) -> str:
    return str(value(item, "measurement_id", ""))


def image_id(item: object) -> int:
    return int(value(item, "image_id", 0))


def timestamp(item: object) -> datetime:
    raw = value(item, "image_timestamp")
    return raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))


def boundary_sectors(item: object) -> frozenset[int]:
    raw = value(item, "boundary_sectors")
    if raw is None:
        raw = value(item, "boundary_sectors_json", "[]")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raw = []
    sectors = (
        frozenset()
        if not isinstance(raw, list | tuple | set | frozenset)
        else frozenset(
            int(sector)
            for sector in raw
            if isinstance(sector, int | float) and 0 <= int(sector) < BOUNDARY_SECTOR_COUNT
        )
    )
    if sectors:
        return sectors
    coverage = float(value(item, "boundary_coverage", 0) or 0)
    if coverage <= 0:
        coverage = float(value(item, "visible_fraction", 0) or 0)
    return frozenset(range(round(max(0.0, min(1.0, coverage)) * BOUNDARY_SECTOR_COUNT)))


def parse_transform(item: object) -> dict[str, Any]:
    raw = value(item, "transform_json", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def relative_pixel_to_plant_mm_transform(item: object) -> tuple[np.ndarray, float] | None:
    """Map source pixels to plant-centred bed millimetres.

    This is the relative form of ``vision.pixel_to_garden``. The copied FarmBot
    camera rotation is negated exactly once, matching the main calibration
    transform; camera position and offsets cancel because the known plant pixel
    centre is the origin.
    """

    transform = parse_transform(item)
    ppm_x = float(transform.get("pixels_per_mm_x") or 0)
    ppm_y = float(transform.get("pixels_per_mm_y") or 0)
    center = value(item, "plant_center_px")
    if center is None:
        center_x = value(item, "plant_center_px_x")
        center_y = value(item, "plant_center_px_y")
        center = (
            (float(center_x), float(center_y))
            if center_x is not None and center_y is not None
            else None
        )
    if ppm_x <= 0 or ppm_y <= 0 or center is None:
        return None
    theta = math.radians(-float(transform.get("rotation_degrees") or 0))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    origin = str(transform.get("origin_location") or "top_left")
    sign_x = -1 if origin in {"top_right", "bottom_right"} else 1
    sign_y = -1 if origin in {"bottom_left", "bottom_right"} else 1
    cx, cy = float(center[0]), float(center[1])
    matrix = np.float64(
        [
            [
                sign_x * cos_t / ppm_x,
                -sign_x * sin_t / ppm_x,
                sign_x * (-cos_t * cx + sin_t * cy) / ppm_x,
            ],
            [
                sign_y * sin_t / ppm_y,
                sign_y * cos_t / ppm_y,
                sign_y * (-sin_t * cx - cos_t * cy) / ppm_y,
            ],
        ]
    )
    return matrix, math.sqrt(ppm_x * ppm_y)


def _has_target_evidence(item: object) -> bool:
    if bool(value(item, "vegetation_absent", False)):
        return False
    explicit = value(item, "has_plant_evidence")
    if explicit is not None:
        return bool(explicit)
    return (
        not bool(value(item, "vegetation_absent", False))
        and float(value(item, "maximum_accepted_canopy_radius_mm", 0) or 0) > 0
        and float(value(item, "visible_fraction", 0) or 0) > 0
    )


def _is_complete(item: object) -> bool:
    coverage = float(value(item, "boundary_coverage", 0) or 0)
    if coverage <= 0:
        coverage = float(value(item, "visible_fraction", 0) or 0)
    return (
        _has_target_evidence(item)
        and bool(value(item, "center_visible", True))
        and not bool(value(item, "canopy_truncated", False))
        and coverage >= COMPLETE_BOUNDARY_COVERAGE
    )


def _quality(item: object) -> float:
    confidence = max(0.0, min(1.0, float(value(item, "confidence", 0) or 0)))
    image_quality = max(0.05, min(1.0, float(value(item, "image_quality", 1) or 0)))
    segmentation = max(0.05, min(1.0, float(value(item, "segmentation_quality", 1) or 0)))
    return confidence * math.sqrt(image_quality * segmentation)


@dataclass(frozen=True)
class EvidenceSelection:
    candidates: tuple[object, ...]
    useful: tuple[object, ...]
    used: tuple[object, ...]
    excluded: tuple[tuple[object, str], ...]
    mode: str
    boundary_coverage: float

    @property
    def used_ids(self) -> tuple[int, ...]:
        return tuple(image_id(item) for item in self.used)

    @property
    def useful_ids(self) -> tuple[int, ...]:
        return tuple(image_id(item) for item in self.useful)

    def exclusion_reason(self, item: object) -> str | None:
        target_id = measurement_id(item)
        return next(
            (
                reason
                for candidate, reason in self.excluded
                if measurement_id(candidate) == target_id
            ),
            None,
        )


def select_measurement_evidence(measurements: list[object]) -> EvidenceSelection:
    """Select only measurements that can influence this plant's estimate."""

    if not measurements:
        return EvidenceSelection((), (), (), (), "no_evidence", 0.0)
    candidates = tuple(sorted(measurements, key=timestamp, reverse=True))
    newest = timestamp(candidates[0])
    useful: list[object] = []
    excluded: list[tuple[object, str]] = []
    for item in candidates:
        age_hours = max(0.0, (newest - timestamp(item)).total_seconds() / 3600)
        if age_hours > MAXIMUM_EVIDENCE_AGE_HOURS:
            excluded.append((item, "outside the current grid-run evidence window"))
        elif not _has_target_evidence(item):
            reason = str(value(item, "exclusion_reason", "") or "")
            excluded.append((item, reason or "no target-owned canopy evidence"))
        elif float(value(item, "segmentation_quality", 1) or 0) < 0.12:
            excluded.append((item, "segmentation quality is too low"))
        else:
            useful.append(item)

    complete = [item for item in useful if _is_complete(item)]
    if complete:
        chosen = max(complete, key=lambda item: (_quality(item), timestamp(item)))
        for item in useful:
            if measurement_id(item) != measurement_id(chosen):
                excluded.append((item, "a higher-quality complete single view was selected"))
        return EvidenceSelection(
            candidates,
            tuple(useful),
            (chosen,),
            tuple(excluded),
            "single_complete",
            1.0,
        )

    if not useful:
        return EvidenceSelection(candidates, (), (), tuple(excluded), "no_evidence", 0.0)
    if not any(bool(value(item, "center_visible", False)) for item in useful):
        excluded.extend(
            (item, "plant centre is not visible in any useful image") for item in useful
        )
        return EvidenceSelection(
            candidates,
            tuple(useful),
            (),
            tuple(excluded),
            "no_center",
            0.0,
        )

    # Start with the best centre-containing image, then greedily add the tile
    # contributing the most previously unseen outer-boundary sectors. This
    # produces a small evidence set while preserving every tile needed to reach
    # the measurable 50% rule.
    center_views = [item for item in useful if bool(value(item, "center_visible", False))]
    first = max(center_views, key=lambda item: (_quality(item), timestamp(item)))
    used = [first]
    covered = set(boundary_sectors(first))
    remaining = [item for item in useful if measurement_id(item) != measurement_id(first)]
    plant_fits = all(bool(value(item, "plant_fits_single_frame", True)) for item in useful)
    while remaining and (
        len(covered) / BOUNDARY_SECTOR_COUNT < MINIMUM_PARTIAL_BOUNDARY_COVERAGE or not plant_fits
    ):
        best = max(
            remaining,
            key=lambda item: (
                len(set(boundary_sectors(item)) - covered),
                _quality(item),
                timestamp(item),
            ),
        )
        new_sectors = set(boundary_sectors(best)) - covered
        if not new_sectors and plant_fits:
            break
        used.append(best)
        covered.update(boundary_sectors(best))
        remaining.remove(best)
    for item in useful:
        if all(measurement_id(item) != measurement_id(selected) for selected in used):
            excluded.append((item, "useful but added no boundary evidence needed by the estimate"))

    coverage = len(covered) / BOUNDARY_SECTOR_COUNT
    mode = (
        "large_composite"
        if not plant_fits
        else "partial_composite"
        if coverage >= MINIMUM_PARTIAL_BOUNDARY_COVERAGE
        else "insufficient_partial"
    )
    return EvidenceSelection(
        candidates,
        tuple(useful),
        tuple(used),
        tuple(excluded),
        mode,
        coverage,
    )


def selection_diagnostics(selection: EvidenceSelection) -> dict[str, object]:
    return {
        "candidate_image_ids": [image_id(item) for item in selection.candidates],
        "useful_image_ids": list(selection.useful_ids),
        "selected_image_ids": list(selection.used_ids),
        "excluded_images": [
            {"image_id": image_id(item), "reason": reason} for item, reason in selection.excluded
        ],
        "selection_mode": selection.mode,
        "visible_boundary_coverage": selection.boundary_coverage,
    }
