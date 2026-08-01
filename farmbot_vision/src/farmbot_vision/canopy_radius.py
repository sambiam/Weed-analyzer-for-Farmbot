"""Robust, low-cost canopy boundary measurement.

The colour/ownership pipeline deliberately favours recall.  That is useful for
finding pale leaves, but it also means that an attached weed or green soil can
occasionally become target-owned.  A radius must therefore not be the single
farthest owned pixel.

This module measures a radial envelope in small angular sectors.  Sparse pixel
outliers are removed with an in-sector percentile and the final boundary is a
percentile across sectors.  The stored FarmBot radius is also used as a temporal
prior: a broad, sudden expansion is clipped back to the previous canopy edge,
while a narrow supported protrusion (a genuine long leaf) is retained.  The
work is linear in the number of owned pixels and needs no model inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanopyRadiusEstimate:
    """A robust canopy estimate and the point-level evidence it accepted."""

    typical_radius_mm: float
    outer_radius_mm: float
    keep: np.ndarray
    observed_sector_fraction: float
    clipped_point_fraction: float
    broad_overreach: bool


def previous_canopy_edge_mm(
    current_radius_mm: float, protection_margin_mm: float
) -> tuple[float, bool]:
    """Return the prior foliage edge and whether configured margins were removable."""

    current = max(0.0, float(current_radius_mm))
    margin = max(0.0, float(protection_margin_mm))
    margin_aware = current > margin + 5.0
    return max(5.0, current - margin if margin_aware else current), margin_aware


def _circular_true_runs(flags: np.ndarray) -> list[np.ndarray]:
    """Return circular runs of true indexes without splitting the 0-degree run."""

    count = int(flags.size)
    if count == 0 or not np.any(flags):
        return []
    if np.all(flags):
        return [np.arange(count, dtype=np.int32)]
    starts = np.flatnonzero(flags & ~np.roll(flags, 1))
    runs: list[np.ndarray] = []
    for start in starts:
        indexes: list[int] = []
        index = int(start)
        while flags[index]:
            indexes.append(index)
            index = (index + 1) % count
        runs.append(np.asarray(indexes, dtype=np.int32))
    return runs


def estimate_canopy_radius(
    distances_mm: np.ndarray,
    angles_radians: np.ndarray,
    *,
    current_radius_mm: float,
    protection_margin_mm: float,
    angular_sectors: int = 72,
    radial_percentile: float = 97.0,
) -> CanopyRadiusEstimate | None:
    """Measure the supported outer canopy boundary from target-owned points.

    ``current_radius_mm`` is the previously stored protection radius, while
    ``protection_margin_mm`` is the safety plus calibration margin that was
    added to the earlier leaf edge.  Removing that margin gives the most useful
    temporal prior for detecting the normally small amount of new growth.

    Sudden expansion is rejected only when it spans a broad angle.  A long,
    narrow leaf can therefore extend well beyond the temporal band, but a moss
    patch or block of misclassified soil cannot move the whole boundary in one
    observation.
    """

    distances = np.asarray(distances_mm, dtype=np.float64).reshape(-1)
    angles = np.asarray(angles_radians, dtype=np.float64).reshape(-1)
    if distances.size == 0 or distances.size != angles.size:
        return None
    finite = np.isfinite(distances) & np.isfinite(angles) & (distances >= 0)
    if not np.any(finite):
        return None
    # Callers pass only foreground indexes, so preserving the input shape makes
    # ``keep`` directly usable to trim those indexes from an ownership mask.
    valid_indexes = np.flatnonzero(finite)
    valid_distances = distances[finite]
    valid_angles = np.mod(angles[finite], 2 * math.pi)
    sector_indexes = np.floor(valid_angles / (2 * math.pi) * angular_sectors).astype(np.int32)
    sector_indexes = np.clip(sector_indexes, 0, angular_sectors - 1)

    sector_outer = np.full(angular_sectors, np.nan, dtype=np.float64)
    for sector in np.unique(sector_indexes):
        values = valid_distances[sector_indexes == sector]
        if values.size >= 3:
            # One hot JPEG pixel or segmentation spur cannot define an edge.
            sector_outer[sector] = float(np.percentile(values, 98))
    observed = np.isfinite(sector_outer)
    if not np.any(observed):
        return None

    prior_edge, has_margin_aware_prior = previous_canopy_edge_mm(
        current_radius_mm, protection_margin_mm
    )
    # Twenty millimetres tolerates calibration jitter and ordinary growth.  A
    # larger established canopy receives a proportional, but bounded, band.
    growth_band = (
        max(20.0, min(40.0, prior_edge * 0.30))
        if has_margin_aware_prior
        # A radius smaller than its configured protection margins cannot be
        # converted into a trustworthy previous leaf edge.  Give bootstrap
        # measurements enough room to establish the first real canopy body.
        else max(40.0, min(75.0, prior_edge * 1.5))
    )
    plausible_limit = prior_edge + growth_band
    overextended = observed & (sector_outer > plausible_limit)

    # Ten percent of the circumference is a deliberately generous definition
    # of a single protruding leaf.  Wider sudden growth is much more likely to
    # be background vegetation or an attached neighbouring plant.
    narrow_run_limit = max(3, round(angular_sectors * 0.10))
    total_overreach_limit = max(narrow_run_limit * 2, round(angular_sectors * 0.22))
    broad_sectors = np.zeros(angular_sectors, dtype=bool)
    runs = _circular_true_runs(overextended)
    for run in runs:
        if run.size > narrow_run_limit:
            broad_sectors[run] = True
    if int(np.count_nonzero(overextended)) > total_overreach_limit:
        broad_sectors |= overextended
    broad_overreach = bool(np.any(broad_sectors))

    accepted_limits = sector_outer.copy()
    if broad_overreach:
        # Do not manufacture one full growth-band increase from a bad mask.
        # Returning to the previous leaf edge makes the final protection radius
        # stable until a clean observation supplies actual growth evidence.
        accepted_limits[broad_sectors] = prior_edge

    valid_keep = valid_distances <= (accepted_limits[sector_indexes] + 1e-6)
    keep = np.zeros(distances.shape, dtype=bool)
    keep[valid_indexes] = valid_keep
    accepted_distances = distances[keep]
    if accepted_distances.size == 0:
        return None

    accepted_sector_outer = np.full(angular_sectors, np.nan, dtype=np.float64)
    kept_sector_indexes = sector_indexes[valid_keep]
    kept_distances = valid_distances[valid_keep]
    for sector in np.unique(kept_sector_indexes):
        values = kept_distances[kept_sector_indexes == sector]
        if values.size >= 3:
            accepted_sector_outer[sector] = float(np.percentile(values, 98))
    accepted_outer_values = accepted_sector_outer[np.isfinite(accepted_sector_outer)]
    if accepted_outer_values.size == 0:
        return None

    outer = float(np.percentile(accepted_outer_values, radial_percentile))
    narrow_protrusions = overextended & ~broad_sectors & np.isfinite(accepted_sector_outer)
    if np.any(narrow_protrusions):
        # Sector-to-sector percentiles intentionally suppress scattered outer
        # noise, but a real narrow leaf can occupy only one 5-degree sector.
        # Its in-sector 98th percentile is already robust to isolated pixels,
        # so preserve that supported tip explicitly.
        outer = max(outer, float(np.max(accepted_sector_outer[narrow_protrusions])))
    typical = float(np.percentile(accepted_distances, 90))
    return CanopyRadiusEstimate(
        typical_radius_mm=min(typical, outer),
        outer_radius_mm=outer,
        keep=keep,
        observed_sector_fraction=float(np.mean(observed)),
        clipped_point_fraction=float(1.0 - np.mean(keep[finite])),
        broad_overreach=broad_overreach,
    )
