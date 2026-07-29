from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .canopy_settings import CanopyFusionSettings
from .plant_measurement import (
    MINIMUM_PARTIAL_BOUNDARY_COVERAGE,
    relative_pixel_to_plant_mm_transform,
    select_measurement_evidence,
)


@dataclass(frozen=True)
class FusedCanopyResult:
    typical_radius_mm: float
    maximum_radius_mm: float
    confidence: float
    view_count: int
    angular_coverage: float
    corroborated_fraction: float
    source_radius_spread_mm: float
    activated_by_partial_view: bool
    mask_png: bytes | None
    diagnostic_jpeg: bytes | None


def fuse_canopy_masks(
    measurements: list,
    settings: CanopyFusionSettings,
) -> FusedCanopyResult | None:
    """Fuse per-image plant ownership masks on a plant-centred metric canvas."""
    if not settings.enabled:
        return None
    selection = select_measurement_evidence(measurements)
    if selection.mode in {"single_complete", "no_evidence", "no_center"}:
        return None
    selected = list(selection.used)
    if not selected:
        return None
    newest = max(item.image_timestamp for item in selected)
    activated_by_partial = selection.mode in {
        "partial_composite",
        "insufficient_partial",
        "large_composite",
    }
    if not settings.always_fuse_when_available and not activated_by_partial:
        return None

    frames: list[dict] = []
    for item in selected:
        if (
            item.vegetation_absent
            or not item.mask_path
            or item.confidence < settings.minimum_view_confidence
            or (newest - item.image_timestamp).total_seconds()
            > settings.maximum_time_gap_hours * 3600
        ):
            continue
        mask = cv2.imread(item.mask_path, cv2.IMREAD_GRAYSCALE)
        relative = relative_pixel_to_plant_mm_transform(item)
        if mask is None or relative is None or not np.any(mask > 0):
            continue
        transform, scale = relative
        height, width = mask.shape
        corners = cv2.transform(
            np.float64([[[0, 0], [width, 0], [0, height], [width, height]]]), transform
        )[0]
        frames.append(
            {
                "item": item,
                "mask": mask,
                "transform": transform,
                "scale": scale,
                "bounds": (
                    float(corners[:, 0].min()),
                    float(corners[:, 0].max()),
                    float(corners[:, 1].min()),
                    float(corners[:, 1].max()),
                ),
            }
        )
    minimum_frames = 1 if activated_by_partial else settings.minimum_views
    if len(frames) < minimum_frames:
        return None

    min_x = min(frame["bounds"][0] for frame in frames)
    max_x = max(frame["bounds"][1] for frame in frames)
    min_y = min(frame["bounds"][2] for frame in frames)
    max_y = max(frame["bounds"][3] for frame in frames)
    ppm = float(np.median([frame["scale"] for frame in frames]))
    ppm = min(
        ppm,
        settings.maximum_canvas_pixels / max(1.0, max(max_x - min_x, max_y - min_y)),
    )
    ppm = max(0.25, ppm)
    canvas_width = max(1, round((max_x - min_x) * ppm))
    canvas_height = max(1, round((max_y - min_y) * ppm))
    size = (canvas_width, canvas_height)
    support_count = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
    visible_count = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
    strongest_evidence = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for frame in frames:
        affine = frame["transform"].copy()
        affine[0] *= ppm
        affine[1] *= ppm
        affine[0, 2] -= min_x * ppm
        affine[1, 2] -= min_y * ppm
        mask = frame["mask"] > 0
        height, width = mask.shape
        yy, xx = np.indices(mask.shape)
        border_px = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy)).astype(np.float32)
        edge_margin_px = settings.source_edge_margin_mm * frame["scale"]
        edge_quality = (
            np.ones_like(border_px)
            if edge_margin_px <= 0
            else np.clip(border_px / max(1.0, edge_margin_px), 0.2, 1.0)
        )
        evidence = mask.astype(np.float32) * edge_quality * float(frame["item"].confidence)
        warped_mask = cv2.warpAffine(
            mask.astype(np.uint8),
            affine,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_visible = cv2.warpAffine(
            np.ones(mask.shape, dtype=np.uint8),
            affine,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_evidence = cv2.warpAffine(
            evidence,
            affine,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        support_count += (warped_mask > 0).astype(np.uint16)
        visible_count += (warped_visible > 0).astype(np.uint16)
        strongest_evidence = np.maximum(strongest_evidence, warped_evidence)

    single_threshold = (
        settings.minimum_view_confidence
        if len(frames) == 1
        else settings.single_view_acceptance_confidence
    )
    accepted = (support_count >= settings.minimum_supporting_views) | (
        (support_count >= 1) & (strongest_evidence >= single_threshold)
    )
    if not np.any(accepted):
        return None
    # Remove interpolation specks without joining separate leaves, then drop
    # disconnected fragments smaller than the existing 12 mm² physical noise
    # floor. Legitimate separated leaves remain available to the radial sectors.
    accepted_u8 = accepted.astype(np.uint8) * 255
    accepted_u8 = cv2.morphologyEx(
        accepted_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        accepted_u8, 8
    )
    minimum_component_area = max(3, round(12.0 * ppm * ppm))
    accepted_u8 = np.zeros_like(accepted_u8)
    for label in range(1, component_count):
        if int(component_stats[label, cv2.CC_STAT_AREA]) >= minimum_component_area:
            accepted_u8[component_labels == label] = 255
    accepted = accepted_u8 > 0
    ys, xs = np.where(accepted)
    if not len(xs):
        return None

    center_x, center_y = -min_x * ppm, -min_y * ppm
    dx = (xs - center_x) / ppm
    dy = (ys - center_y) / ppm
    distances = np.hypot(dx, dy)
    angles = (np.arctan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
    sectors = np.floor(angles / (2 * math.pi) * settings.angular_sectors).astype(int)
    sector_outer = [
        float(np.percentile(distances[sectors == sector], 98))
        for sector in range(settings.angular_sectors)
        if np.count_nonzero(sectors == sector) >= 3
    ]
    if not sector_outer:
        return None
    typical = float(np.percentile(distances, 90))
    maximum = float(np.percentile(sector_outer, settings.radial_percentile))

    yy, xx = np.indices(accepted.shape)
    all_dx = (xx - center_x) / ppm
    all_dy = (yy - center_y) / ppm
    all_distances = np.hypot(all_dx, all_dy)
    all_angles = (np.arctan2(all_dy, all_dx) + 2 * math.pi) % (2 * math.pi)
    annulus = (all_distances >= max(1.0, maximum * 0.75)) & (
        all_distances <= max(2.0, maximum * 1.05)
    )
    observed_sectors = set(
        np.floor(
            all_angles[annulus & (visible_count > 0)] / (2 * math.pi) * settings.angular_sectors
        )
        .astype(int)
        .tolist()
    )
    coverage = max(
        len(observed_sectors) / settings.angular_sectors,
        selection.boundary_coverage,
    )
    corroborated = float(np.mean(support_count[accepted] >= 2))
    source_radii = sorted(
        float(frame["item"].maximum_accepted_canopy_radius_mm) for frame in frames
    )
    source_spread = (
        float(np.percentile(source_radii, 90) - np.percentile(source_radii, 10))
        if len(source_radii) > 1
        else 0.0
    )
    mean_confidence = float(np.mean([frame["item"].confidence for frame in frames]))
    confidence = float(
        np.clip(
            mean_confidence * 0.75
            + 0.20 * coverage
            + min(0.08, 0.02 * (len(frames) - 1))
            + min(0.05, corroborated * 0.1),
            0.05,
            0.99,
        )
    )
    if coverage < MINIMUM_PARTIAL_BOUNDARY_COVERAGE:
        confidence = min(confidence, 0.45)

    ok_mask, encoded_mask = cv2.imencode(".png", accepted_u8)
    diagnostic = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    diagnostic[visible_count > 0] = (35, 35, 35)
    diagnostic[accepted] = (40, 190, 40)
    diagnostic[accepted & (support_count >= 2)] = (0, 215, 255)
    center = (round(center_x), round(center_y))
    cv2.circle(diagnostic, center, max(1, round(typical * ppm)), (255, 255, 0), 2)
    cv2.circle(diagnostic, center, max(1, round(maximum * ppm)), (0, 0, 255), 2)
    cv2.drawMarker(diagnostic, center, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
    label = (
        f"{len(frames)} views | {maximum:.1f} mm | coverage {coverage:.0%} | "
        f"corroborated {corroborated:.0%}"
    )
    cv2.putText(
        diagnostic,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ok_diagnostic, encoded_diagnostic = cv2.imencode(
        ".jpg", diagnostic, [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    return FusedCanopyResult(
        typical_radius_mm=typical,
        maximum_radius_mm=maximum,
        confidence=confidence,
        view_count=len(frames),
        angular_coverage=coverage,
        corroborated_fraction=corroborated,
        source_radius_spread_mm=source_spread,
        activated_by_partial_view=activated_by_partial,
        mask_png=encoded_mask.tobytes() if ok_mask else None,
        diagnostic_jpeg=encoded_diagnostic.tobytes() if ok_diagnostic else None,
    )
