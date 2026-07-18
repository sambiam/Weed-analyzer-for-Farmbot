"""Resolve the metric calibration for the exact processed image.

Calibration must always describe the pixels handed to OpenCV. A scale computed
for a 2592 x 1944 native frame is never applied directly to a resized 960 x 720
image; it is either supplied already-processed by the integration, or
transformed to the processed resolution here.

Preference order (see :func:`resolve_calibration`):

1. valid ``processed_calibration`` returned with the image
2. reference calibration transformed to the processed resolution
3. a compatible manual calibration
4. no metric calibration (pixel-only analysis; no writes)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Calibration, CameraCalibration, VisionImage
from .resolution import Resolution

# How closely two aspect ratios must agree before a reference scale is trusted.
_ASPECT_TOLERANCE = 0.02


@dataclass
class CalibrationResolution:
    """Outcome of resolving calibration for one processed image."""

    calibration: Calibration | None
    source: str
    reason: str
    warnings: list[str]


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _aspect_matches(a: float, b: float) -> bool:
    return abs(a - b) <= _ASPECT_TOLERANCE * max(a, b)


def _provenance(image: VisionImage, resolution: Resolution) -> dict[str, object]:
    return {
        "analysis_resolution": resolution.value,
        "image_id": image.image_id,
        "processed_width": image.width,
        "processed_height": image.height,
        "source_width": image.source_width,
        "source_height": image.source_height,
        "oriented_width": image.oriented_width,
        "oriented_height": image.oriented_height,
        "resize_scale_x": image.resize_scale_x,
        "resize_scale_y": image.resize_scale_y,
    }


def from_processed_calibration(
    image: VisionImage, resolution: Resolution, uncertainty_mm: float
) -> Calibration | None:
    """Use integration-supplied calibration that already targets these pixels."""
    processed = image.processed_calibration
    if processed is None or not processed.available:
        return None
    if processed.basis != "processed_image":
        raise ValueError("processed_calibration basis is not processed_image")
    if processed.width != image.width or processed.height != image.height:
        raise ValueError("processed_calibration dimensions do not match the returned image")
    if not (
        _finite_positive(processed.pixels_per_mm_x) and _finite_positive(processed.pixels_per_mm_y)
    ):
        raise ValueError("processed_calibration pixel scales are not positive and finite")
    return Calibration(
        source="processed_image",
        pixels_per_mm_x=processed.pixels_per_mm_x,
        pixels_per_mm_y=processed.pixels_per_mm_y,
        rotation_degrees=processed.rotation_degrees,
        offset_x_mm=processed.offset_x_mm,
        offset_y_mm=processed.offset_y_mm,
        uncertainty_mm=uncertainty_mm,
        calibration_version="processed_image",
        basis="processed_image",
        **_provenance(image, resolution),
    )


def scale_reference_to_processed(
    reference: CameraCalibration,
    image: VisionImage,
    resolution: Resolution,
    uncertainty_mm: float,
) -> Calibration | None:
    """Transform a normalized reference calibration to the processed resolution.

    ``processed_pixels_per_mm_x = reference_pixels_per_mm_x * width / reference_width``
    (and likewise for y), using oriented dimensions so EXIF orientation is
    accounted for. Refuses when reference dimensions are missing, the image
    lacks full v2 metadata, or the aspect ratios do not agree -- rather than
    guess.
    """
    if not reference.available or not reference.has_reference_dimensions:
        return None
    if not image.full_metadata:
        return None
    ref_w, ref_h = reference.reference_width, reference.reference_height
    if not (
        _finite_positive(reference.pixels_per_mm_x) and _finite_positive(reference.pixels_per_mm_y)
    ):
        return None
    # The reference frame must share the processed frame's orientation so the
    # x/y scales map to the same physical axes.
    oriented_landscape = image.oriented_width >= image.oriented_height
    if oriented_landscape != (ref_w >= ref_h):
        return None
    if not _aspect_matches(image.oriented_width / image.oriented_height, ref_w / ref_h):
        return None
    ppm_x = reference.pixels_per_mm_x * image.width / ref_w
    ppm_y = reference.pixels_per_mm_y * image.height / ref_h
    if not (_finite_positive(ppm_x) and _finite_positive(ppm_y)):
        return None
    return Calibration(
        source="reference_scaled",
        pixels_per_mm_x=ppm_x,
        pixels_per_mm_y=ppm_y,
        rotation_degrees=reference.rotation_degrees or 0.0,
        offset_x_mm=reference.offset_x_mm or 0.0,
        offset_y_mm=reference.offset_y_mm or 0.0,
        uncertainty_mm=uncertainty_mm,
        calibration_version=f"reference@{ref_w}x{ref_h}",
        basis="reference_scaled",
        **_provenance(image, resolution),
    )


def transform_manual_calibration(
    existing: Calibration, target_width: int, target_height: int
) -> Calibration | None:
    """Rescale a manual calibration to a new processed resolution.

    Only possible when the source processed dimensions are known so the scaling
    relationship is fully determined. Returns ``None`` otherwise -- the caller
    must then require confirmation or recalibration (Part 5); pixel coordinates
    are never silently reused at an incompatible resolution.
    """
    if not existing.processed_width or not existing.processed_height:
        return None
    if existing.processed_width == target_width and existing.processed_height == target_height:
        return existing
    scale_x = target_width / existing.processed_width
    scale_y = target_height / existing.processed_height
    if not (_finite_positive(scale_x) and _finite_positive(scale_y)):
        return None

    def _scaled_point(x: float | None, y: float | None) -> tuple[float | None, float | None]:
        return (
            None if x is None else x * scale_x,
            None if y is None else y * scale_y,
        )

    ax, ay = _scaled_point(existing.point_a_x, existing.point_a_y)
    bx, by = _scaled_point(existing.point_b_x, existing.point_b_y)
    return existing.model_copy(
        update={
            "version_id": None,
            "source": "manual_transformed",
            "pixels_per_mm_x": existing.pixels_per_mm_x * scale_x,
            "pixels_per_mm_y": existing.pixels_per_mm_y * scale_y,
            "processed_width": target_width,
            "processed_height": target_height,
            "point_a_x": ax,
            "point_a_y": ay,
            "point_b_x": bx,
            "point_b_y": by,
            "transformed_from_id": existing.version_id,
        }
    )


def _use_manual(
    manual: Calibration,
    image: VisionImage,
    resolution: Resolution,
    warnings: list[str],
) -> Calibration | None:
    """Return a manual calibration usable for this processed image, or None."""
    if manual.processed_width and manual.processed_height:
        if manual.processed_width == image.width and manual.processed_height == image.height:
            return manual.model_copy(update=_provenance(image, resolution))
        transformed = transform_manual_calibration(manual, image.width, image.height)
        if transformed is not None:
            warnings.append(
                "manual calibration was mathematically transformed from "
                f"{manual.processed_width}x{manual.processed_height} to "
                f"{image.width}x{image.height}"
            )
            return transformed.model_copy(update=_provenance(image, resolution))
        warnings.append("manual calibration belongs to another resolution and cannot be verified")
        return None
    # Legacy manual calibration with no recorded resolution: cannot verify the
    # pixel relationship, so it must not be reused for metric measurement.
    warnings.append("manual calibration has no recorded resolution; recalibration required")
    return None


def resolve_calibration(
    image: VisionImage,
    reference: CameraCalibration,
    manual: Calibration | None,
    resolution: Resolution,
    uncertainty_mm: float,
) -> CalibrationResolution:
    """Pick the calibration that corresponds to this exact processed image."""
    warnings: list[str] = []

    processed = from_processed_calibration(image, resolution, uncertainty_mm)
    if processed is not None:
        return CalibrationResolution(
            processed, "processed_image", "processed calibration", warnings
        )

    scaled = scale_reference_to_processed(reference, image, resolution, uncertainty_mm)
    if scaled is not None:
        return CalibrationResolution(
            scaled, "reference_scaled", "reference calibration scaled to resolution", warnings
        )

    if reference.available and not scaled:
        warnings.append(
            "reference calibration could not be scaled to the processed resolution safely"
        )

    if manual is not None:
        usable = _use_manual(manual, image, resolution, warnings)
        if usable is not None:
            return CalibrationResolution(usable, usable.source, "manual calibration", warnings)

    return CalibrationResolution(None, "none", "no valid calibration for this resolution", warnings)
