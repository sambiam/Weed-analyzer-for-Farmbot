from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np

from . import ALGORITHM_VERSION, CONTRACT_VERSION
from .models import AnalysisResult, Calibration, Decision, Measurement, PlantSeed
from .resolution import MAX_PROCESSED_HEIGHT, MAX_PROCESSED_WIDTH

cv2.setNumThreads(1)

# Physical thresholds. Pixel thresholds are derived from these and the
# effective pixels-per-millimetre so behaviour stays comparable across
# 640x480, 960x720 and 1280x960 (see ScaleParams). The 640x480 frame at
# 1 px/mm reproduces the original hard-coded pixel values.
BASELINE_WIDTH = 640
BASELINE_HEIGHT = 480
MIN_COMPONENT_AREA_MM2 = 12.0  # noise floor at 1 px/mm -> 12 px
IRRIGATION_AREA_FACTOR = 20.0  # long thin components above this are rejected
MORPH_OPEN_MM = 3.0
MORPH_CLOSE_MM = 5.0


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class ScaleParams:
    """Resolution-aware pixel thresholds derived from calibration and size."""

    min_area: int
    max_area: int
    irrigation_area: int
    open_kernel: int
    close_kernel: int
    mean_ppm: float

    @classmethod
    def build(cls, width: int, height: int, calibration: Calibration | None) -> ScaleParams:
        if calibration is not None:
            # Physical thresholds converted through the effective scale.
            ppm_x = calibration.pixels_per_mm_x
            ppm_y = calibration.pixels_per_mm_y
            area_scale = ppm_x * ppm_y
            mean_ppm = (ppm_x + ppm_y) / 2
        else:
            # Uncalibrated: scale relative to the 640x480 baseline so noise
            # rejection still tracks resolution even without metric units.
            linear = ((width / BASELINE_WIDTH) + (height / BASELINE_HEIGHT)) / 2
            area_scale = linear * linear
            mean_ppm = linear
        min_area = max(8, round(MIN_COMPONENT_AREA_MM2 * area_scale))
        return cls(
            min_area=min_area,
            max_area=round(200_000 * area_scale),
            irrigation_area=round(min_area * IRRIGATION_AREA_FACTOR),
            open_kernel=_odd(max(3, round(MORPH_OPEN_MM * mean_ppm))),
            close_kernel=_odd(max(5, round(MORPH_CLOSE_MM * mean_ppm))),
            mean_ppm=mean_ppm,
        )


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


class ImageAnalysisEngine(ABC):
    @abstractmethod
    def analyse(
        self,
        image_bytes: bytes,
        image_id: int,
        image_timestamp: datetime,
        seeds: list[PlantSeed],
        calibration: Calibration,
        previous_masks: dict[int, np.ndarray] | None = None,
    ) -> AnalysisResult: ...


def decode_jpeg(
    data: bytes,
    max_bytes: int = 5 * 1024 * 1024,
    max_width: int = MAX_PROCESSED_WIDTH,
    max_height: int = MAX_PROCESSED_HEIGHT,
) -> np.ndarray:
    """Decode a JPEG that the integration already resized to the processed size.

    The image is only downscaled here as a defensive ceiling; it is never
    upscaled. In normal operation the returned array is exactly the processed
    resolution the integration produced.
    """
    if not data or len(data) > max_bytes or not data.startswith(b"\xff\xd8"):
        raise InvalidImageError("invalid or oversized JPEG")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("JPEG could not be decoded")
    if image.shape[1] > max_width or image.shape[0] > max_height:
        scale = min(max_width / image.shape[1], max_height / image.shape[0])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def resize_prior_mask(prior: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Return ``prior`` fitted to ``shape``, or None when it cannot be trusted.

    A historical mask from another resolution is only reused when its aspect
    ratio matches the current frame -- then it is a safe dimensional rescale of
    the same field of view. A mismatched aspect ratio is rejected rather than
    stretched.
    """
    if prior.shape[:2] == shape:
        return prior
    ph, pw = prior.shape[:2]
    h, w = shape
    if pw == 0 or ph == 0:
        return None
    if abs((pw / ph) - (w / h)) > 0.02 * (w / h):
        return None
    return cv2.resize(prior, (w, h), interpolation=cv2.INTER_NEAREST)


def register_translation(previous: np.ndarray, current: np.ndarray) -> tuple[float, float, float]:
    if previous.shape != current.shape:
        return 0.0, 0.0, 0.0
    shift, response = cv2.phaseCorrelate(previous.astype(np.float32), current.astype(np.float32))
    if response < 0.05 or math.hypot(*shift) > 40:
        return 0.0, 0.0, float(response)
    return float(shift[0]), float(shift[1]), float(response)


def vegetation_mask(image: np.ndarray, params: ScaleParams) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image.astype(np.int16))
    excess_green = 2 * g - r - b
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hsv_green = cv2.inRange(hsv, (25, 35, 25), (100, 255, 255)) > 0
    exg_threshold = max(18, int(np.percentile(excess_green, 70)))
    mask = hsv_green & (excess_green > exg_threshold) & (saturation > 35) & (value > 25)
    binary = mask.astype(np.uint8) * 255
    open_k = np.ones((params.open_kernel, params.open_kernel), np.uint8)
    close_k = np.ones((params.close_kernel, params.close_kernel), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)


def _valid_component(stats: np.ndarray, label: int, params: ScaleParams) -> bool:
    _, _, width, height, area = stats[label]
    aspect = max(width, height) / max(1, min(width, height))
    return params.min_area <= area <= params.max_area and not (
        aspect > 9 and area > params.irrigation_area
    )


class ClassicalVisionEngine(ImageAnalysisEngine):
    def __init__(self, safety_margin_mm: float = 20, calibration_uncertainty_mm: float = 10):
        self.safety_margin_mm = safety_margin_mm
        self.calibration_uncertainty_mm = calibration_uncertainty_mm

    def diagnostic_only(self, image_bytes: bytes) -> AnalysisResult:
        """Pixel-space segmentation with no metric measurement (Part 6).

        Used when no valid calibration exists: a vegetation overlay is still
        produced for the operator, but no radius is measured and nothing can be
        written.
        """
        image = decode_jpeg(image_bytes)
        params = ScaleParams.build(image.shape[1], image.shape[0], None)
        mask = vegetation_mask(image, params)
        overlay = image.copy()
        overlay[mask > 0] = (40, 220, 40)
        cv2.putText(
            overlay,
            "Calibration required for millimetre measurements",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        ok_mask, encoded_mask = cv2.imencode(".png", mask)
        ok_overlay, encoded_overlay = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        del image, mask, overlay
        return AnalysisResult(
            measurements=[],
            mask=encoded_mask.tobytes() if ok_mask else None,
            overlay_jpeg=encoded_overlay.tobytes() if ok_overlay else None,
            skipped={},
        )

    def analyse(
        self,
        image_bytes: bytes,
        image_id: int,
        image_timestamp: datetime,
        seeds: list[PlantSeed],
        calibration: Calibration,
        previous_masks: dict[int, np.ndarray] | None = None,
    ) -> AnalysisResult:
        image = decode_jpeg(image_bytes)
        params = ScaleParams.build(image.shape[1], image.shape[0], calibration)
        mask = vegetation_mask(image, params)
        # Normalize any historical masks to this resolution (Part 8); reject
        # those from an incompatible aspect ratio.
        normalized: dict[int, np.ndarray] = {}
        for plant_id, prior in (previous_masks or {}).items():
            fitted = resize_prior_mask(prior, mask.shape)
            if fitted is not None:
                normalized[plant_id] = fitted
        previous_masks = normalized
        if previous_masks:
            combined_prior = np.zeros_like(mask)
            for prior in previous_masks.values():
                combined_prior = cv2.bitwise_or(combined_prior, (prior > 0).astype(np.uint8) * 255)
            dx, dy, response = register_translation(combined_prior, mask)
            if response >= 0.05 and (abs(dx) >= 0.5 or abs(dy) >= 0.5):
                transform = np.float32([[1, 0, dx], [0, 1, dy]])
                previous_masks = {
                    plant_id: cv2.warpAffine(prior, transform, (mask.shape[1], mask.shape[0]))
                    for plant_id, prior in previous_masks.items()
                }
        height, width = mask.shape
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        centers = np.array([seed.center_px for seed in seeds], dtype=np.float32)
        overlay = image.copy()
        ownership = np.zeros_like(labels, dtype=np.int16)
        ambiguous = np.zeros_like(mask, dtype=bool)
        uncertain_seeds: set[int] = set()
        skipped: dict[int, str] = {}
        ambiguity_gap = max(8.0, params.mean_ppm * 8)

        valid_indices: list[int] = []
        for index, seed in enumerate(seeds):
            x, y = seed.center_px
            border = max(3, min(seed.current_radius_mm * calibration.pixels_per_mm_x * 0.1, 15))
            if x < border or y < border or x >= width - border or y >= height - border:
                skipped[seed.plant_id] = "plant centre outside image or too close to border"
            else:
                valid_indices.append(index)

        for label in range(1, labels_count):
            if not _valid_component(stats, label, params):
                continue
            component = labels == label
            ys, xs = np.where(component)
            if not len(xs) or not valid_indices:
                continue
            distances = np.stack(
                [(xs - centers[i, 0]) ** 2 + (ys - centers[i, 1]) ** 2 for i in valid_indices]
            )
            nearest_order = np.argsort(distances, axis=0)
            nearest = np.array(valid_indices)[nearest_order[0]]
            if len(valid_indices) > 1:
                first = np.sqrt(np.take_along_axis(distances, nearest_order[:1], axis=0)[0])
                second = np.sqrt(np.take_along_axis(distances, nearest_order[1:2], axis=0)[0])
                is_ambiguous = (second - first) < ambiguity_gap
            else:
                is_ambiguous = np.zeros(len(xs), dtype=bool)
            for index in set(nearest.tolist()):
                candidate = nearest == index
                seed = seeds[index]
                cx, cy = seed.center_px
                seed_radius_px = max(
                    8,
                    seed.current_radius_mm
                    * (calibration.pixels_per_mm_x + calibration.pixels_per_mm_y)
                    / 2,
                )
                component_near_seed = np.any((xs - cx) ** 2 + (ys - cy) ** 2 <= seed_radius_px**2)
                prior = previous_masks.get(seed.plant_id)
                historical_overlap = (
                    prior is not None and prior.shape == mask.shape and np.any(prior[ys, xs] > 0)
                )
                if component_near_seed or historical_overlap:
                    ownership[ys[candidate], xs[candidate]] = index + 1
                    ambiguous[ys[candidate & is_ambiguous], xs[candidate & is_ambiguous]] = True
                elif (
                    np.min(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2))
                    < seed_radius_px
                    + 50 * (calibration.pixels_per_mm_x + calibration.pixels_per_mm_y) / 2
                ):
                    uncertain_seeds.add(index)

        measurements: list[Measurement] = []
        for index in valid_indices:
            seed = seeds[index]
            owned = ownership == index + 1
            ys, xs = np.where(owned)
            if len(xs) < params.min_area:
                skipped[seed.plant_id] = "no vegetation connected to known plant centre"
                continue
            dx_mm = (xs - seed.center_px[0]) / calibration.pixels_per_mm_x
            dy_mm = (ys - seed.center_px[1]) / calibration.pixels_per_mm_y
            distances_mm = np.sqrt(dx_mm**2 + dy_mm**2)
            typical = float(np.percentile(distances_mm, 90))
            maximum = float(distances_mm.max())
            plant_ambiguous = bool(np.any(ambiguous & owned)) or index in uncertain_seeds
            component_coverage = min(1.0, len(xs) / (500.0 * params.mean_ppm**2))
            border_distance = min(
                seed.center_px[0],
                seed.center_px[1],
                width - seed.center_px[0],
                height - seed.center_px[1],
            )
            edge_score = min(
                1.0, border_distance / max(1, maximum * calibration.pixels_per_mm_x + 8)
            )
            confidence = max(
                0.05,
                min(
                    0.99,
                    0.55
                    + 0.25 * component_coverage
                    + 0.2 * edge_score
                    - (0.4 if plant_ambiguous else 0),
                ),
            )
            recommendation = (
                maximum
                + self.safety_margin_mm
                + max(self.calibration_uncertainty_mm, calibration.uncertainty_mm)
            )
            decision = Decision.UNCERTAIN if plant_ambiguous else Decision.OBSERVED
            reason = (
                "ownership is ambiguous or a new disconnected region needs history"
                if plant_ambiguous
                else "maximum accepted leaf extent plus safety and calibration margins"
            )
            age = None
            if seed.planted_at:
                age = max(0, (image_timestamp.date() - seed.planted_at.date()).days)
            measurements.append(
                Measurement(
                    measurement_id=uuid4(),
                    plant_id=seed.plant_id,
                    crop_slug=seed.crop_slug,
                    image_id=image_id,
                    image_timestamp=image_timestamp,
                    current_radius_mm=seed.current_radius_mm,
                    typical_canopy_radius_mm=typical,
                    maximum_accepted_canopy_radius_mm=maximum,
                    recommended_protection_radius_mm=recommendation,
                    confidence=confidence,
                    decision=decision,
                    reason=reason,
                    ambiguous=plant_ambiguous,
                    calibration_version_id=calibration.version_id,
                    transform_json=json.dumps(
                        {
                            "pixels_per_mm_x": calibration.pixels_per_mm_x,
                            "pixels_per_mm_y": calibration.pixels_per_mm_y,
                            "rotation_degrees": calibration.rotation_degrees,
                            "offset_x_mm": calibration.offset_x_mm,
                            "offset_y_mm": calibration.offset_y_mm,
                            "calibration_source": calibration.source,
                            "calibration_version": calibration.calibration_version,
                            "analysis_resolution": calibration.analysis_resolution,
                            "processed_width": calibration.processed_width,
                            "processed_height": calibration.processed_height,
                            "source_width": calibration.source_width,
                            "source_height": calibration.source_height,
                            "oriented_width": calibration.oriented_width,
                            "oriented_height": calibration.oriented_height,
                            "resize_scale_x": calibration.resize_scale_x,
                            "resize_scale_y": calibration.resize_scale_y,
                            "contract_version": CONTRACT_VERSION,
                            "algorithm_version": ALGORITHM_VERSION,
                        },
                        separators=(",", ":"),
                    ),
                    algorithm_version=ALGORITHM_VERSION,
                    plant_age_days=age,
                    analysis_resolution=calibration.analysis_resolution,
                    processed_width=width,
                    processed_height=height,
                    calibration_source=calibration.source,
                    calibrated=True,
                    contract_version=CONTRACT_VERSION,
                )
            )
            color = (0, 165, 255) if plant_ambiguous else (40, 220, 40)
            cv2.circle(
                overlay,
                (round(seed.center_px[0]), round(seed.center_px[1])),
                round(maximum * calibration.pixels_per_mm_x),
                color,
                2,
            )
            cv2.putText(
                overlay,
                f"{seed.plant_id}: {recommendation:.0f}mm {confidence:.2f}",
                (round(seed.center_px[0]) + 4, round(seed.center_px[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        overlay[ambiguous] = (0, 0, 255)
        ok_mask, encoded_mask = cv2.imencode(".png", ownership.astype(np.uint16))
        ok_overlay, encoded_overlay = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        # Release the large working arrays before returning (Part 7).
        del image, mask, overlay, labels, ownership, ambiguous
        return AnalysisResult(
            measurements=measurements,
            mask=encoded_mask.tobytes() if ok_mask else None,
            overlay_jpeg=encoded_overlay.tobytes() if ok_overlay else None,
            skipped=skipped,
        )


def garden_to_pixel(
    plant_x: float,
    plant_y: float,
    image_x: float,
    image_y: float,
    width: int,
    height: int,
    calibration: Calibration,
) -> tuple[float, float]:
    dx = plant_x - image_x + calibration.offset_x_mm
    dy = plant_y - image_y + calibration.offset_y_mm
    theta = math.radians(calibration.rotation_degrees)
    rx = dx * math.cos(theta) - dy * math.sin(theta)
    ry = dx * math.sin(theta) + dy * math.cos(theta)
    return (
        width / 2 + rx * calibration.pixels_per_mm_x,
        height / 2 + ry * calibration.pixels_per_mm_y,
    )


def manual_scale(
    point_a: tuple[float, float], point_b: tuple[float, float], distance_mm: float
) -> float:
    if distance_mm <= 0:
        raise ValueError("real-world separation must be positive")
    pixels = math.dist(point_a, point_b)
    if pixels < 2:
        raise ValueError("calibration points are too close")
    return pixels / distance_mm
