from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np

from . import ALGORITHM_VERSION
from .models import AnalysisResult, Calibration, Decision, Measurement, PlantSeed

cv2.setNumThreads(1)


class InvalidImageError(ValueError):
    pass


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


def decode_jpeg(data: bytes, max_bytes: int = 5 * 1024 * 1024) -> np.ndarray:
    if not data or len(data) > max_bytes or not data.startswith(b"\xff\xd8"):
        raise InvalidImageError("invalid or oversized JPEG")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("JPEG could not be decoded")
    if image.shape[1] > 640 or image.shape[0] > 480:
        scale = min(640 / image.shape[1], 480 / image.shape[0])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def register_translation(previous: np.ndarray, current: np.ndarray) -> tuple[float, float, float]:
    if previous.shape != current.shape:
        return 0.0, 0.0, 0.0
    shift, response = cv2.phaseCorrelate(previous.astype(np.float32), current.astype(np.float32))
    if response < 0.05 or math.hypot(*shift) > 40:
        return 0.0, 0.0, float(response)
    return float(shift[0]), float(shift[1]), float(response)


def vegetation_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image.astype(np.int16))
    excess_green = 2 * g - r - b
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hsv_green = cv2.inRange(hsv, (25, 35, 25), (100, 255, 255)) > 0
    exg_threshold = max(18, int(np.percentile(excess_green, 70)))
    mask = hsv_green & (excess_green > exg_threshold) & (saturation > 35) & (value > 25)
    binary = mask.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _valid_component(stats: np.ndarray, label: int) -> bool:
    _, _, width, height, area = stats[label]
    aspect = max(width, height) / max(1, min(width, height))
    return 12 <= area <= 200_000 and not (aspect > 9 and area > 250)


class ClassicalVisionEngine(ImageAnalysisEngine):
    def __init__(self, safety_margin_mm: float = 20, calibration_uncertainty_mm: float = 10):
        self.safety_margin_mm = safety_margin_mm
        self.calibration_uncertainty_mm = calibration_uncertainty_mm

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
        mask = vegetation_mask(image)
        previous_masks = previous_masks or {}
        if previous_masks:
            combined_prior = np.zeros_like(mask)
            for prior in previous_masks.values():
                if prior.shape == mask.shape:
                    combined_prior = cv2.bitwise_or(
                        combined_prior, (prior > 0).astype(np.uint8) * 255
                    )
            dx, dy, response = register_translation(combined_prior, mask)
            if response >= 0.05 and (abs(dx) >= 0.5 or abs(dy) >= 0.5):
                transform = np.float32([[1, 0, dx], [0, 1, dy]])
                previous_masks = {
                    plant_id: cv2.warpAffine(prior, transform, (mask.shape[1], mask.shape[0]))
                    for plant_id, prior in previous_masks.items()
                    if prior.shape == mask.shape
                }
        height, width = mask.shape
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        centers = np.array([seed.center_px for seed in seeds], dtype=np.float32)
        overlay = image.copy()
        ownership = np.zeros_like(labels, dtype=np.int16)
        ambiguous = np.zeros_like(mask, dtype=bool)
        uncertain_seeds: set[int] = set()
        skipped: dict[int, str] = {}

        valid_indices: list[int] = []
        for index, seed in enumerate(seeds):
            x, y = seed.center_px
            border = max(3, min(seed.current_radius_mm * calibration.pixels_per_mm_x * 0.1, 15))
            if x < border or y < border or x >= width - border or y >= height - border:
                skipped[seed.plant_id] = "plant centre outside image or too close to border"
            else:
                valid_indices.append(index)

        for label in range(1, labels_count):
            if not _valid_component(stats, label):
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
                is_ambiguous = (second - first) < 8
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
            if len(xs) < 12:
                skipped[seed.plant_id] = "no vegetation connected to known plant centre"
                continue
            dx_mm = (xs - seed.center_px[0]) / calibration.pixels_per_mm_x
            dy_mm = (ys - seed.center_px[1]) / calibration.pixels_per_mm_y
            distances_mm = np.sqrt(dx_mm**2 + dy_mm**2)
            typical = float(np.percentile(distances_mm, 90))
            maximum = float(distances_mm.max())
            plant_ambiguous = bool(np.any(ambiguous & owned)) or index in uncertain_seeds
            component_coverage = min(1.0, len(xs) / 500.0)
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
                        },
                        separators=(",", ":"),
                    ),
                    algorithm_version=ALGORITHM_VERSION,
                    plant_age_days=age,
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
