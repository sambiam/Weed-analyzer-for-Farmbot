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
from .models import (
    AnalysisResult,
    Calibration,
    Decision,
    Measurement,
    OriginLocation,
    PlantSeed,
    WeedDetection,
)
from .resolution import MAX_PROCESSED_HEIGHT, MAX_PROCESSED_WIDTH
from .weed_settings import WeedSettings

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
    # A global 70th percentile discarded pale or shadowed leaves whenever a
    # large, very green plant dominated the frame. Otsu adapts to each image,
    # while the bounded threshold still rejects brown mulch and grey hardware.
    positive = np.clip(excess_green, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(positive, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    exg_threshold = max(10, min(32, int(otsu)))
    broad_green = (g > r * 0.92) & (g > b * 0.92) & (excess_green > exg_threshold)
    mask = (
        (hsv_green | broad_green)
        & (excess_green > exg_threshold)
        & (saturation > 22)
        & (value > 20)
    )
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


_PLANT_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 180, 40),
    (180, 80, 255),
    (40, 210, 255),
    (255, 120, 120),
    (180, 255, 80),
    (255, 80, 200),
)


def _tint_pixels(
    image: np.ndarray, selected: np.ndarray, color: tuple[int, int, int], alpha: float
) -> None:
    """Alpha-blend one flat colour without allocating another full frame."""
    if not np.any(selected):
        return
    pixels = image[selected].astype(np.float32)
    image[selected] = np.clip(pixels * (1 - alpha) + np.asarray(color) * alpha, 0, 255)


def _dashed_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    for start in range(0, 360, 24):
        cv2.ellipse(image, center, (radius, radius), 0, start, start + 12, color, thickness)


def _draw_legend(image: np.ndarray) -> None:
    entries = (
        ("vegetation", (40, 220, 40)),
        ("plant-owned pixels", _PLANT_COLORS[0]),
        ("ambiguous", (0, 0, 255)),
        ("typical / maximum / protected", (255, 255, 0)),
    )
    width = min(image.shape[1] - 8, 250)
    if width < 100:
        return
    height = 18 + len(entries) * 18
    panel = image[4 : 4 + height, 4 : 4 + width]
    panel[:] = (panel.astype(np.float32) * 0.35).astype(np.uint8)
    for index, (label, color) in enumerate(entries):
        y = 18 + index * 18
        cv2.rectangle(image, (12, y - 8), (24, y + 3), color, -1)
        cv2.putText(
            image,
            label,
            (30, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
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
        _tint_pixels(overlay, mask > 0, (40, 220, 40), 0.4)
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
        _draw_legend(overlay)
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
        weed_settings: WeedSettings | None = None,
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
        edge_truncated: set[int] = set()
        skipped: dict[int, str] = {}
        ambiguity_gap = max(5.0, params.mean_ppm * 5)

        valid_indices: list[int] = []
        for index, seed in enumerate(seeds):
            x, y = seed.center_px
            visible_radius = (
                max(
                    30.0,
                    seed.current_radius_mm + self.safety_margin_mm,
                )
                * params.mean_ppm
            )
            if (
                x + visible_radius < 0
                or y + visible_radius < 0
                or x - visible_radius >= width
                or y - visible_radius >= height
            ):
                skipped[seed.plant_id] = "plant protection area is outside this image"
                continue
            valid_indices.append(index)
            if x < 0 or y < 0 or x >= width or y >= height:
                edge_truncated.add(index)
            else:
                border = min(x, y, width - x, height - y)
                if border < visible_radius:
                    edge_truncated.add(index)

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
                core_radius_px = max(
                    6.0,
                    min(35.0, max(12.0, seed.current_radius_mm * 0.28)) * params.mean_ppm,
                )
                component_near_seed = np.any((xs - cx) ** 2 + (ys - cy) ** 2 <= core_radius_px**2)
                prior = previous_masks.get(seed.plant_id)
                historical_overlap = (
                    prior is not None and prior.shape == mask.shape and np.any(prior[ys, xs] > 0)
                )
                if component_near_seed or historical_overlap:
                    ownership[ys[candidate], xs[candidate]] = index + 1
                    ambiguous[ys[candidate & is_ambiguous], xs[candidate & is_ambiguous]] = True
                else:
                    nearest_distance = np.min(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2))
                    component_area_mm2 = len(xs) / (
                        calibration.pixels_per_mm_x * calibration.pixels_per_mm_y
                    )
                    soft_radius_px = (
                        max(seed.current_radius_mm * 1.35, seed.current_radius_mm + 60)
                        * params.mean_ppm
                    )
                    likely_disconnected_canopy = (
                        nearest_distance < soft_radius_px
                        and component_area_mm2 >= max(MIN_COMPONENT_AREA_MM2 * 12, 600)
                    )
                    if likely_disconnected_canopy:
                        # Soft ownership keeps a known neighbouring plant
                        # (for example marjoram beside spinach) out of the weed
                        # list while retaining uncertainty for automation.
                        ownership[ys[candidate], xs[candidate]] = index + 1
                    if nearest_distance < soft_radius_px:
                        uncertain_seeds.add(index)

        # Explain segmentation before adding geometry: all vegetation is green,
        # each plant's owned pixels have a stable palette colour, and ambiguous
        # pixels remain an unmistakable red warning.
        _tint_pixels(overlay, mask > 0, (40, 220, 40), 0.35)
        for index in valid_indices:
            _tint_pixels(
                overlay, ownership == index + 1, _PLANT_COLORS[index % len(_PLANT_COLORS)], 0.5
            )
        _tint_pixels(overlay, ambiguous, (0, 0, 255), 0.75)

        measurements: list[Measurement] = []
        for index in valid_indices:
            seed = seeds[index]
            owned = ownership == index + 1
            ys, xs = np.where(owned)
            age = None
            if seed.planted_at:
                age = max(0, (image_timestamp.date() - seed.planted_at.date()).days)
            transform_json = json.dumps(
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
            )
            cx, cy = seed.center_px
            core_radius_px = max(
                6.0,
                min(35.0, max(12.0, seed.current_radius_mm * 0.28)) * params.mean_ppm,
            )
            yy, xx = np.ogrid[:height, :width]
            core = (xx - cx) ** 2 + (yy - cy) ** 2 <= core_radius_px**2
            core_vegetation = int(np.count_nonzero((mask > 0) & core))
            core_area = max(1, int(np.count_nonzero(core)))
            core_coverage = core_vegetation / core_area
            # A plant must have evidence at its recorded centre. Vegetation
            # merely touching the outer radius is not proof the plant remains.
            center_present = core_coverage >= 0.035 or (
                mask[
                    min(height - 1, max(0, round(cy))),
                    min(width - 1, max(0, round(cx))),
                ]
                > 0
                and core_vegetation >= params.min_area
            )
            if len(xs) < params.min_area and index in edge_truncated:
                measurements.append(
                    Measurement(
                        measurement_id=uuid4(),
                        plant_id=seed.plant_id,
                        crop_slug=seed.crop_slug,
                        image_id=image_id,
                        image_timestamp=image_timestamp,
                        current_radius_mm=seed.current_radius_mm,
                        typical_canopy_radius_mm=0,
                        maximum_accepted_canopy_radius_mm=0,
                        recommended_protection_radius_mm=seed.current_radius_mm,
                        confidence=0.1,
                        decision=Decision.UNCERTAIN,
                        reason=(
                            "plant is only partly in frame and no connected canopy was found; "
                            "manual review is available"
                        ),
                        ambiguous=True,
                        calibration_version_id=calibration.version_id,
                        transform_json=transform_json,
                        algorithm_version=ALGORITHM_VERSION,
                        plant_age_days=age,
                        safety_margin_mm=self.safety_margin_mm,
                        calibration_uncertainty_mm=max(
                            self.calibration_uncertainty_mm, calibration.uncertainty_mm
                        ),
                        analysis_resolution=calibration.analysis_resolution,
                        processed_width=width,
                        processed_height=height,
                        calibration_source=calibration.source,
                        calibrated=True,
                        contract_version=CONTRACT_VERSION,
                    )
                )
                continue
            if len(xs) < params.min_area or (not center_present and index not in edge_truncated):
                nearby = (mask > 0) & (
                    (xx - cx) ** 2 + (yy - cy) ** 2
                    <= max(core_radius_px * 3, seed.current_radius_mm * params.mean_ppm) ** 2
                )
                nearby_y, nearby_x = np.where(nearby)
                suggested = (
                    (float(np.median(nearby_x)), float(np.median(nearby_y)))
                    if len(nearby_x) >= params.min_area
                    else None
                )
                absence_confidence = min(0.98, 0.72 + (0.25 * (1 - core_coverage / 0.035)))
                measurements.append(
                    Measurement(
                        measurement_id=uuid4(),
                        plant_id=seed.plant_id,
                        crop_slug=seed.crop_slug,
                        image_id=image_id,
                        image_timestamp=image_timestamp,
                        current_radius_mm=seed.current_radius_mm,
                        typical_canopy_radius_mm=0,
                        maximum_accepted_canopy_radius_mm=0,
                        recommended_protection_radius_mm=0,
                        confidence=absence_confidence,
                        decision=Decision.OBSERVED,
                        reason=(
                            "recorded plant centre is empty; vegetation remains nearby, so "
                            "removal is primary and moving the centre is secondary"
                            if suggested
                            else "no vegetation at the known in-frame plant centre"
                        ),
                        vegetation_absent=True,
                        center_misaligned=suggested is not None,
                        recommended_center_px=suggested,
                        calibration_version_id=calibration.version_id,
                        transform_json=transform_json,
                        algorithm_version=ALGORITHM_VERSION,
                        plant_age_days=age,
                        safety_margin_mm=self.safety_margin_mm,
                        calibration_uncertainty_mm=max(
                            self.calibration_uncertainty_mm, calibration.uncertainty_mm
                        ),
                        analysis_resolution=calibration.analysis_resolution,
                        processed_width=width,
                        processed_height=height,
                        calibration_source=calibration.source,
                        calibrated=True,
                        contract_version=CONTRACT_VERSION,
                    )
                )
                center = (round(seed.center_px[0]), round(seed.center_px[1]))
                absent_radius = max(10, round(seed.current_radius_mm * params.mean_ppm))
                _dashed_circle(overlay, center, absent_radius, (255, 0, 255))
                cv2.drawMarker(overlay, center, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(
                    overlay,
                    f"{seed.plant_id}: absent?",
                    (center[0] + 5, center[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                continue
            dx_mm = (xs - seed.center_px[0]) / calibration.pixels_per_mm_x
            dy_mm = (ys - seed.center_px[1]) / calibration.pixels_per_mm_y
            distances_mm = np.sqrt(dx_mm**2 + dy_mm**2)
            typical = float(np.percentile(distances_mm, 90))
            maximum = float(distances_mm.max())
            ambiguous_pixels = int(np.count_nonzero(ambiguous & owned))
            ambiguous_fraction = ambiguous_pixels / max(1, len(xs))
            # Overlap is expected in a mature bed. Only make the result
            # unreviewable when ambiguity dominates the evidence; preserve the
            # unambiguous core and nearest-seed pixels for growth measurement.
            plant_ambiguous = ambiguous_fraction > 0.45 or index in uncertain_seeds
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
                    - min(0.28, ambiguous_fraction * 0.35)
                    - (0.18 if index in uncertain_seeds else 0)
                    - (0.22 if index in edge_truncated else 0),
                ),
            )
            recommendation = (
                maximum
                + self.safety_margin_mm
                + max(self.calibration_uncertainty_mm, calibration.uncertainty_mm)
            )
            canopy_center = (float(np.median(xs)), float(np.median(ys)))
            center_offset_mm = math.hypot(
                (canopy_center[0] - seed.center_px[0]) / calibration.pixels_per_mm_x,
                (canopy_center[1] - seed.center_px[1]) / calibration.pixels_per_mm_y,
            )
            center_misaligned = center_offset_mm > max(
                20.0, min(60.0, seed.current_radius_mm * 0.3)
            )
            decision = Decision.UNCERTAIN if plant_ambiguous else Decision.OBSERVED
            reason = (
                "partial-frame or soft-owned canopy requires human review"
                if index in edge_truncated
                else "ownership is ambiguous or a new disconnected region needs history"
                if plant_ambiguous
                else "maximum accepted leaf extent plus safety and calibration margins"
            )
            if center_misaligned:
                reason += "; canopy centre is offset and can be moved during review"
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
                    center_misaligned=center_misaligned,
                    recommended_center_px=canopy_center if center_misaligned else None,
                    calibration_version_id=calibration.version_id,
                    transform_json=transform_json,
                    algorithm_version=ALGORITHM_VERSION,
                    plant_age_days=age,
                    safety_margin_mm=self.safety_margin_mm,
                    calibration_uncertainty_mm=max(
                        self.calibration_uncertainty_mm, calibration.uncertainty_mm
                    ),
                    analysis_resolution=calibration.analysis_resolution,
                    processed_width=width,
                    processed_height=height,
                    calibration_source=calibration.source,
                    calibrated=True,
                    contract_version=CONTRACT_VERSION,
                )
            )
            color = (0, 165, 255) if plant_ambiguous else _PLANT_COLORS[index % len(_PLANT_COLORS)]
            center = (round(seed.center_px[0]), round(seed.center_px[1]))
            cv2.circle(overlay, center, max(1, round(typical * params.mean_ppm)), (255, 255, 0), 1)
            cv2.circle(
                overlay,
                center,
                max(1, round(maximum * params.mean_ppm)),
                color,
                2,
            )
            cv2.circle(
                overlay,
                center,
                max(1, round(recommendation * params.mean_ppm)),
                (255, 255, 255),
                1,
            )
            cv2.circle(overlay, center, 3, color, -1)
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
        weed_detections: list[WeedDetection] = []
        if weed_settings and weed_settings.enabled:
            claimed = ownership > 0
            exclusion = np.zeros_like(mask)
            for seed in seeds:
                exclusion_radius = (
                    min(60.0, max(20.0, seed.current_radius_mm * 0.3))
                    + weed_settings.plant_exclusion_margin_mm
                ) * params.mean_ppm
                cv2.circle(
                    exclusion,
                    (round(seed.center_px[0]), round(seed.center_px[1])),
                    max(1, round(exclusion_radius)),
                    255,
                    -1,
                )
            candidate_mask = ((mask > 0) & ~claimed & (exclusion == 0)).astype(np.uint8) * 255
            count, candidate_labels, candidate_stats, candidate_centroids = (
                cv2.connectedComponentsWithStats(candidate_mask, 8)
            )
            area_scale = calibration.pixels_per_mm_x * calibration.pixels_per_mm_y
            owned_seed_indices = [
                index for index in range(len(seeds)) if np.any(ownership == index + 1)
            ]
            for label in range(1, count):
                area_mm2 = float(candidate_stats[label, cv2.CC_STAT_AREA] / area_scale)
                wx, wy = candidate_centroids[label]
                supported_seed = None
                supported_distance = math.inf
                for seed_index in owned_seed_indices:
                    seed = seeds[seed_index]
                    distance_mm = math.hypot(
                        (wx - seed.center_px[0]) / calibration.pixels_per_mm_x,
                        (wy - seed.center_px[1]) / calibration.pixels_per_mm_y,
                    )
                    support_radius_mm = min(100.0, seed.current_radius_mm + 40.0)
                    if distance_mm <= support_radius_mm and distance_mm < supported_distance:
                        supported_seed = seed_index
                        supported_distance = distance_mm
                if supported_seed is not None:
                    # Second-pass soft ownership: unconnected leaves clustered
                    # around an identified known plant are possible canopy,
                    # not a FarmBot weed.
                    component = candidate_labels == label
                    ownership[component] = supported_seed + 1
                    _tint_pixels(
                        overlay,
                        component,
                        _PLANT_COLORS[supported_seed % len(_PLANT_COLORS)],
                        0.35,
                    )
                    continue
                if not (
                    weed_settings.minimum_area_mm2 <= area_mm2 <= weed_settings.maximum_area_mm2
                ):
                    continue
                compactness = min(1.0, area_mm2 / max(weed_settings.minimum_area_mm2 * 2, 1))
                confidence = min(0.98, 0.62 + compactness * 0.28)
                if confidence < weed_settings.minimum_confidence:
                    continue
                weed_detections.append(
                    WeedDetection(
                        detection_id=uuid4(),
                        image_id=image_id,
                        image_timestamp=image_timestamp,
                        center_px=(float(wx), float(wy)),
                        area_mm2=area_mm2,
                        radius_mm=max(weed_settings.weed_radius_mm, math.sqrt(area_mm2 / math.pi)),
                        confidence=confidence,
                    )
                )
                cv2.drawMarker(
                    overlay,
                    (round(wx), round(wy)),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    18,
                    2,
                )
        _draw_legend(overlay)
        ok_mask, encoded_mask = cv2.imencode(".png", mask)
        ok_ownership, encoded_ownership = cv2.imencode(".png", ownership.astype(np.uint16))
        ok_overlay, encoded_overlay = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        # Release the large working arrays before returning (Part 7).
        del image, mask, overlay, labels, ownership, ambiguous
        return AnalysisResult(
            measurements=measurements,
            mask=encoded_mask.tobytes() if ok_mask else None,
            ownership_mask=encoded_ownership.tobytes() if ok_ownership else None,
            overlay_jpeg=encoded_overlay.tobytes() if ok_overlay else None,
            skipped=skipped,
            weeds=weed_detections,
        )


# Sign of the rotation applied when mapping garden coordinates into the
# *unrotated* processed image. FarmBot calibrates a ``Camera rotation`` and
# physically rotates each photo by it to align the frame with the garden axes
# (see FarmBot's plant_detection/P2C.py, which rotates the image and only then
# maps pixels<->coordinates linearly). We overlay on the unrotated image, so we
# apply the inverse rotation about the image centre. This constant selects the
# rotation direction; it is verified against real FarmBot images in the
# composite calibration view (flip it if a rotated camera overlays the wrong
# way).
ROTATION_SIGN = 1.0


def garden_to_pixel(
    plant_x: float,
    plant_y: float,
    image_x: float,
    image_y: float,
    width: int,
    height: int,
    calibration: Calibration,
) -> tuple[float, float]:
    """Map a garden coordinate to a pixel in the unrotated processed image.

    This is the algebraic inverse of FarmBot's own pixel->coordinate model. The
    metric offset from the image centre is scaled to pixels and reflected by the
    origin location *first* (garden<->pixel axis reflection), then rotated in
    pixel space by the inverse of the camera rotation -- exactly mirroring
    FarmBot rotating the image to align it before applying a pure scale. With
    ``rotation_degrees == 0`` this reduces to the historical identity map, so
    every origin/offset behaviour is preserved.
    """
    dx = plant_x - image_x + calibration.offset_x_mm
    dy = plant_y - image_y + calibration.offset_y_mm
    origin = OriginLocation(calibration.origin_location)
    # Pixel-space offset from centre, with the origin reflection applied before
    # rotation (in the garden->pixel direction).
    vx = origin.sign_x * dx * calibration.pixels_per_mm_x
    vy = origin.sign_y * dy * calibration.pixels_per_mm_y
    # Rotate by -rotation about the centre (inverse of the image-align rotation).
    theta = math.radians(ROTATION_SIGN * calibration.rotation_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rx = cos_t * vx + sin_t * vy
    ry = -sin_t * vx + cos_t * vy
    return (width / 2 + rx, height / 2 + ry)


def pixel_to_garden(
    pixel_x: float,
    pixel_y: float,
    image_x: float,
    image_y: float,
    width: int,
    height: int,
    calibration: Calibration,
) -> tuple[float, float]:
    """Inverse of :func:`garden_to_pixel`, used for weed and centre proposals."""
    rx, ry = pixel_x - width / 2, pixel_y - height / 2
    theta = math.radians(ROTATION_SIGN * calibration.rotation_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    vx = cos_t * rx - sin_t * ry
    vy = sin_t * rx + cos_t * ry
    origin = OriginLocation(calibration.origin_location)
    dx = vx / (origin.sign_x * calibration.pixels_per_mm_x)
    dy = vy / (origin.sign_y * calibration.pixels_per_mm_y)
    return (
        image_x + dx - calibration.offset_x_mm,
        image_y + dy - calibration.offset_y_mm,
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
