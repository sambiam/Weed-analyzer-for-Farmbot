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
from .canopy_radius import estimate_canopy_radius, previous_canopy_edge_mm
from .models import (
    AnalysisResult,
    Calibration,
    Decision,
    KnownWeedSeed,
    Measurement,
    OriginLocation,
    PlantSeed,
    WeedDetection,
)
from .plant_measurement import BOUNDARY_SECTOR_COUNT
from .resolution import MAX_PROCESSED_HEIGHT, MAX_PROCESSED_WIDTH
from .weed_settings import WeedSettings
from .weed_verifier import WeedVisualVerifier, encode_candidate_crop, extract_visual_features

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
# Weed proposals use their own recall mask.  In particular, do not apply the
# crop mask's 3 mm opening here: it erases narrow grass blades and the stems
# joining small rosette leaves before candidate grouping has a chance to see
# them.  A small closing repairs JPEG holes without inventing foreground.
WEED_CANDIDATE_CLOSE_MM = 2.0
# Weed leaves and thin stems often separate during colour segmentation even
# though they belong to one plant. Group nearby vegetation over this physical
# gap while retaining only the original vegetation pixels for area/features.
WEED_GROUP_MAX_GAP_MM = 12.0
# Proximity grouping must not chain across a weedy bed.  Disconnected leaves
# can form one proposal, but a chain of neighbouring plants is split once its
# bounding span reaches this limit.  A genuinely connected large rosette is
# kept intact; the bound applies only while merging separate islands.
WEED_GROUP_MAX_SPAN_MM = 120.0
# Pale leaves are often separated from the darker canopy by glare or a
# colour-temperature shift. Only recover such pixels when they are close to
# an already detected vegetation island; this prevents pale mulch from becoming
# global vegetation evidence.
PALE_LEAF_MIN_SATURATION = 5
PALE_LEAF_MIN_VALUE = 35
PALE_LEAF_MIN_EXCESS_GREEN = 2
PALE_LEAF_SUPPORT_GAP_MM = 10.0
# A vegetation island that touches a crop's exclusion zone is protected only
# this far beyond it. The grouping pass joins islands up to
# WEED_GROUP_MAX_GAP_MM apart, and without a reach limit those joins chain
# transitively: one weed near a crop protected an entire connected network of
# weeds right across the frame, which measured at ~70% of all vegetation in a
# weedy bed. The reach is comfortably wider than one grouping hop, so a crop
# leaf straddling the exclusion boundary is still protected whole.
CROP_SUPPORT_REACH_MM = 30.0

# Candidate generation is a recall stage: its job is to hand the verifier every
# blob that could be a weed, never to decide. A missed weed is invisible --
# it is never scored, stored, reviewed or labelled, so the mistake cannot be
# noticed or corrected -- while a false candidate costs one verifier call and
# one rejection. These ceilings therefore clamp the saved configuration so no
# size/shape value, including one inherited from an earlier build whose defaults
# could not be met by real foliage, can veto a candidate before it is scored.
# The confidence thresholds remain the real accept/reject stage.
#
# The area floor matches MIN_COMPONENT_AREA_MM2: nothing smaller survives the
# vegetation mask anyway, so a higher candidate floor only discards real weeds.
CANDIDATE_MIN_AREA_CEILING_MM2 = MIN_COMPONENT_AREA_MM2
CANDIDATE_GREEN_PURITY_CEILING = 0.05
CANDIDATE_SOLIDITY_CEILING = 0.05
CANDIDATE_CIRCULARITY_CEILING = 0.005
# ``aspect_ratio`` saturates at 12:1, so this floor disables the gate outright.
CANDIDATE_ASPECT_RATIO_FLOOR = 12.0
# How far past a crop's own canopy the no-candidate zone may reach while the
# verifier is scoring. The verifier receives ``distance_to_plant`` as a feature,
# so it can judge "too close to the crop to be a weed" far better than a fixed
# circle can -- and unlike the circle it cannot hide a weed it got wrong. The
# compounded default (1.2x radius + 25 mm + 35 mm) blanked 120 mm around a
# 60 mm plant, which is where interrow weeds actually grow.
CANDIDATE_EXCLUSION_REACH_CEILING_MM = 12.0
CANDIDATE_CANOPY_MULTIPLIER_CEILING = 1.05
# Reference size for the heuristic's area term. Scaling it by the configured
# minimum area instead made every score move when the size filter was retuned.
CANDIDATE_REFERENCE_AREA_MM2 = 150.0


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


@dataclass(frozen=True)
class CandidateGates:
    """The size/shape/colour thresholds actually used to admit a candidate.

    These are deliberately *not* the user's settings applied verbatim. Rejecting
    a blob here means the verifier never scores it, it is never stored, and it
    can never be reviewed or labelled -- so a gate set too tightly is invisible
    and self-reinforcing. Every gate is therefore clamped to an absolute recall
    ceiling, leaving the confidence thresholds as the real accept/reject stage.
    """

    min_area_mm2: float
    max_area_mm2: float
    green_purity: float
    solidity: float
    circularity: float
    aspect_ratio: float
    shape_filter_enabled: bool
    canopy_multiplier: float
    canopy_extra_mm: float
    exclusion_margin_mm: float

    @classmethod
    def build(cls, settings: WeedSettings, verifier_scoring: bool) -> CandidateGates:
        # A trained verifier turns the shape rules into pure recall aids, so the
        # recall boost applies whenever it is scoring -- shadow mode included,
        # where the entire point is to collect candidates worth labelling.
        boost = settings.candidate_recall_boost if verifier_scoring else 1.0
        return cls(
            min_area_mm2=min(settings.minimum_area_mm2, CANDIDATE_MIN_AREA_CEILING_MM2),
            # Area is supplied to the verifier as ``log_area_mm2`` and stored
            # with every review item.  A configured maximum is an automatic
            # action guard, not permission to make a large rosette invisible.
            max_area_mm2=math.inf,
            green_purity=min(settings.minimum_green_purity * boost, CANDIDATE_GREEN_PURITY_CEILING),
            solidity=min(settings.minimum_solidity * boost, CANDIDATE_SOLIDITY_CEILING),
            circularity=min(settings.minimum_circularity * boost, CANDIDATE_CIRCULARITY_CEILING),
            aspect_ratio=max(settings.maximum_aspect_ratio / boost, CANDIDATE_ASPECT_RATIO_FLOOR),
            shape_filter_enabled=settings.shape_filter_enabled,
            # The crop's own footprint is never clamped -- a 60 mm plant really
            # does occupy 60 mm. Only the padding *beyond* it is, and only while
            # the verifier can weigh distance-to-crop for itself. Grown-out
            # foliage is still protected by ownership and by the contiguous
            # crop-support pass, neither of which depends on this padding.
            canopy_multiplier=(
                min(settings.crop_support_radius_multiplier, CANDIDATE_CANOPY_MULTIPLIER_CEILING)
                if verifier_scoring
                else settings.crop_support_radius_multiplier
            ),
            canopy_extra_mm=(
                min(settings.crop_support_extra_mm, CANDIDATE_EXCLUSION_REACH_CEILING_MM)
                if verifier_scoring
                else settings.crop_support_extra_mm
            ),
            exclusion_margin_mm=(
                min(settings.plant_exclusion_margin_mm, CANDIDATE_EXCLUSION_REACH_CEILING_MM)
                if verifier_scoring
                else settings.plant_exclusion_margin_mm
            ),
        )

    def crop_canopy_mm(self, radius_mm: float) -> float:
        """The radius treated as the crop's own canopy for protection purposes."""
        return max(radius_mm * self.canopy_multiplier, radius_mm + self.canopy_extra_mm)

    def rejects(self, features: dict[str, float]) -> bool:
        """Whether the colour/shape gates veto this candidate."""
        if not self.shape_filter_enabled:
            return False
        return (
            features["strong_green_fraction"] < self.green_purity
            or features["solidity"] < self.solidity
            or features["circularity"] < self.circularity
            # ``aspect_ratio`` is stored normalised against the 12:1 saturation
            # point, so undo that before comparing with the configured ratio.
            or features["aspect_ratio"] * 12 > self.aspect_ratio
        )


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
        weed_settings: WeedSettings | None = None,
        known_weeds: list[KnownWeedSeed] | None = None,
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


def _vegetation_layers(
    image: np.ndarray, params: ScaleParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strong vegetation, recovered pale vegetation, and their union.

    The strong layer intentionally retains the established colour classifier.
    The second layer handles leaves washed towards white by the camera, but is
    only retained when its connected component is close to strong vegetation.
    Keeping the layers separate lets ownership use the same conservative
    attachment rule while still exposing the recovered pixels in diagnostics.
    """

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

    # Glare can reduce saturation and excess-green on genuine leaves without
    # changing their green hue. Require a small green colour bias as well as a
    # green hue/value range, so neutral grey and brown mulch do not qualify.
    pale_hue = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 105)
    pale_colour_bias = (g >= r - 6) & (g >= b - 6)
    pale = (
        pale_hue
        & (saturation >= PALE_LEAF_MIN_SATURATION)
        & (value >= PALE_LEAF_MIN_VALUE)
        & (excess_green >= PALE_LEAF_MIN_EXCESS_GREEN)
        & pale_colour_bias
    )
    pale_only = pale & ~mask

    # Recover a complete pale leaf only when it is anchored to the strong
    # vegetation layer. The support is a proximity check, not a dilation of
    # the returned mask, so it cannot paint the gap or surrounding soil.
    support_radius = max(1, round(PALE_LEAF_SUPPORT_GAP_MM * params.mean_ppm))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (support_radius * 2 + 1, support_radius * 2 + 1),
    )
    strong_u8 = mask.astype(np.uint8)
    supported_strong = cv2.dilate(strong_u8, support_kernel) > 0
    pale_labels_count, pale_labels = cv2.connectedComponents(pale_only.astype(np.uint8), 8)
    recovered_pale = np.zeros_like(mask, dtype=bool)
    for label in range(1, pale_labels_count):
        component = pale_labels == label
        if np.any(component & supported_strong):
            recovered_pale[component] = True

    combined = mask | recovered_pale
    binary = combined.astype(np.uint8) * 255
    open_k = np.ones((params.open_kernel, params.open_kernel), np.uint8)
    close_k = np.ones((params.close_kernel, params.close_kernel), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    combined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k) > 0
    return mask, recovered_pale, combined


def vegetation_mask(image: np.ndarray, params: ScaleParams) -> np.ndarray:
    """Return vegetation, including pale leaf sections anchored to foliage."""

    _strong, _pale, combined = _vegetation_layers(image, params)
    return combined.astype(np.uint8) * 255


def weed_candidate_vegetation_mask(
    image: np.ndarray,
    params: ScaleParams,
    settings: WeedSettings,
    canopy_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return recall-first vegetation used only for weed proposals.

    Crop measurement deliberately keeps its established conservative mask.
    Weed discovery instead honours the colour controls shown on the weed
    settings page.  Previously those controls were applied only to features
    *after* a fixed HSV/ExG mask and a 3 mm opening; pixels rejected there could
    never reach the verifier no matter how permissive the operator made the
    controls.

    The established canopy pixels are retained as a baseline.  Configured
    pixels are unioned with it and only closed, not opened, so thin foliage is
    preserved.  Component area and the verifier remain responsible for noise.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    b, g, r = cv2.split(image.astype(np.int16))
    excess_green = 2 * g - r - b
    if settings.green_hue_min <= settings.green_hue_max:
        hue_matches = (hue >= settings.green_hue_min) & (hue <= settings.green_hue_max)
    else:  # Defensive support for a hue band crossing OpenCV's 179 -> 0 seam.
        hue_matches = (hue >= settings.green_hue_min) | (hue <= settings.green_hue_max)
    configured = (
        hue_matches
        & (saturation >= settings.strong_green_minimum_saturation)
        & (excess_green >= settings.strong_green_minimum_excess_green)
        & (value > 20)
    )
    combined = configured
    if canopy_mask is not None and canopy_mask.shape == configured.shape:
        combined = combined | (canopy_mask > 0)
    close_radius = max(1, round((WEED_CANDIDATE_CLOSE_MM * params.mean_ppm) / 2))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_radius * 2 + 1, close_radius * 2 + 1),
    )
    return cv2.morphologyEx(combined.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)


def _core_radius_px(seed: PlantSeed, params: ScaleParams) -> float:
    return max(
        6.0,
        min(35.0, max(12.0, seed.current_radius_mm * 0.28)) * params.mean_ppm,
    )


def _absence_confidence(core_coverage: float) -> float:
    """Return bounded confidence for an empty known plant centre.

    Very small calibrated footprints can be completely covered while still
    containing fewer pixels than the minimum connected-component area. That
    is weak evidence, not a reason to emit a negative confidence.
    """

    return max(0.05, min(0.98, 0.72 + (0.25 * (1 - core_coverage / 0.035))))


def _valid_component(stats: np.ndarray, label: int, params: ScaleParams) -> bool:
    _, _, width, height, area = stats[label]
    aspect = max(width, height) / max(1, min(width, height))
    return params.min_area <= area <= params.max_area and not (
        aspect > 9 and area > params.irrigation_area
    )


def _nearby_component_labels(binary: np.ndarray, max_gap_px: float) -> tuple[int, np.ndarray]:
    """Label foreground islands as one object when their edges are nearby.

    Labels are calculated on a dilated support mask, but callers select pixels
    from the original binary mask. The joining pixels therefore affect only
    grouping, never measured weed area, colour, shape, or the review crop.
    """
    radius = max(1, round(max_gap_px / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    support = cv2.dilate((binary > 0).astype(np.uint8), kernel)
    count, labels = cv2.connectedComponents(support, 8)
    return count, labels


def _bounded_nearby_component_labels(
    binary: np.ndarray, max_gap_px: float, max_span_px: float
) -> tuple[int, np.ndarray]:
    """Group nearby islands without transitive chains crossing the frame.

    A normal dilation makes A join B and B join C even when A and C are
    different weeds.  Here original connected islands are merged by increasing
    bounding-box gap only while the combined group stays within a physical
    span.  Foreground pixels alone receive labels, so grouping never changes
    measured area or visual features.
    """

    count, base_labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8
    )
    if count <= 2:
        return count, base_labels
    parent = list(range(count))
    bounds = {
        label: [
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
        ]
        for label in range(1, count)
    }

    def root(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    # Spatial buckets avoid an O(islands²) scan on noisy soil. Each component
    # is registered in the grid cells touched by its gap-expanded box; only
    # components sharing a cell can possibly be close enough to merge.
    cell_size = max(1.0, max_gap_px)
    buckets: dict[tuple[int, int], list[int]] = {}
    for label, (x0, y0, x1, y1) in bounds.items():
        cell_x0 = math.floor((x0 - max_gap_px) / cell_size)
        cell_y0 = math.floor((y0 - max_gap_px) / cell_size)
        cell_x1 = math.floor((x1 + max_gap_px) / cell_size)
        cell_y1 = math.floor((y1 + max_gap_px) / cell_size)
        for cell_y in range(cell_y0, cell_y1 + 1):
            for cell_x in range(cell_x0, cell_x1 + 1):
                buckets.setdefault((cell_x, cell_y), []).append(label)
    possible_pairs: set[tuple[int, int]] = set()
    for labels_in_cell in buckets.values():
        for position, left in enumerate(labels_in_cell):
            for right in labels_in_cell[position + 1 :]:
                if left != right:
                    possible_pairs.add((min(left, right), max(left, right)))
    pairs: list[tuple[float, int, int]] = []
    for left, right in possible_pairs:
        lx0, ly0, lx1, ly1 = bounds[left]
        rx0, ry0, rx1, ry1 = bounds[right]
        gap_x = max(0, max(lx0, rx0) - min(lx1, rx1))
        gap_y = max(0, max(ly0, ry0) - min(ly1, ry1))
        gap = math.hypot(gap_x, gap_y)
        if gap <= max_gap_px:
            pairs.append((gap, left, right))
    for _gap, left, right in sorted(pairs):
        left_root, right_root = root(left), root(right)
        if left_root == right_root:
            continue
        left_bounds, right_bounds = bounds[left_root], bounds[right_root]
        merged = [
            min(left_bounds[0], right_bounds[0]),
            min(left_bounds[1], right_bounds[1]),
            max(left_bounds[2], right_bounds[2]),
            max(left_bounds[3], right_bounds[3]),
        ]
        if max(merged[2] - merged[0], merged[3] - merged[1]) > max_span_px:
            continue
        parent[right_root] = left_root
        bounds[left_root] = merged

    output = np.zeros_like(base_labels)
    relabel: dict[int, int] = {}
    for label in range(1, count):
        component_root = root(label)
        output_label = relabel.setdefault(component_root, len(relabel) + 1)
        output[base_labels == label] = output_label
    return len(relabel) + 1, output


def _circle_visible_fraction(
    center: tuple[float, float], radius_px: float, width: int, height: int
) -> float:
    """Estimate how much of a circular canopy can be inspected in this frame."""
    if radius_px <= 1:
        x, y = center
        return 1.0 if 0 <= x < width and 0 <= y < height else 0.0
    axis = np.linspace(-1.0, 1.0, 61)
    xx, yy = np.meshgrid(axis, axis)
    inside = (xx * xx + yy * yy) <= 1.0
    px = center[0] + xx * radius_px
    py = center[1] + yy * radius_px
    visible = inside & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return float(np.count_nonzero(visible) / max(1, np.count_nonzero(inside)))


def _pixel_offsets_to_world_mm(
    dx: np.ndarray | float,
    dy: np.ndarray | float,
    calibration: Calibration,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Rotate and scale image offsets into calibrated bed-space millimetres."""

    theta = math.radians(ROTATION_SIGN * calibration.rotation_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    vx = cos_t * dx - sin_t * dy
    vy = sin_t * dx + cos_t * dy
    origin = OriginLocation(calibration.origin_location)
    return (
        vx / (origin.sign_x * calibration.pixels_per_mm_x),
        vy / (origin.sign_y * calibration.pixels_per_mm_y),
    )


def _world_offsets_to_pixel(
    dx_mm: np.ndarray,
    dy_mm: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`_pixel_offsets_to_world_mm` for visibility sampling."""

    origin = OriginLocation(calibration.origin_location)
    vx = origin.sign_x * dx_mm * calibration.pixels_per_mm_x
    vy = origin.sign_y * dy_mm * calibration.pixels_per_mm_y
    theta = math.radians(ROTATION_SIGN * calibration.rotation_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return cos_t * vx + sin_t * vy, -sin_t * vx + cos_t * vy


def _canopy_visibility(
    center: tuple[float, float],
    radius_mm: float,
    width: int,
    height: int,
    calibration: Calibration,
) -> tuple[float, list[int]]:
    """Return visible canopy area and visible outer-boundary sectors.

    The boundary uses 72 equal angular sectors in bed space. A sector is visible
    when its outer-canopy sample maps inside the calibrated source image.
    """

    radius_mm = max(1.0, float(radius_mm))
    angles = np.arange(BOUNDARY_SECTOR_COUNT, dtype=np.float64) * (
        2 * math.pi / BOUNDARY_SECTOR_COUNT
    )
    boundary_dx, boundary_dy = _world_offsets_to_pixel(
        np.cos(angles) * radius_mm,
        np.sin(angles) * radius_mm,
        calibration,
    )
    boundary_x = center[0] + boundary_dx
    boundary_y = center[1] + boundary_dy
    sectors = (
        np.flatnonzero(
            (boundary_x >= 0) & (boundary_x < width) & (boundary_y >= 0) & (boundary_y < height)
        )
        .astype(int)
        .tolist()
    )

    # A deterministic 41x41 disc sample measures visible canopy area in the
    # same bed-space circle rather than in an uncalibrated display-pixel circle.
    axis = np.linspace(-1.0, 1.0, 41)
    xx, yy = np.meshgrid(axis, axis)
    inside = (xx * xx + yy * yy) <= 1.0
    area_dx, area_dy = _world_offsets_to_pixel(
        xx * radius_mm,
        yy * radius_mm,
        calibration,
    )
    area_x = center[0] + area_dx
    area_y = center[1] + area_dy
    visible = inside & (area_x >= 0) & (area_x < width) & (area_y >= 0) & (area_y < height)
    area_fraction = float(np.count_nonzero(visible) / max(1, np.count_nonzero(inside)))
    return area_fraction, sectors


@dataclass(frozen=True)
class BoundaryVerificationResult:
    """Accepted plant evidence after known-weed and learned boundary checks."""

    accepted: np.ndarray
    verified_weed: np.ndarray
    uncertain: bool
    notes: tuple[str, ...]
    stats: dict[str, int]


def _verify_new_boundary(
    image: np.ndarray,
    owned: np.ndarray,
    seed: PlantSeed,
    calibration: Calibration,
    protection_margin_mm: float,
    previous_mask: np.ndarray | None,
    known_weeds: list[KnownWeedSeed],
    weed_settings: WeedSettings | None,
    verifier: WeedVisualVerifier | None,
) -> BoundaryVerificationResult:
    """Check only vegetation newly extending beyond the prior canopy edge.

    Known FarmBot weed points are authoritative outside established canopy.
    The learned verifier is deliberately tri-state: confirmed crop is kept,
    confirmed weed/non-crop is removed, and uncertain new growth is held for a
    later observation. Without a trained crop category head the geometric mask
    remains the fallback, so an older binary model cannot freeze plant growth.
    """

    accepted = owned.copy()
    verified_weed = np.zeros_like(owned, dtype=bool)
    stats = {
        "known_weed_regions": 0,
        "known_weed_pixels_removed": 0,
        "components_checked": 0,
        "crop_accepted": 0,
        "weed_rejected": 0,
        "noncrop_rejected": 0,
        "uncertain_held": 0,
        "shadow_scored": 0,
        "geometric_fallback": 0,
    }
    notes: list[str] = []
    ys, xs = np.where(owned)
    if not len(xs):
        return BoundaryVerificationResult(accepted, verified_weed, False, (), stats)

    dx_mm, dy_mm = _pixel_offsets_to_world_mm(
        xs - seed.center_px[0], ys - seed.center_px[1], calibration
    )
    distances_mm = np.hypot(dx_mm, dy_mm)
    prior_edge_mm, _margin_aware = previous_canopy_edge_mm(
        seed.current_radius_mm, protection_margin_mm
    )
    # Ignore a narrow band at the old edge: JPEG/mask jitter there is not new
    # biological growth and should not consume verifier calls.
    new_point = distances_mm > prior_edge_mm + max(2.0, calibration.uncertainty_mm * 0.2)
    historical_point = np.zeros(len(xs), dtype=bool)
    if previous_mask is not None and previous_mask.shape == owned.shape:
        historical_point = previous_mask[ys, xs] > 0
    point_sectors = np.floor(
        (np.mod(np.arctan2(dy_mm, dx_mm), 2 * math.pi) / (2 * math.pi)) * BOUNDARY_SECTOR_COUNT
    ).astype(np.int32)

    fallback_weed_radius = weed_settings.weed_radius_mm if weed_settings is not None else 15.0
    for weed in known_weeds:
        weed_dx_mm, weed_dy_mm = _pixel_offsets_to_world_mm(
            xs - weed.center_px[0], ys - weed.center_px[1], calibration
        )
        weed_radius_mm = max(float(weed.radius_mm), float(fallback_weed_radius))
        overlap = new_point & (np.hypot(weed_dx_mm, weed_dy_mm) <= weed_radius_mm)
        if not np.any(overlap):
            continue
        # Remove the complete outward radial path feeding the known weed, not
        # only the pixels inside its FarmBot circle. Otherwise a green bridge
        # left between the old canopy and that circle could still define a
        # false leaf tip. One neighbouring sector on either side absorbs map
        # and segmentation jitter without affecting growth elsewhere.
        weed_sectors = set(point_sectors[overlap].tolist())
        weed_sectors |= {
            (sector + offset) % BOUNDARY_SECTOR_COUNT
            for sector in tuple(weed_sectors)
            for offset in (-1, 1)
        }
        directional_overlap = new_point & np.isin(point_sectors, list(weed_sectors))
        before = int(np.count_nonzero(accepted[ys, xs]))
        accepted[ys[directional_overlap], xs[directional_overlap]] = False
        stats["known_weed_regions"] += 1
        stats["known_weed_pixels_removed"] += before - int(np.count_nonzero(accepted[ys, xs]))
    if stats["known_weed_regions"]:
        notes.append(
            f"excluded overlap with {stats['known_weed_regions']} known FarmBot weed"
            f"{'s' if stats['known_weed_regions'] != 1 else ''}"
        )

    learned_enabled = bool(
        weed_settings
        and weed_settings.enabled
        and weed_settings.visual_verifier_enabled
        and weed_settings.boundary_verifier_enabled
        and verifier is not None
        and verifier.available
    )
    if not learned_enabled:
        stats["geometric_fallback"] = 1
        return BoundaryVerificationResult(accepted, verified_weed, False, tuple(notes), stats)

    new_growth = np.zeros_like(owned, dtype=np.uint8)
    still_accepted = accepted[ys, xs]
    candidate_points = new_point & ~historical_point & still_accepted
    new_growth[ys[candidate_points], xs[candidate_points]] = 255
    component_count, component_labels, component_stats, component_centroids = (
        cv2.connectedComponentsWithStats(new_growth, 8)
    )
    minimum_area_px = max(
        3,
        round(MIN_COMPONENT_AREA_MM2 * calibration.pixels_per_mm_x * calibration.pixels_per_mm_y),
    )
    uncertain = False
    has_crop_head = bool(
        (getattr(verifier, "model", None) or {}).get("class_heads", {}).get("crop")
    )
    if not has_crop_head:
        stats["geometric_fallback"] = 1

    for label in range(1, component_count):
        area_px = int(component_stats[label, cv2.CC_STAT_AREA])
        if area_px < minimum_area_px:
            continue
        component = component_labels == label
        area_mm2 = area_px / (calibration.pixels_per_mm_x * calibration.pixels_per_mm_y)
        component_x, component_y = component_centroids[label]
        center_dx_mm, center_dy_mm = _pixel_offsets_to_world_mm(
            component_x - seed.center_px[0], component_y - seed.center_px[1], calibration
        )
        features = extract_visual_features(
            image,
            component,
            area_mm2,
            green_hue_min=weed_settings.green_hue_min,
            green_hue_max=weed_settings.green_hue_max,
            strong_green_minimum_saturation=weed_settings.strong_green_minimum_saturation,
            strong_green_minimum_excess_green=(weed_settings.strong_green_minimum_excess_green),
            distance_to_plant_mm=math.hypot(center_dx_mm, center_dy_mm),
        )
        weed_probability = verifier.predict(features)
        explanations = dict(verifier.explain(features)) if hasattr(verifier, "explain") else {}
        stats["components_checked"] += 1
        if weed_settings.visual_verifier_shadow_mode:
            stats["shadow_scored"] += 1
            continue
        if weed_probability is not None and (
            weed_probability >= weed_settings.visual_verifier_minimum_confidence
        ):
            accepted[component] = False
            verified_weed[component] = True
            stats["weed_rejected"] += 1
            continue
        if not has_crop_head:
            continue
        if not explanations:
            stats["geometric_fallback"] = 1
            continue
        crop_probability = float(explanations.get("crop", 0.0))
        top_label, top_probability = max(explanations.items(), key=lambda item: item[1])
        if (
            top_label == "crop"
            and crop_probability >= weed_settings.boundary_crop_minimum_confidence
        ):
            stats["crop_accepted"] += 1
        elif (
            top_label != "crop"
            and top_probability >= weed_settings.boundary_noncrop_minimum_confidence
        ):
            accepted[component] = False
            stats["noncrop_rejected"] += 1
        else:
            accepted[component] = False
            stats["uncertain_held"] += 1
            uncertain = True

    if stats["components_checked"]:
        notes.append(
            "boundary verifier: "
            f"{stats['crop_accepted']} crop accepted, "
            f"{stats['weed_rejected']} weed and {stats['noncrop_rejected']} non-crop rejected, "
            f"{stats['uncertain_held']} uncertain held"
        )
    return BoundaryVerificationResult(accepted, verified_weed, uncertain, tuple(notes), stats)


def _image_quality(image: np.ndarray) -> float:
    """Compact sharpness/exposure score used as one confidence factor."""

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, math.log1p(float(cv2.Laplacian(grey, cv2.CV_32F).var())) / math.log(101))
    clipped = float(np.mean((grey <= 5) | (grey >= 250)))
    exposure = max(0.0, 1.0 - clipped * 2.5)
    return float(np.clip(0.6 * sharpness + 0.4 * exposure, 0.05, 1.0))


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
    def __init__(
        self,
        safety_margin_mm: float = 20,
        calibration_uncertainty_mm: float = 10,
        weed_verifier: WeedVisualVerifier | None = None,
    ):
        self.safety_margin_mm = safety_margin_mm
        self.calibration_uncertainty_mm = calibration_uncertainty_mm
        self.weed_verifier = weed_verifier

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
        known_weeds: list[KnownWeedSeed] | None = None,
    ) -> AnalysisResult:
        image = decode_jpeg(image_bytes)
        params = ScaleParams.build(image.shape[1], image.shape[0], calibration)
        _strong_mask, _recovered_pale, mask = _vegetation_layers(image, params)
        mask = mask.astype(np.uint8) * 255
        # Normalize any historical masks to this resolution (Part 8); reject
        # those from an incompatible aspect ratio.
        normalized: dict[int, np.ndarray] = {}
        for plant_id, prior in (previous_masks or {}).items():
            fitted = resize_prior_mask(prior, mask.shape)
            if fitted is not None:
                normalized[plant_id] = fitted
        previous_masks = normalized
        known_weeds = list(known_weeds or [])
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
        # Break hairline bridges before ownership. A weed touching a crop
        # through a few green pixels must not become radius evidence.
        ownership_input = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(ownership_input, 8)
        centers = np.array([seed.center_px for seed in seeds], dtype=np.float32)
        overlay = image.copy()
        image_quality = _image_quality(image)
        weed_review = image.copy() if weed_settings and weed_settings.enabled else None
        ownership = np.zeros_like(labels, dtype=np.int16)
        ambiguous = np.zeros_like(mask, dtype=bool)
        boundary_verified_weed = np.zeros_like(mask, dtype=bool)
        boundary_diagnostics: dict[int, dict[str, object]] = {}
        boundary_stats: dict[str, int] = {}
        uncertain_seeds: set[int] = set()
        edge_truncated: set[int] = set()
        out_of_frame: set[int] = set()
        expected_visibility: dict[int, tuple[float, list[int]]] = {}
        skipped: dict[int, str] = {}
        ambiguity_gap = max(5.0, params.mean_ppm * 5)

        valid_indices: list[int] = []
        for index, seed in enumerate(seeds):
            x, y = seed.center_px
            expected_radius_mm = max(
                30.0,
                seed.current_radius_mm + self.safety_margin_mm,
            )
            expected_visibility[index] = _canopy_visibility(
                seed.center_px,
                expected_radius_mm,
                width,
                height,
                calibration,
            )
            if expected_visibility[index][0] <= 0:
                # Fully off-frame: there is no pixel evidence to analyse, so
                # this plant never joins ownership/connected-components. It
                # still gets a low-confidence Measurement below so it surfaces
                # for manual review instead of silently vanishing.
                out_of_frame.add(index)
                continue
            valid_indices.append(index)
            if x < 0 or y < 0 or x >= width or y >= height:
                edge_truncated.add(index)
            else:
                if len(expected_visibility[index][1]) < BOUNDARY_SECTOR_COUNT:
                    edge_truncated.add(index)

        for label in range(1, labels_count):
            component = labels == label
            ys, xs = np.where(component)
            if not len(xs) or not valid_indices:
                continue
            if not _valid_component(stats, label, params):
                # Long, narrow crop leaves (especially chives) resemble
                # irrigation lines by geometry. Admit one only when it
                # reaches a known plant's centre; an isolated line/weed keeps
                # the existing rejection path.
                component_area = int(stats[label, cv2.CC_STAT_AREA])
                in_area_range = params.min_area <= component_area <= params.max_area
                touches_plant_core = any(
                    np.any(
                        (xs - seeds[index].center_px[0]) ** 2
                        + (ys - seeds[index].center_px[1]) ** 2
                        <= _core_radius_px(seeds[index], params) ** 2
                    )
                    for index in valid_indices
                )
                if not in_area_range or not touches_plant_core:
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
                core_radius_px = _core_radius_px(seed, params)
                component_near_seed = np.any((xs - cx) ** 2 + (ys - cy) ** 2 <= core_radius_px**2)
                prior = previous_masks.get(seed.plant_id)
                historical_overlap = (
                    prior is not None and prior.shape == mask.shape and np.any(prior[ys, xs] > 0)
                )
                candidate_farthest = float(
                    np.max(np.sqrt((xs[candidate] - cx) ** 2 + (ys[candidate] - cy) ** 2))
                )
                candidate_dx_mm, candidate_dy_mm = _pixel_offsets_to_world_mm(
                    xs[candidate] - cx,
                    ys[candidate] - cy,
                    calibration,
                )
                candidate_nearest_mm = float(np.min(np.hypot(candidate_dx_mm, candidate_dy_mm)))
                bounded_historical_leaf = historical_overlap and candidate_farthest <= (
                    max(seed.current_radius_mm * 1.5, seed.current_radius_mm + 30) * params.mean_ppm
                )
                intersects_expected_canopy = candidate_nearest_mm <= max(
                    30.0,
                    seed.current_radius_mm * 1.25,
                    seed.current_radius_mm + self.safety_margin_mm,
                )
                center_outside = not (0 <= cx < width and 0 <= cy < height)
                if (
                    component_near_seed
                    or bounded_historical_leaf
                    or (center_outside and intersects_expected_canopy)
                ):
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
                    # History and proximity may flag a region for review, but
                    # disconnected vegetation never expands a crop radius.
                    if likely_disconnected_canopy or historical_overlap:
                        ambiguous[ys[candidate], xs[candidate]] = True
                    if nearest_distance < soft_radius_px or historical_overlap:
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

        # Invariant across every plant in this image, so it's built once.
        transform_json = json.dumps(
            {
                "pixels_per_mm_x": calibration.pixels_per_mm_x,
                "pixels_per_mm_y": calibration.pixels_per_mm_y,
                "rotation_degrees": calibration.rotation_degrees,
                "offset_x_mm": calibration.offset_x_mm,
                "offset_y_mm": calibration.offset_y_mm,
                "origin_location": calibration.origin_location.value,
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

        def evidence_metadata(
            seed_index: int,
            radius_mm: float,
            *,
            has_evidence: bool,
            segmentation_quality: float,
            canopy_truncated: bool = False,
            exclusion_reason: str | None = None,
        ) -> dict[str, object]:
            seed = seeds[seed_index]
            area_fraction, sectors = _canopy_visibility(
                seed.center_px,
                max(1.0, radius_mm),
                width,
                height,
                calibration,
            )
            center_visible = 0 <= seed.center_px[0] < width and 0 <= seed.center_px[1] < height
            expected_sectors = expected_visibility.get(seed_index, (0.0, []))[1]
            boundary_coverage = len(sectors) / BOUNDARY_SECTOR_COUNT
            details = {
                "plant_id": seed.plant_id,
                "plant_center_px": [float(seed.center_px[0]), float(seed.center_px[1])],
                "center_visible": center_visible,
                "visible_canopy_fraction": area_fraction,
                "visible_boundary_coverage": boundary_coverage,
                "visible_boundary_sectors": sectors,
                "image_quality": image_quality,
                "segmentation_quality": segmentation_quality,
                "calibration_source": calibration.source,
                "calibration_uncertainty_mm": calibration.uncertainty_mm,
            }
            details.update(boundary_diagnostics.get(seed_index, {}))
            return {
                "visible_fraction": area_fraction,
                "center_visible": center_visible,
                "boundary_coverage": boundary_coverage,
                "boundary_sectors": sectors,
                "canopy_truncated": canopy_truncated,
                "has_plant_evidence": has_evidence,
                "plant_fits_single_frame": (len(expected_sectors) == BOUNDARY_SECTOR_COUNT),
                "image_quality": image_quality,
                "segmentation_quality": float(np.clip(segmentation_quality, 0.0, 1.0)),
                "evidence_status": "useful" if has_evidence else "excluded",
                "exclusion_reason": exclusion_reason,
                "diagnostics_json": json.dumps(details, separators=(",", ":")),
            }

        measurements: list[Measurement] = []
        for index in out_of_frame:
            seed = seeds[index]
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
                    typical_canopy_radius_mm=0,
                    maximum_accepted_canopy_radius_mm=0,
                    recommended_protection_radius_mm=seed.current_radius_mm,
                    confidence=0.05,
                    decision=Decision.UNCERTAIN,
                    reason=(
                        "plant protection area is entirely outside this image; "
                        "no automatic change is made and manual review is available"
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
                    plant_center_px=seed.center_px,
                    **evidence_metadata(
                        index,
                        max(1.0, seed.current_radius_mm),
                        has_evidence=False,
                        segmentation_quality=0.0,
                        exclusion_reason="plant footprint is outside this image",
                    ),
                )
            )
        for index in valid_indices:
            seed = seeds[index]
            owned = ownership == index + 1
            ys, xs = np.where(owned)
            age = None
            if seed.planted_at:
                age = max(0, (image_timestamp.date() - seed.planted_at.date()).days)
            cx, cy = seed.center_px
            core_radius_px = _core_radius_px(seed, params)
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
                        plant_center_px=seed.center_px,
                        **evidence_metadata(
                            index,
                            max(1.0, seed.current_radius_mm),
                            has_evidence=False,
                            segmentation_quality=0.0,
                            exclusion_reason="no connected target canopy was found",
                        ),
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
                absence_confidence = _absence_confidence(core_coverage)
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
                        plant_center_px=seed.center_px,
                        **evidence_metadata(
                            index,
                            max(1.0, seed.current_radius_mm),
                            has_evidence=False,
                            segmentation_quality=core_coverage,
                            exclusion_reason="no vegetation at the known plant centre",
                        ),
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
            effective_calibration_margin = max(
                self.calibration_uncertainty_mm, calibration.uncertainty_mm
            )
            boundary_result = _verify_new_boundary(
                image,
                owned,
                seed,
                calibration,
                self.safety_margin_mm + effective_calibration_margin,
                previous_masks.get(seed.plant_id),
                known_weeds,
                weed_settings,
                self.weed_verifier,
            )
            for stat_name, stat_value in boundary_result.stats.items():
                boundary_stats[stat_name] = boundary_stats.get(stat_name, 0) + stat_value
            boundary_diagnostics[index] = {
                "boundary_verifier": boundary_result.stats,
                "boundary_verifier_notes": list(boundary_result.notes),
            }
            rejected_boundary = owned & ~boundary_result.accepted
            if np.any(rejected_boundary):
                ownership[rejected_boundary] = 0
                boundary_verified_weed |= boundary_result.verified_weed
                owned = boundary_result.accepted
                ys, xs = np.where(owned)
            boundary_uncertain = boundary_result.uncertain
            dx_mm, dy_mm = _pixel_offsets_to_world_mm(
                xs - seed.center_px[0],
                ys - seed.center_px[1],
                calibration,
            )
            distances_mm = np.hypot(dx_mm, dy_mm)
            angles_radians = np.arctan2(dy_mm, dx_mm)
            radius_estimate = estimate_canopy_radius(
                distances_mm,
                angles_radians,
                current_radius_mm=seed.current_radius_mm,
                protection_margin_mm=self.safety_margin_mm + effective_calibration_margin,
            )
            if radius_estimate is None:
                # Connected-component validation above guarantees meaningful
                # foreground, but retain a conservative fallback for extremely
                # small/angularly degenerate calibrated masks.
                typical = float(np.percentile(distances_mm, 90))
                maximum = float(np.percentile(distances_mm, 98))
                clipped_fraction = 0.0
                broad_overreach = False
            else:
                keep = radius_estimate.keep
                if not np.all(keep):
                    ownership[ys[~keep], xs[~keep]] = 0
                    ys = ys[keep]
                    xs = xs[keep]
                    dx_mm = np.asarray(dx_mm)[keep]
                    dy_mm = np.asarray(dy_mm)[keep]
                    distances_mm = distances_mm[keep]
                    owned = ownership == index + 1
                typical = radius_estimate.typical_radius_mm
                maximum = radius_estimate.outer_radius_mm
                clipped_fraction = radius_estimate.clipped_point_fraction
                broad_overreach = radius_estimate.broad_overreach
            _visible_fraction, visible_boundary_sectors = _canopy_visibility(
                seed.center_px,
                max(seed.current_radius_mm, maximum),
                width,
                height,
                calibration,
            )
            boundary_coverage = len(visible_boundary_sectors) / BOUNDARY_SECTOR_COUNT
            canopy_truncated = bool(
                np.any(owned[:2, :])
                or np.any(owned[-2:, :])
                or np.any(owned[:, :2])
                or np.any(owned[:, -2:])
            )
            ambiguous_pixels = int(np.count_nonzero(ambiguous & owned))
            ambiguous_fraction = ambiguous_pixels / max(1, len(xs))
            # Overlap is expected in a mature bed. Only make the result
            # unreviewable when ambiguity dominates the evidence; preserve the
            # unambiguous core and nearest-seed pixels for growth measurement.
            plant_ambiguous = ambiguous_fraction > 0.45 or index in uncertain_seeds
            component_coverage = min(1.0, len(xs) / (500.0 * params.mean_ppm**2))
            segmentation_quality = float(
                np.clip(
                    0.5 * component_coverage
                    + 0.3 * min(1.0, core_coverage / 0.18)
                    + 0.2 * (1.0 - ambiguous_fraction),
                    0.0,
                    1.0,
                )
            )
            confidence = max(
                0.05,
                min(
                    0.99,
                    0.34
                    + 0.25 * component_coverage
                    + 0.23 * boundary_coverage
                    + 0.10 * image_quality
                    + 0.08 * segmentation_quality
                    - min(0.28, ambiguous_fraction * 0.35)
                    - (0.18 if index in uncertain_seeds else 0)
                    - (0.08 if canopy_truncated else 0),
                ),
            )
            if broad_overreach or boundary_uncertain:
                # A clipped broad expansion is useful review evidence, but it
                # must never pass the automatic-write confidence threshold.
                # The same is true when the learned verifier held an uncertain
                # new boundary region for another observation.
                confidence = min(confidence, 0.74)
            recommendation = maximum + self.safety_margin_mm + effective_calibration_margin
            canopy_center = (float(np.median(xs)), float(np.median(ys)))
            center_offset_x_mm, center_offset_y_mm = _pixel_offsets_to_world_mm(
                canopy_center[0] - seed.center_px[0],
                canopy_center[1] - seed.center_px[1],
                calibration,
            )
            center_offset_mm = math.hypot(center_offset_x_mm, center_offset_y_mm)
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
            if broad_overreach:
                reason += (
                    f"; broad outer-mask expansion was rejected as implausible "
                    f"single-observation growth ({clipped_fraction:.0%} of owned pixels removed)"
                )
            if boundary_result.notes:
                reason += "; " + "; ".join(boundary_result.notes)
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
                    plant_center_px=seed.center_px,
                    **evidence_metadata(
                        index,
                        max(seed.current_radius_mm, maximum),
                        has_evidence=True,
                        segmentation_quality=segmentation_quality,
                        canopy_truncated=canopy_truncated,
                    ),
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
        candidate_stats: dict[str, int] = {}
        if weed_settings and weed_settings.enabled:
            # When the learned verifier is doing the accepting and rejecting,
            # the heuristic's remaining job is recall: hand it every candidate
            # worth judging rather than pre-filtering with rules that cannot
            # tell a weed from moss anyway.
            verifier_scoring = bool(
                weed_settings.visual_verifier_enabled
                and self.weed_verifier is not None
                and self.weed_verifier.available
            )
            verifier_authoritative = (
                verifier_scoring
                and weed_settings.visual_verifier_enabled
                and not weed_settings.visual_verifier_shadow_mode
            )
            gates = CandidateGates.build(weed_settings, verifier_scoring)
            claimed = ownership > 0
            candidate_vegetation = (
                weed_candidate_vegetation_mask(image, params, weed_settings, mask) > 0
            )
            exclusion = np.zeros_like(mask)
            for seed in seeds:
                if weed_settings.crop_protection_enabled:
                    known_canopy = gates.crop_canopy_mm(seed.current_radius_mm)
                else:
                    known_canopy = min(60.0, max(20.0, seed.current_radius_mm * 0.3))
                exclusion_radius = (known_canopy + gates.exclusion_margin_mm) * params.mean_ppm
                cv2.circle(
                    exclusion,
                    (round(seed.center_px[0]), round(seed.center_px[1])),
                    max(1, round(exclusion_radius)),
                    255,
                    -1,
                )
            if weed_settings.crop_protection_enabled:
                prior_margin_px = max(1, round(gates.exclusion_margin_mm * params.mean_ppm))
                for prior in previous_masks.values():
                    distance_from_prior = cv2.distanceTransform(
                        (prior == 0).astype(np.uint8), cv2.DIST_L2, 5
                    )
                    exclusion[distance_from_prior <= prior_margin_px] = 255
            group_gap_px = WEED_GROUP_MAX_GAP_MM * params.mean_ppm

            # Protect a complete vegetation cluster when any part of it belongs
            # to a crop or crosses a known crop's exclusion area, so the
            # circular exclusion cannot cut through an outer leaf and leave its
            # tip behind as a spurious weed.
            #
            # The protection is bounded by distance from the crop itself. Group
            # membership alone chains: island A joins B because they are within
            # the grouping gap, B joins C, and a single weed touching the crop
            # zone used to protect every weed transitively reachable from it.
            vegetation = candidate_vegetation.astype(np.uint8) * 255
            _, vegetation_groups = _nearby_component_labels(vegetation, group_gap_px)
            crop_anchor = candidate_vegetation & (claimed | (exclusion > 0))
            crop_group_ids = np.unique(vegetation_groups[crop_anchor])
            crop_group_ids = crop_group_ids[crop_group_ids > 0]
            if len(crop_group_ids):
                distance_from_crop = cv2.distanceTransform(
                    (~crop_anchor).astype(np.uint8), cv2.DIST_L2, 5
                )
                reach_px = max(1.0, CROP_SUPPORT_REACH_MM * params.mean_ppm)
                crop_supported = np.isin(vegetation_groups, crop_group_ids) & (
                    distance_from_crop <= reach_px
                )
            else:
                crop_supported = np.zeros_like(claimed)
            crop_protected = (exclusion > 0) | crop_supported
            unclaimed_vegetation = candidate_vegetation & ~claimed
            # Proximity to a crop is context rather than a candidate veto.  Its
            # feature vector receives the distance and overlap below; protected
            # detections can be reviewed and labelled, but jobs.py prevents them
            # from being created automatically. Crop pixels already owned from
            # the known plant centre remain excluded.
            candidate_pixels = (
                unclaimed_vegetation
                if weed_settings.visual_verifier_enabled
                else unclaimed_vegetation & ~crop_protected
            )
            candidate_mask = (candidate_pixels | boundary_verified_weed).astype(np.uint8) * 255

            # Label on a proximity-expanded support mask so the leaves and
            # stem fragments of one continuous weed produce one detection.
            count, candidate_labels = _bounded_nearby_component_labels(
                candidate_mask,
                group_gap_px,
                WEED_GROUP_MAX_SPAN_MM * params.mean_ppm,
            )
            area_scale = calibration.pixels_per_mm_x * calibration.pixels_per_mm_y
            # Counted so an operator can see whether the heuristic is starving
            # the verifier rather than having to infer it from missing weeds.
            candidate_stats.update(
                {
                    "blobs": 0,
                    "protected_scored": 0,
                    "oversized_scored": 0,
                    "size": 0,
                    "shape": 0,
                    "score": 0,
                }
            )
            for label in range(1, count):
                component = (candidate_mask > 0) & (candidate_labels == label)
                ys, xs = np.where(component)
                if not len(xs):
                    continue
                candidate_stats["blobs"] += 1
                area_mm2 = float(len(xs) / area_scale)
                points = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
                (wx, wy), _ = cv2.minEnclosingCircle(points)
                protection_overlap = float(np.mean(crop_protected[component]))
                if protection_overlap > 0:
                    candidate_stats["protected_scored"] += 1
                if area_mm2 > weed_settings.maximum_area_mm2:
                    candidate_stats["oversized_scored"] += 1
                if not (gates.min_area_mm2 <= area_mm2 <= gates.max_area_mm2):
                    candidate_stats["size"] += 1
                    continue
                # Distance to the nearest known crop separates an interrow weed
                # from a crop leaf that escaped the exclusion mask, and it is
                # the one feature the candidate crop cannot supply on its own.
                distance_to_plant_mm = min(
                    (
                        math.hypot(
                            (wx - seed.center_px[0]) / calibration.pixels_per_mm_x,
                            (wy - seed.center_px[1]) / calibration.pixels_per_mm_y,
                        )
                        for seed in seeds
                    ),
                    default=None,
                )
                features = extract_visual_features(
                    image,
                    component,
                    area_mm2,
                    green_hue_min=weed_settings.green_hue_min,
                    green_hue_max=weed_settings.green_hue_max,
                    strong_green_minimum_saturation=(weed_settings.strong_green_minimum_saturation),
                    strong_green_minimum_excess_green=(
                        weed_settings.strong_green_minimum_excess_green
                    ),
                    distance_to_plant_mm=distance_to_plant_mm,
                )
                # Not a learned input (legacy and current models ignore unknown
                # keys); this persists the safety context with the detection so
                # automatic creation can never act inside crop protection.
                features["crop_protection_overlap"] = protection_overlap
                features["configured_maximum_area_exceeded"] = float(
                    area_mm2 > weed_settings.maximum_area_mm2
                )
                if gates.rejects(features):
                    candidate_stats["shape"] += 1
                    continue
                # The heuristic score measures plant-ness, not weed-ness: every
                # term rises for moss, fallen leaves and crop foliage just as it
                # does for a weed. It is a usable fallback ordering before the
                # verifier is trained, never a substitute for it.
                #
                # Each term is normalised against what real foliage actually
                # scores rather than against a perfect synthetic blob. Measured
                # against photographs, the raw feature values for a genuine weed
                # sit around a third of their nominal range, so the previous
                # unnormalised sum could not exceed ~0.70 -- which was also the
                # default review threshold, leaving nothing able to pass it.
                area_score = min(1.0, math.sqrt(area_mm2 / CANDIDATE_REFERENCE_AREA_MM2))
                heuristic_confidence = min(
                    0.98,
                    0.30
                    + area_score * 0.25
                    + min(1.0, features["strong_green_fraction"] / 0.5) * 0.25
                    + min(1.0, features["solidity"] / 0.5) * 0.12
                    + min(1.0, features["circularity"] / 0.25) * 0.08,
                )
                verifier_confidence = (
                    self.weed_verifier.predict(features) if verifier_scoring else None
                )
                if verifier_authoritative and verifier_confidence is not None:
                    # A trained verifier is the score. Blending it with the
                    # heuristic used to compress its calibrated range into
                    # roughly [0.25, 0.95] and drag every rejection upwards,
                    # which made the verifier threshold the only real gate
                    # while the reported confidence said otherwise.
                    confidence = verifier_confidence
                    if verifier_confidence < weed_settings.visual_verifier_minimum_confidence:
                        candidate_stats["score"] += 1
                        continue
                else:
                    confidence = heuristic_confidence
                    if confidence < weed_settings.minimum_confidence:
                        candidate_stats["score"] += 1
                        continue
                component_radius_mm = float(
                    np.max(
                        np.sqrt(
                            ((xs - wx) / calibration.pixels_per_mm_x) ** 2
                            + ((ys - wy) / calibration.pixels_per_mm_y) ** 2
                        )
                    )
                )
                weed_detections.append(
                    WeedDetection(
                        detection_id=uuid4(),
                        image_id=image_id,
                        image_timestamp=image_timestamp,
                        center_px=(float(wx), float(wy)),
                        area_mm2=area_mm2,
                        radius_mm=max(weed_settings.weed_radius_mm, component_radius_mm),
                        confidence=confidence,
                        heuristic_confidence=heuristic_confidence,
                        verifier_confidence=verifier_confidence,
                        features=features,
                        crop_jpeg=(
                            encode_candidate_crop(image, component)
                            if weed_settings.candidate_crop_storage_enabled
                            else None
                        ),
                    )
                )
                # Enclose the full grouped weed rather than drawing an
                # equivalent-area circle around only its densest leaf.
                weed_radius_px = max(
                    10,
                    round(np.max(np.sqrt((xs - wx) ** 2 + (ys - wy) ** 2)) * 1.12),
                )
                cv2.circle(
                    overlay,
                    (round(wx), round(wy)),
                    weed_radius_px,
                    (0, 0, 255),
                    3,
                )
                cv2.circle(
                    weed_review,
                    (round(wx), round(wy)),
                    weed_radius_px,
                    (0, 0, 255),
                    3,
                )
        _draw_legend(overlay)
        ok_mask, encoded_mask = cv2.imencode(".png", mask)
        ok_ownership, encoded_ownership = cv2.imencode(".png", ownership.astype(np.uint16))
        ok_overlay, encoded_overlay = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        ok_weed_review, encoded_weed_review = (
            cv2.imencode(".jpg", weed_review, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if weed_review is not None and weed_detections
            else (False, None)
        )
        # Release the large working arrays before returning (Part 7).
        del image, mask, overlay, weed_review, labels, ownership, ambiguous
        return AnalysisResult(
            measurements=measurements,
            mask=encoded_mask.tobytes() if ok_mask else None,
            ownership_mask=encoded_ownership.tobytes() if ok_ownership else None,
            overlay_jpeg=encoded_overlay.tobytes() if ok_overlay else None,
            weed_review_jpeg=encoded_weed_review.tobytes() if ok_weed_review else None,
            skipped=skipped,
            weeds=weed_detections,
            weed_candidate_stats=candidate_stats,
            boundary_verifier_stats=boundary_stats,
        )


# FarmBot Web App renders an unmodified upload with
# ``rotate(-CAMERA_CALIBRATION_total_rotation_angle)``. Our pixel transforms
# operate on that same unrotated upload, so copied rotation values use the
# Web App's negative display direction.
ROTATION_SIGN = -1.0


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
    # FarmBot places the optical centre at the recorded gantry coordinate plus
    # CAMERA_OFFSET_X/Y. Keep copied offset signs unchanged.
    dx = plant_x - image_x - calibration.offset_x_mm
    dy = plant_y - image_y - calibration.offset_y_mm
    origin = OriginLocation(calibration.origin_location)
    # Pixel-space offset from centre, with the origin reflection applied before
    # rotation (in the garden->pixel direction).
    vx = origin.sign_x * dx * calibration.pixels_per_mm_x
    vy = origin.sign_y * dy * calibration.pixels_per_mm_y
    # The source upload is displayed by FarmBot at -rotation; mapping a garden
    # delta back into that source image applies the inverse (+rotation).
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
        image_x + dx + calibration.offset_x_mm,
        image_y + dy + calibration.offset_y_mm,
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
