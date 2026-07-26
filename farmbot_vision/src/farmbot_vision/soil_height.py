"""Calibrated virtual-stereo soil-height measurement.

The engine intentionally has no Home Assistant, database, or FastAPI
dependencies. Capture orchestration can therefore fail independently from
image quality, and the numerical gates are directly testable with synthetic
frames.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import cv2
import numpy as np

from .models import SoilStereoCalibration
from .vision import ScaleParams, vegetation_mask

ALGORITHM_VERSION = "soil-stereo-v1"
MIN_VALID_COVERAGE = 0.15
MIN_PLANE_SUPPORT = 0.50
MAX_LR_ERROR_PX = 1.5
MAX_PLANE_MAD_PX = 2.0
MAX_PAIR_SPREAD_MM = 8.0
MAX_UNCERTAINTY_MM = 10.0


class SoilHeightError(RuntimeError):
    """Expected, user-actionable failure of calibration or measurement."""


@dataclass(slots=True)
class SoilFrame:
    image_id: int
    jpeg: bytes
    x: float
    y: float
    z: float
    lateral_offset_mm: float
    z_offset_mm: float
    processed_width: int
    processed_height: int
    source_width: int
    source_height: int


@dataclass(slots=True)
class PairEstimate:
    normalized_disparity: float
    disparity_px: float
    baseline_mm: float
    valid_coverage: float
    plane_support: float
    lr_error_px: float
    plane_mad_px: float
    disparity_map: np.ndarray
    valid_mask: np.ndarray
    plane_mask: np.ndarray
    rectification_overlay: np.ndarray

    @property
    def passes_geometry(self) -> bool:
        return (
            self.valid_coverage >= MIN_VALID_COVERAGE
            and self.plane_support >= MIN_PLANE_SUPPORT
            and self.lr_error_px <= MAX_LR_ERROR_PX
            and self.plane_mad_px <= MAX_PLANE_MAD_PX
        )


@dataclass(slots=True)
class SoilAnalysis:
    measurement_id: UUID = field(default_factory=uuid4)
    valid: bool = False
    proposed_z_mm: float | None = None
    confidence: float = 0
    uncertainty_mm: float | None = None
    reason: str = ""
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    artifacts: dict[str, bytes] = field(default_factory=dict)


def camera_signature(frames: list[SoilFrame], declared_signature: str = "") -> str:
    """Bind calibration to exact source/processed geometry and declared camera data."""
    if not frames:
        raise SoilHeightError("no soil images were supplied")
    geometries = {
        (
            frame.processed_width,
            frame.processed_height,
            frame.source_width,
            frame.source_height,
        )
        for frame in frames
    }
    if len(geometries) != 1:
        raise SoilHeightError("soil image geometry changed within the capture")
    processed_width, processed_height, source_width, source_height = geometries.pop()
    payload = {
        "processed": [processed_width, processed_height],
        "source": [source_width, source_height],
        "declared": declared_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _decode(frame: SoilFrame) -> np.ndarray:
    data = np.frombuffer(frame.jpeg, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SoilHeightError(f"image {frame.image_id} is not a decodable JPEG")
    if image.shape[1] != frame.processed_width or image.shape[0] != frame.processed_height:
        raise SoilHeightError(f"image {frame.image_id} dimensions changed")
    return image


def _enhance_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _green_mask(image: np.ndarray) -> np.ndarray:
    """Reuse the canopy/weed vegetation segmentation, then add a safety margin."""
    height, width = image.shape[:2]
    mask = vegetation_mask(image, ScaleParams.build(width, height, calibration=None))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=1) > 0


def _rectify_pair(
    left_color: np.ndarray, right_color: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate both views so robust feature flow follows the horizontal epipolar axis."""
    left = _enhance_gray(left_color)
    right = _enhance_gray(right_color)
    points = cv2.goodFeaturesToTrack(
        left, maxCorners=800, qualityLevel=0.01, minDistance=7, blockSize=7
    )
    if points is None or len(points) < 20:
        raise SoilHeightError("not enough image texture for stereo rectification")
    tracked, status, _errors = cv2.calcOpticalFlowPyrLK(
        left, right, points, None, winSize=(31, 31), maxLevel=4
    )
    if tracked is None or status is None:
        raise SoilHeightError("optical-flow rectification failed")
    good_a = points.reshape(-1, 2)[status.reshape(-1) > 0]
    good_b = tracked.reshape(-1, 2)[status.reshape(-1) > 0]
    if len(good_a) < 20:
        raise SoilHeightError("not enough consistent features for rectification")
    transform, inliers = cv2.estimateAffinePartial2D(
        good_a, good_b, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if transform is None or inliers is None or int(inliers.sum()) < 15:
        raise SoilHeightError("camera motion could not be rectified")
    deltas = good_b[inliers.reshape(-1) > 0] - good_a[inliers.reshape(-1) > 0]
    dx, dy = np.median(deltas, axis=0)
    if math.hypot(float(dx), float(dy)) < 1:
        raise SoilHeightError("camera movement is too small to measure disparity")
    # OpenCV's positive-disparity matcher expects a feature to appear leftward
    # in the second image (x_left - x_right > 0).
    if dx > 0:
        left_color, right_color = right_color, left_color
        left, right = right, left
        dx, dy = -dx, -dy
    angle = math.degrees(math.atan2(float(dy), float(dx)))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    height, width = left.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    left_rect = cv2.warpAffine(left, matrix, (width, height), flags=cv2.INTER_LINEAR)
    right_rect = cv2.warpAffine(right, matrix, (width, height), flags=cv2.INTER_LINEAR)
    left_color_rect = cv2.warpAffine(left_color, matrix, (width, height), flags=cv2.INTER_LINEAR)
    right_color_rect = cv2.warpAffine(right_color, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return left_rect, right_rect, left_color_rect, right_color_rect


def _stereo_maps(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = left.shape[1]
    num = min(256, max(16, (width // 4 // 16) * 16))
    block = 5
    common = dict(
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    forward = (
        cv2.StereoSGBM_create(minDisparity=0, numDisparities=num, **common)
        .compute(left, right)
        .astype(np.float32)
        / 16.0
    )
    reverse = (
        cv2.StereoSGBM_create(minDisparity=-num, numDisparities=num, **common)
        .compute(right, left)
        .astype(np.float32)
        / 16.0
    )
    return forward, reverse


def _fit_plane(disparity: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    ys, xs = np.nonzero(valid)
    if len(xs) < 500:
        raise SoilHeightError("not enough valid stereo pixels")
    rng = np.random.default_rng(2407)
    if len(xs) > 20_000:
        chosen = rng.choice(len(xs), 20_000, replace=False)
        xs, ys = xs[chosen], ys[chosen]
    values = disparity[ys, xs]
    coords = np.column_stack((xs, ys, np.ones_like(xs)))
    best = None
    best_count = 0
    for _ in range(100):
        sample = rng.choice(len(xs), 3, replace=False)
        try:
            plane = np.linalg.solve(coords[sample], values[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(coords @ plane - values)
        count = int(np.count_nonzero(residuals <= 2.0))
        if count > best_count:
            best, best_count = residuals <= 2.0, count
    if best is None or best_count < 100:
        raise SoilHeightError("dominant soil plane was not found")
    plane, *_ = np.linalg.lstsq(coords[best], values[best], rcond=None)
    residuals = np.abs(coords @ plane - values)
    inliers = residuals <= 2.0
    support = float(np.count_nonzero(inliers) / len(inliers))
    mad = float(np.median(residuals[inliers]))
    height, width = disparity.shape
    center_disparity = float(plane @ np.array([width / 2, height / 2, 1.0]))
    plane_mask = np.zeros_like(valid)
    plane_mask[ys[inliers], xs[inliers]] = True
    return plane_mask, support, mad, center_disparity


def estimate_pair(first: SoilFrame, second: SoilFrame) -> PairEstimate:
    baseline = abs(second.lateral_offset_mm - first.lateral_offset_mm)
    if baseline < 4.99:
        raise SoilHeightError("stereo baseline is too small")
    first_color, second_color = _decode(first), _decode(second)
    left, right, left_color, right_color = _rectify_pair(first_color, second_color)
    disparity, reverse = _stereo_maps(left, right)
    height, width = disparity.shape
    yy, xx = np.indices((height, width))
    xr = np.rint(xx - disparity).astype(np.int32)
    in_bounds = (xr >= 0) & (xr < width)
    sampled_reverse = np.full_like(disparity, np.nan)
    sampled_reverse[in_bounds] = reverse[yy[in_bounds], xr[in_bounds]]
    lr_error = np.abs(disparity + sampled_reverse)
    border = max(8, round(min(width, height) * 0.05))
    valid = (
        (disparity > 0.5)
        & np.isfinite(sampled_reverse)
        & (lr_error <= MAX_LR_ERROR_PX)
        & ~_green_mask(left_color)
        & ~_green_mask(right_color)
    )
    valid[:border] = False
    valid[-border:] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    coverage = float(np.count_nonzero(valid) / valid.size)
    plane_mask, support, mad, center = _fit_plane(disparity, valid)
    median_lr = float(np.median(lr_error[valid])) if np.any(valid) else math.inf
    rectification_overlay = cv2.addWeighted(left_color, 0.5, right_color, 0.5, 0)
    for row in range(40, height, 80):
        cv2.line(rectification_overlay, (0, row), (width - 1, row), (0, 220, 255), 1)
    return PairEstimate(
        normalized_disparity=center / baseline,
        disparity_px=center,
        baseline_mm=baseline,
        valid_coverage=coverage,
        plane_support=support,
        lr_error_px=median_lr,
        plane_mad_px=mad,
        disparity_map=disparity,
        valid_mask=valid,
        plane_mask=plane_mask,
        rectification_overlay=rectification_overlay,
    )


def estimate_triplet(frames: list[SoilFrame]) -> list[PairEstimate]:
    if len(frames) != 3:
        raise SoilHeightError("each soil stereo set must contain exactly three images")
    ordered = sorted(frames, key=lambda frame: frame.lateral_offset_mm)
    pairs = ((ordered[0], ordered[1]), (ordered[1], ordered[2]), (ordered[0], ordered[2]))
    estimates, errors = [], []
    for first, second in pairs:
        try:
            estimates.append(estimate_pair(first, second))
        except SoilHeightError as err:
            errors.append(str(err))
    if len(estimates) < 2:
        raise SoilHeightError(errors[0] if errors else "fewer than two stereo pairs succeeded")
    return estimates


def fit_calibration(
    *,
    config_entry_id: str,
    point_id: int,
    capture_z: float,
    baseline_mm: float,
    reference_distance_mm: float,
    z_direction: int,
    frames: list[SoilFrame],
    declared_camera_signature: str = "",
) -> SoilStereoCalibration:
    groups: dict[float, list[SoilFrame]] = {}
    for frame in frames:
        groups.setdefault(frame.z_offset_mm, []).append(frame)
    if len(groups) < 3:
        raise SoilHeightError("calibration requires three distinct Z levels")
    samples = []
    for z_offset, group in sorted(groups.items()):
        distance = reference_distance_mm - z_offset
        if distance <= 0:
            raise SoilHeightError("calibration movement reaches or passes the soil")
        estimates = [item for item in estimate_triplet(group) if item.passes_geometry]
        if len(estimates) < 2:
            raise SoilHeightError(f"calibration imagery failed quality gates at {z_offset:g} mm")
        normalized = float(np.median([item.normalized_disparity for item in estimates]))
        samples.append((normalized, 1.0 / distance, distance))
    disparities = np.array([sample[0] for sample in samples], dtype=np.float64)
    inverse_distances = np.array([sample[1] for sample in samples], dtype=np.float64)
    if np.any(np.diff(disparities) <= 0):
        raise SoilHeightError("calibration disparity is not monotonic with camera distance")
    slope, intercept = np.polyfit(disparities, inverse_distances, 1)
    if not math.isfinite(slope) or slope <= 0:
        raise SoilHeightError("calibration produced an invalid inverse-depth slope")
    predicted = 1.0 / (slope * disparities + intercept)
    residual = float(np.max(np.abs(predicted - np.array([sample[2] for sample in samples]))))
    if residual > 5:
        raise SoilHeightError(f"calibration residual {residual:.1f} mm exceeds 5 mm")
    first = frames[0]
    return SoilStereoCalibration(
        config_entry_id=config_entry_id,
        point_id=point_id,
        capture_z=capture_z,
        baseline_mm=baseline_mm,
        reference_distance_mm=reference_distance_mm,
        z_direction=-1 if z_direction < 0 else 1,
        inverse_depth_slope=float(slope),
        inverse_depth_intercept=float(intercept),
        residual_mm=residual,
        processed_width=first.processed_width,
        processed_height=first.processed_height,
        source_width=first.source_width,
        source_height=first.source_height,
        source_image_ids=[frame.image_id for frame in frames],
        camera_signature=camera_signature(frames, declared_camera_signature),
    )


def _encode(extension: str, image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(extension, image)
    return data.tobytes() if ok else b""


def _triptych_artifact(frames: list[SoilFrame]) -> bytes:
    colors = [_decode(frame) for frame in sorted(frames, key=lambda item: item.lateral_offset_mm)]
    height = min(image.shape[0] for image in colors)
    triptych = np.hstack(
        [
            cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))
            for image in colors
        ]
    )
    return _encode(".jpg", triptych)


def _pair_artifacts(estimates: list[PairEstimate]) -> dict[str, bytes]:
    if not estimates:
        return {}
    best = max(estimates, key=lambda item: item.valid_coverage * item.plane_support)
    disparity = cv2.normalize(best.disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparity_color = cv2.applyColorMap(disparity, cv2.COLORMAP_TURBO)
    plane_overlay = disparity_color.copy()
    plane_overlay[best.plane_mask] = (
        0.35 * plane_overlay[best.plane_mask] + 0.65 * np.array([0, 220, 0])
    ).astype(np.uint8)
    return {
        "rectification-overlay.jpg": _encode(".jpg", best.rectification_overlay),
        "disparity.jpg": _encode(".jpg", disparity_color),
        "valid-mask.png": _encode(".png", best.valid_mask.astype(np.uint8) * 255),
        "soil-plane.jpg": _encode(".jpg", plane_overlay),
    }


def analyse_soil_height(
    frames: list[SoilFrame],
    calibration: SoilStereoCalibration,
    *,
    declared_camera_signature: str = "",
) -> SoilAnalysis:
    result = SoilAnalysis()
    try:
        if len(frames) != 3:
            raise SoilHeightError("measurement requires exactly three images")
        # The triptych remains useful even when every stereo quality gate fails.
        result.artifacts["triptych.jpg"] = _triptych_artifact(frames)
        signature = camera_signature(frames, declared_camera_signature)
        if signature != calibration.camera_signature:
            raise SoilHeightError("camera geometry changed; recalibration is required")
        estimates = estimate_triplet(frames)
        result.artifacts.update(_pair_artifacts(estimates))
        passing = [estimate for estimate in estimates if estimate.passes_geometry]
        if len(passing) < 2:
            raise SoilHeightError("fewer than two stereo pairs passed the quality gates")
        converted = []
        for estimate in passing:
            inverse = (
                calibration.inverse_depth_slope * estimate.normalized_disparity
                + calibration.inverse_depth_intercept
            )
            if inverse <= 0:
                continue
            converted.append((1.0 / inverse, estimate))
        if len(converted) < 2:
            raise SoilHeightError("calibration could not convert two stereo pairs")
        converted.sort(key=lambda item: item[0])
        # At least two pairs must agree. A third outlier is diagnostic evidence,
        # not a reason to discard two mutually consistent independent estimates.
        first, second = min(
            zip(converted, converted[1:], strict=False),
            key=lambda pair: pair[1][0] - pair[0][0],
        )
        agreeing = [first, second]
        if len(converted) == 3 and converted[-1][0] - converted[0][0] <= MAX_PAIR_SPREAD_MM:
            agreeing = converted
        distances = [item[0] for item in agreeing]
        agreeing_estimates = [item[1] for item in agreeing]
        spread = float(max(distances) - min(distances))
        distance = float(np.median(distances))
        uncertainty = max(calibration.residual_mm, spread / 2)
        if spread > MAX_PAIR_SPREAD_MM:
            raise SoilHeightError(f"stereo pairs disagree by {spread:.1f} mm")
        if uncertainty > MAX_UNCERTAINTY_MM:
            raise SoilHeightError(f"measurement uncertainty is {uncertainty:.1f} mm")
        camera_z = float(np.median([frame.z for frame in frames]))
        proposed = round(camera_z + calibration.z_direction * distance)
        coverage = float(np.median([item.valid_coverage for item in agreeing_estimates]))
        support = float(np.median([item.plane_support for item in agreeing_estimates]))
        lr_error = float(np.median([item.lr_error_px for item in agreeing_estimates]))
        plane_mad = float(np.median([item.plane_mad_px for item in agreeing_estimates]))
        confidence = float(
            np.clip(
                0.20 * min(1.0, coverage / 0.35)
                + 0.20 * min(1.0, support / 0.8)
                + 0.15 * max(0.0, 1 - lr_error / MAX_LR_ERROR_PX)
                + 0.15 * max(0.0, 1 - plane_mad / MAX_PLANE_MAD_PX)
                + 0.15 * max(0.0, 1 - spread / MAX_PAIR_SPREAD_MM)
                + 0.15 * max(0.0, 1 - calibration.residual_mm / 5.0),
                0,
                0.99,
            )
        )
        result.valid = True
        result.proposed_z_mm = float(proposed)
        result.confidence = confidence
        result.uncertainty_mm = uncertainty
        result.reason = "Three-view stereo result passed all quality gates"
        result.metrics = {
            "valid_coverage": coverage,
            "plane_support": support,
            "lr_error_px": lr_error,
            "plane_mad_px": plane_mad,
            "pair_spread_mm": spread,
            "distance_mm": distance,
            "passing_pairs": len(passing),
            "agreeing_pairs": len(agreeing),
        }
    except SoilHeightError as err:
        result.reason = str(err)
    return result
