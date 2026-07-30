"""Conservative second-pass quality checks for photo-grid frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

PhotoIssue = Literal["usable", "washed_out", "leaf_obstruction"]


@dataclass(frozen=True)
class PhotoQuality:
    issue: PhotoIssue
    washed_out_score: float
    leaf_obstruction_score: float
    vegetation_fraction: float
    clear_plant_fraction: float
    sharpness: float


def _decoded(jpeg: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 80:
        return None
    height, width = image.shape[:2]
    scale = min(1.0, 640 / max(width, height))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def inspect_photo_quality(jpeg: bytes) -> PhotoQuality:
    """Classify exposure failure and a close, border-blocking plant leaf.

    The obstruction rule deliberately requires a thick vegetation component
    connected to at least two frame edges. Ordinary small leaves, stems, and a
    canopy merely near one edge therefore remain usable.
    """

    image = _decoded(jpeg)
    if image is None:
        return PhotoQuality("usable", 0.0, 0.0, 0.0, 0.0, 0.0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright_neutral = float(np.mean((value >= 205) & (saturation <= 85)))
    median_value = float(np.median(value)) / 255.0
    lower_value = float(np.percentile(value, 10)) / 255.0
    washed_score = float(
        np.clip(
            0.50 * bright_neutral + 0.30 * median_value + 0.20 * lower_value,
            0.0,
            1.0,
        )
    )

    b, g, r = cv2.split(image.astype(np.int16))
    excess_green = 2 * g - r - b
    green = (
        (hsv[:, :, 0] >= 25)
        & (hsv[:, :, 0] <= 105)
        & (saturation >= 32)
        & (value >= 25)
        & (excess_green >= 9)
    ).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    vegetation_fraction = float(np.mean(green > 0))

    height, width = green.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    largest_obstruction = 0.0
    clear_pixels = 0
    minimum_thickness = min(height, width) * 0.075
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        selected = labels == label
        touches = sum(
            (
                x == 0,
                y == 0,
                x + component_width >= width,
                y + component_height >= height,
            )
        )
        distance = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, 3)
        thick = float(distance.max()) >= minimum_thickness
        fraction = float(area) / float(height * width)
        if touches >= 2 and thick:
            largest_obstruction = max(largest_obstruction, fraction)
        else:
            clear_pixels += int(area)

    # A close leaf must be both a substantial fraction of the photograph and
    # a substantial fraction of all detected vegetation.
    dominance = largest_obstruction / max(0.01, vegetation_fraction)
    leaf_score = float(
        np.clip(
            0.65 * (largest_obstruction / 0.40) + 0.35 * dominance,
            0.0,
            1.0,
        )
    )
    clear_fraction = float(clear_pixels) / float(height * width)
    sharpness = float(
        np.clip(
            np.log1p(float(cv2.Laplacian(grey, cv2.CV_32F).var())) / np.log(101),
            0.0,
            1.0,
        )
    )

    washed_out = (
        bright_neutral >= 0.68
        and median_value >= 0.86
        and lower_value >= 0.58
        and vegetation_fraction < 0.10
    )
    leaf_obstruction = (
        largest_obstruction >= 0.24 and vegetation_fraction >= 0.30 and dominance >= 0.62
    )
    issue: PhotoIssue = (
        "washed_out" if washed_out else "leaf_obstruction" if leaf_obstruction else "usable"
    )
    return PhotoQuality(
        issue,
        washed_score,
        leaf_score,
        vegetation_fraction,
        clear_fraction,
        sharpness,
    )


def best_unobscured_photo(
    candidates: list[tuple[int, PhotoQuality]],
) -> int | None:
    """Return the image with the most clear plant and least close-leaf cover."""

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[1].issue != "washed_out",
            item[1].clear_plant_fraction,
            -item[1].leaf_obstruction_score,
            item[1].vegetation_fraction,
            item[1].sharpness,
        ),
    )[0]
