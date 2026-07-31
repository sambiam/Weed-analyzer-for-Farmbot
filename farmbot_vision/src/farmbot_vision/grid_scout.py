"""Lightweight, in-flight assessment of photo-grid frames.

The scout runs *while* the grid is still being captured, in the dead time the
worker already spends waiting for FarmBot to drive to the next coordinate. It
deliberately does only work that is either free or already owed:

* **Photo quality** (blur, washed-out exposure, close-leaf obstruction) is the
  single :func:`~farmbot_vision.photo_quality.inspect_photo_quality` call the
  post-grid quality pass would have made anyway, moved earlier. Its result is
  cached and reused, so the grid does not inspect anything twice.
* **Weed candidates** are the vegetation components that inspection *already*
  found, mapped into garden millimetres and filtered against the plants FarmBot
  already knows about. No second decode, no segmentation of its own.
* **Plant framing** is pure calibration geometry with no pixels involved.

What the scout explicitly does not do is measure plant radius or confirm weeds.
A plant whose canopy spans several cells has to be measured from a composite,
and a weed is only a weed once the real detector and its verifier have run.
Both stay in the post-grid analysis pipeline, unchanged.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from .calibration_store import FarmbotCalibrationInput
from .photo_grid import KnownMapPoint, PhotoGridCellAnalysis, PhotoGridTarget, plant_area_coverage
from .photo_quality import PhotoQuality, VegetationBlob

# A plant hides vegetation this far beyond its recorded canopy before a blob
# counts as something else growing. FarmBot canopy radii are nominal, and a
# candidate is only ever a hint, so this stays generous: over-attributing a
# blob to a plant merely means the real detector reports the weed later, while
# under-attributing it puts a false marker on the user's grid.
WEED_CANDIDATE_CLEARANCE_MM = 45.0
# A blob must be at least this fraction of the frame to be worth flagging as a
# possible weed. Below it, single leaves and moss speckle dominate.
WEED_CANDIDATE_MINIMUM_AREA_FRACTION = 0.004
# Coverage at which a plant's whole safety-margined canopy is inside one frame.
FULLY_FRAMED_COVERAGE = 0.999


def blob_to_garden(
    blob: VegetationBlob,
    target: PhotoGridTarget,
    calibration: FarmbotCalibrationInput,
) -> tuple[float, float]:
    """Map a blob's frame-relative centroid to a garden coordinate.

    This is the exact algebraic inverse of the garden-to-frame transform
    :func:`~farmbot_vision.photo_grid.plant_area_coverage` uses, so a blob
    placed at a plant's projected position maps back onto that plant.
    """
    half_width = calibration.coordinate_scale * calibration.reference_width / 2
    half_height = calibration.coordinate_scale * calibration.reference_height / 2
    local_x = (blob.center_u - 0.5) * 2 * half_width
    local_y = (blob.center_v - 0.5) * 2 * half_height
    theta = math.radians(-calibration.rotation_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx = cos_t * local_x - sin_t * local_y
    dy = sin_t * local_x + cos_t * local_y
    return (
        target.x + calibration.offset_x_mm + dx * calibration.origin_location.sign_x,
        target.y + calibration.offset_y_mm + dy * calibration.origin_location.sign_y,
    )


def weed_candidates(
    quality: PhotoQuality,
    target: PhotoGridTarget,
    calibration: FarmbotCalibrationInput,
    known_points: list[KnownMapPoint],
) -> list[tuple[float, float]]:
    """Return garden coordinates of vegetation no known plant accounts for."""

    plants = [point for point in known_points if point.kind == "plant"]
    candidates: list[tuple[float, float]] = []
    for blob in quality.vegetation_blobs:
        if blob.area_fraction < WEED_CANDIDATE_MINIMUM_AREA_FRACTION:
            continue
        x, y = blob_to_garden(blob, target, calibration)
        if any(
            math.dist((x, y), (plant.x, plant.y)) <= plant.radius + WEED_CANDIDATE_CLEARANCE_MM
            for plant in plants
        ):
            continue
        candidates.append((round(x, 1), round(y, 1)))
    return candidates


def framed_plants(
    target: PhotoGridTarget,
    calibration: FarmbotCalibrationInput,
    known_points: list[KnownMapPoint],
    *,
    safety_margin_mm: float,
) -> tuple[list[int], list[int]]:
    """Split known plants into those wholly and those partly inside this cell.

    A wholly framed plant can be measured from this single photo. A partly
    framed one needs the composite the post-grid pipeline builds, which is why
    the scout records the distinction rather than attempting a radius here.
    """

    fully: list[int] = []
    partially: list[int] = []
    for point in known_points:
        if point.kind != "plant":
            continue
        coverage = plant_area_coverage(
            point.x,
            point.y,
            max(1.0, point.radius + safety_margin_mm),
            target,
            calibration,
        )
        if coverage >= FULLY_FRAMED_COVERAGE:
            fully.append(point.id)
        elif coverage > 0:
            partially.append(point.id)
    return fully, partially


def scout_cell(
    quality: PhotoQuality,
    target: PhotoGridTarget,
    image_id: int,
    calibration: FarmbotCalibrationInput,
    known_points: list[KnownMapPoint],
    *,
    safety_margin_mm: float,
) -> PhotoGridCellAnalysis:
    """Summarise one freshly captured cell from one quality inspection."""

    # An unusable frame's vegetation is not trustworthy evidence of anything:
    # a washed-out or defocused photo under-segments, and a close leaf hides
    # whatever is behind it. Report the issue and leave the rest for the retake.
    candidates = (
        weed_candidates(quality, target, calibration, known_points)
        if quality.issue == "usable"
        else []
    )
    fully, partially = framed_plants(
        target,
        calibration,
        known_points,
        safety_margin_mm=safety_margin_mm,
    )
    return PhotoGridCellAnalysis(
        target_index=target.index,
        image_id=image_id,
        issue=quality.issue,
        blur_score=round(quality.blur_score, 3),
        washed_out_score=round(quality.washed_out_score, 3),
        leaf_obstruction_score=round(quality.leaf_obstruction_score, 3),
        vegetation_fraction=round(quality.vegetation_fraction, 4),
        weed_candidates=len(candidates),
        weed_candidate_points=candidates,
        fully_framed_plant_ids=fully,
        partially_framed_plant_ids=partially,
        analysed_at=datetime.now(UTC),
    )
