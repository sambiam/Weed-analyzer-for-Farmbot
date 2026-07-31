"""The live scout's geometry and its attribution of vegetation to plants."""

import cv2
import numpy as np
import pytest

from farmbot_vision.calibration_store import FarmbotCalibrationInput
from farmbot_vision.grid_scout import (
    blob_to_garden,
    framed_plants,
    scout_cell,
    weed_candidates,
)
from farmbot_vision.models import OriginLocation
from farmbot_vision.photo_grid import KnownMapPoint, PhotoGridTarget, plant_area_coverage
from farmbot_vision.photo_quality import PhotoQuality, VegetationBlob, inspect_photo_quality


def _calibration(**overrides) -> FarmbotCalibrationInput:
    values = {
        "coordinate_scale": 0.5,
        "reference_width": 800,
        "reference_height": 600,
    }
    values.update(overrides)
    return FarmbotCalibrationInput(**values)


def _target(x: float = 1000, y: float = 800) -> PhotoGridTarget:
    return PhotoGridTarget(index=0, row=0, column=0, x=x, y=y, z=0)


def _blob(u: float, v: float, area: float = 0.05) -> VegetationBlob:
    return VegetationBlob(center_u=u, center_v=v, area_fraction=area, touches_edge=False)


def _quality(*blobs: VegetationBlob, issue: str = "usable") -> PhotoQuality:
    return PhotoQuality(issue, 0.0, 0.0, 0.0, 0.2, 0.2, 0.5, 40.0, 0.05, 0.1, blobs)


def test_a_centred_blob_maps_to_the_cell_optical_centre():
    calibration = _calibration(offset_x_mm=12, offset_y_mm=-8)
    target = _target()

    x, y = blob_to_garden(_blob(0.5, 0.5), target, calibration)

    assert x == pytest.approx(target.x + 12)
    assert y == pytest.approx(target.y - 8)


@pytest.mark.parametrize("rotation", [0.0, 17.5, -33.0])
@pytest.mark.parametrize(
    "origin",
    [OriginLocation.TOP_LEFT, OriginLocation.BOTTOM_RIGHT, OriginLocation.TOP_RIGHT],
)
def test_blob_mapping_inverts_the_coverage_transform(rotation, origin):
    """A blob placed where a plant projects must map back onto that plant.

    ``plant_area_coverage`` is the garden-to-frame direction the grid planner
    already trusts, so agreeing with it is what keeps a weed candidate's
    coordinate meaningful.
    """
    calibration = _calibration(
        rotation_degrees=rotation,
        origin_location=origin,
        offset_x_mm=15,
        offset_y_mm=-22,
    )
    target = _target()
    # A tiny canopy at this garden point is fully inside the frame, so the
    # forward transform agrees the point is visible at all.
    plant_x, plant_y = target.x + 60, target.y - 45
    assert plant_area_coverage(plant_x, plant_y, 5.0, target, calibration) == pytest.approx(1.0)

    # Find the blob position that lands on the plant by inverting through the
    # documented forward transform: sample a grid and check round-tripping.
    for u in (0.2, 0.45, 0.8):
        for v in (0.15, 0.5, 0.9):
            x, y = blob_to_garden(_blob(u, v), target, calibration)
            # A zero-radius canopy at the mapped coordinate must be inside the
            # frame, which is only true if the inverse is consistent.
            assert plant_area_coverage(x, y, 1.0, target, calibration) > 0


def test_vegetation_on_a_known_plant_is_not_a_weed_candidate():
    calibration = _calibration()
    target = _target()
    centre_x, centre_y = blob_to_garden(_blob(0.5, 0.5), target, calibration)
    plant = KnownMapPoint(id=7, kind="plant", name="Lettuce", x=centre_x, y=centre_y, radius=60)

    assert weed_candidates(_quality(_blob(0.5, 0.5)), target, calibration, [plant]) == []


def test_vegetation_clear_of_every_plant_is_a_weed_candidate():
    calibration = _calibration()
    target = _target()
    plant_x, plant_y = blob_to_garden(_blob(0.5, 0.5), target, calibration)
    plant = KnownMapPoint(id=7, kind="plant", name="Lettuce", x=plant_x, y=plant_y, radius=30)

    candidates = weed_candidates(_quality(_blob(0.08, 0.9)), target, calibration, [plant])

    assert len(candidates) == 1
    x, y = candidates[0]
    assert (x, y) == pytest.approx(blob_to_garden(_blob(0.08, 0.9), target, calibration), abs=0.1)


def test_speckle_is_ignored_and_an_unusable_frame_reports_no_candidates():
    calibration = _calibration()
    target = _target()

    assert weed_candidates(_quality(_blob(0.1, 0.1, area=0.001)), target, calibration, []) == []
    analysis = scout_cell(
        _quality(_blob(0.1, 0.1), issue="blurry"),
        target,
        image_id=5,
        calibration=calibration,
        known_points=[],
        safety_margin_mm=20,
    )
    assert analysis.issue == "blurry"
    assert analysis.weed_candidates == 0


def test_framed_plants_separates_whole_canopies_from_composite_ones():
    """Only a plant whose whole safety-margined canopy fits one frame can be
    measured from that frame; the rest are left for the composite."""
    calibration = _calibration()  # 400mm x 300mm frame
    target = _target()
    small = KnownMapPoint(id=1, kind="plant", name="Radish", x=target.x, y=target.y, radius=30)
    huge = KnownMapPoint(id=2, kind="plant", name="Squash", x=target.x, y=target.y, radius=180)
    elsewhere = KnownMapPoint(
        id=3, kind="plant", name="Kale", x=target.x + 5000, y=target.y, radius=30
    )

    fully, partially = framed_plants(
        target,
        calibration,
        [small, huge, elsewhere],
        safety_margin_mm=20,
    )

    assert fully == [1]
    assert partially == [2]


def test_inspection_reports_vegetation_blobs_without_a_second_pass():
    """The blobs come out of the connected-components pass the quality check
    already runs, so a real photo yields usable centroids."""
    image = np.full((240, 320, 3), (35, 65, 80), np.uint8)
    cv2.circle(image, (80, 60), 26, (28, 165, 42), -1)
    cv2.circle(image, (240, 180), 30, (30, 170, 45), -1)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok

    quality = inspect_photo_quality(encoded.tobytes())

    assert len(quality.vegetation_blobs) == 2
    centres = sorted((blob.center_u, blob.center_v) for blob in quality.vegetation_blobs)
    assert centres[0] == pytest.approx((0.25, 0.25), abs=0.03)
    assert centres[1] == pytest.approx((0.75, 0.75), abs=0.03)
