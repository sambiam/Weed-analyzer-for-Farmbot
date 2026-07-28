from datetime import UTC, datetime, timedelta
from uuid import uuid4

import cv2
import numpy as np

from farmbot_vision.jobs import build_plant_composite
from farmbot_vision.models import Decision, Measurement


def _measurement(
    tmp_path,
    image_id: int,
    center_x: float,
    timestamp: datetime,
    *,
    transform_json: str = '{"pixels_per_mm_x":1,"pixels_per_mm_y":1}',
    visible_fraction: float = 1.0,
    size: tuple[int, int] = (100, 200),
) -> Measurement:
    photo = tmp_path / f"{image_id}.jpg"
    mask = tmp_path / f"{image_id}.png"
    height, width = size
    image = np.full((height, width, 3), (55 + image_id, 90, 110), np.uint8)
    ownership = np.zeros((height, width), np.uint8)
    cv2.circle(ownership, (round(center_x), height // 2), 18, 255, -1)
    cv2.imwrite(str(photo), image)
    cv2.imwrite(str(mask), ownership)
    return Measurement(
        measurement_id=uuid4(),
        config_entry_id="bot",
        plant_id=7,
        crop_slug="broccoli",
        image_id=image_id,
        image_timestamp=timestamp,
        current_radius_mm=30,
        typical_canopy_radius_mm=35,
        maximum_accepted_canopy_radius_mm=40,
        recommended_protection_radius_mm=60,
        confidence=0.9,
        decision=Decision.RECOMMENDED,
        reason="test",
        transform_json=transform_json,
        algorithm_version="test",
        plant_center_px=(center_x, height // 2),
        source_image_path=str(photo),
        mask_path=str(mask),
        visible_fraction=visible_fraction,
    )


def test_composite_stitches_overlapping_frames_and_draws_bold_radii(tmp_path):
    now = datetime(2026, 7, 26, tzinfo=UTC)
    output = tmp_path / "composite.jpg"
    overlay_output = tmp_path / "composite-overlay.jpg"
    measurements = [
        _measurement(tmp_path, 1, 150, now),
        _measurement(tmp_path, 2, 50, now + timedelta(minutes=1)),
    ]

    assert build_plant_composite(measurements, output, overlay_output)

    composite = cv2.imread(str(output))
    overlay = cv2.imread(str(overlay_output))
    assert composite is not None
    assert overlay is not None
    assert composite.shape[1] > 200
    # The clean and mask-tinted views share identical geometry but differ
    # where the stitched ownership masks identify this plant.
    assert composite.shape == overlay.shape
    assert np.mean(cv2.absdiff(composite, overlay)) > 1
    # Planned circle is red; original circle is cyan.
    assert (
        np.count_nonzero(
            (composite[:, :, 2] > 180) & (composite[:, :, 1] < 100) & (composite[:, :, 0] < 100)
        )
        > 20
    )


def test_composite_ignores_views_that_never_contained_the_plant(tmp_path):
    now = datetime(2026, 7, 26, tzinfo=UTC)
    output = tmp_path / "in-frame.jpg"
    visible = _measurement(tmp_path, 4, 100, now)
    # vision.py records a zero-visibility measurement for plants whose whole
    # protection circle sits outside the frame. Stitching those photos widened
    # the canvas to the entire photo grid and buried the plant.
    elsewhere = _measurement(tmp_path, 5, 4000, now + timedelta(minutes=1), visible_fraction=0)

    assert build_plant_composite([visible], output)
    alone = cv2.imread(str(output))
    assert build_plant_composite([visible, elsewhere], output)
    with_distant_photo = cv2.imread(str(output))

    assert alone is not None
    assert with_distant_photo is not None
    assert with_distant_photo.shape == alone.shape


def test_composite_crops_to_the_plant_neighbourhood(tmp_path):
    now = datetime(2026, 7, 26, tzinfo=UTC)
    output = tmp_path / "cropped.jpg"
    # A 2000x1000 mm photo at 1 px/mm: only the plant's surroundings belong in
    # the composite, not every millimetre the camera happened to see.
    wide = _measurement(tmp_path, 6, 1000, now, size=(1000, 2000))

    assert build_plant_composite([wide], output)

    composite = cv2.imread(str(output))
    assert composite is not None
    # recommended_protection_radius_mm is 60, so the focus window is +-150 mm.
    assert composite.shape[0] == 300
    assert composite.shape[1] == 300


def test_composite_applies_calibrated_rotation_and_origin(tmp_path):
    now = datetime(2026, 7, 26, tzinfo=UTC)
    output = tmp_path / "rotated.jpg"
    measurement = _measurement(
        tmp_path,
        3,
        100,
        now,
        transform_json=(
            '{"pixels_per_mm_x":1,"pixels_per_mm_y":1,'
            '"rotation_degrees":90,"origin_location":"bottom_right"}'
        ),
    )

    assert build_plant_composite([measurement], output)

    composite = cv2.imread(str(output))
    assert composite is not None
    # A 200x100 source becomes a 100x200 garden-aligned composite at 90°.
    assert composite.shape[0] > composite.shape[1]
    assert (
        np.count_nonzero(
            (composite[:, :, 0] > 150) & (composite[:, :, 1] > 150) & (composite[:, :, 2] < 120)
        )
        > 20
    )
