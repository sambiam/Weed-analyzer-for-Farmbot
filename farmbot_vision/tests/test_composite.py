from datetime import UTC, datetime, timedelta
from uuid import uuid4

import cv2
import numpy as np

from farmbot_vision.jobs import build_plant_composite
from farmbot_vision.models import Decision, Measurement


def _measurement(tmp_path, image_id: int, center_x: float, timestamp: datetime) -> Measurement:
    photo = tmp_path / f"{image_id}.jpg"
    mask = tmp_path / f"{image_id}.png"
    image = np.full((100, 200, 3), (55 + image_id, 90, 110), np.uint8)
    ownership = np.zeros((100, 200), np.uint8)
    cv2.circle(ownership, (round(center_x), 50), 18, 255, -1)
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
        transform_json='{"pixels_per_mm_x":1,"pixels_per_mm_y":1}',
        algorithm_version="test",
        plant_center_px=(center_x, 50),
        source_image_path=str(photo),
        mask_path=str(mask),
    )


def test_composite_stitches_overlapping_frames_and_draws_bold_radii(tmp_path):
    now = datetime(2026, 7, 26, tzinfo=UTC)
    output = tmp_path / "composite.jpg"
    measurements = [
        _measurement(tmp_path, 1, 150, now),
        _measurement(tmp_path, 2, 50, now + timedelta(minutes=1)),
    ]

    assert build_plant_composite(measurements, output)

    composite = cv2.imread(str(output))
    assert composite is not None
    assert composite.shape[1] > 200
    # Planned circle is red; original circle is cyan.
    assert np.count_nonzero(
        (composite[:, :, 2] > 180)
        & (composite[:, :, 1] < 100)
        & (composite[:, :, 0] < 100)
    ) > 20
    assert np.count_nonzero(
        (composite[:, :, 0] > 150)
        & (composite[:, :, 1] > 150)
        & (composite[:, :, 2] < 120)
    ) > 20
