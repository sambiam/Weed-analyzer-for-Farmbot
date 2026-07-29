import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np

from farmbot_vision.calibration_store import FarmbotCalibrationInput
from farmbot_vision.jobs import build_plant_composite
from farmbot_vision.models import Decision, Measurement
from farmbot_vision.photo_grid import (
    PhotoGridFrame,
    PhotoGridRecord,
    PhotoGridTarget,
)


def _measurement(
    tmp_path,
    image_id: int,
    center_x: float,
    timestamp: datetime,
    *,
    center_y: float | None = None,
    transform_json: str = '{"pixels_per_mm_x":1,"pixels_per_mm_y":1}',
    visible_fraction: float = 1.0,
    boundary_sectors: list[int] | None = None,
    center_visible: bool = True,
    has_plant_evidence: bool = True,
    size: tuple[int, int] = (100, 200),
) -> Measurement:
    photo = tmp_path / f"{image_id}.jpg"
    mask = tmp_path / f"{image_id}.png"
    height, width = size
    center_y = height // 2 if center_y is None else center_y
    image = np.full((height, width, 3), (55 + image_id, 90, 110), np.uint8)
    ownership = np.zeros((height, width), np.uint8)
    cv2.circle(ownership, (round(center_x), round(center_y)), 18, 255, -1)
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
        plant_center_px=(center_x, center_y),
        source_image_path=str(photo),
        mask_path=str(mask),
        visible_fraction=visible_fraction,
        boundary_coverage=len(boundary_sectors or []) / 72,
        boundary_sectors=boundary_sectors or [],
        center_visible=center_visible,
        has_plant_evidence=has_plant_evidence,
    )


def test_composite_stitches_overlapping_frames_and_draws_bold_radii(tmp_path):
    now = datetime(2026, 7, 26, tzinfo=UTC)
    output = tmp_path / "composite.jpg"
    overlay_output = tmp_path / "composite-overlay.jpg"
    measurements = [
        _measurement(tmp_path, 1, 150, now, boundary_sectors=list(range(18))),
        _measurement(
            tmp_path,
            2,
            50,
            now + timedelta(minutes=1),
            boundary_sectors=list(range(18, 36)),
        ),
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
    elsewhere = _measurement(
        tmp_path,
        5,
        4000,
        now + timedelta(minutes=1),
        visible_fraction=0,
        has_plant_evidence=False,
    )

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


def test_one_by_two_evidence_region_builds_three_by_four_grid_crop(tmp_path):
    now = datetime(2026, 7, 26, tzinfo=UTC)
    output = tmp_path / "grid-composite.jpg"
    overlay = tmp_path / "grid-composite-mask.jpg"
    calibration = FarmbotCalibrationInput(
        coordinate_scale=1,
        reference_width=100,
        reference_height=100,
    )
    targets = [
        PhotoGridTarget(
            index=row * 6 + column, row=row, column=column, x=column * 80, y=row * 80, z=0
        )
        for row in range(5)
        for column in range(6)
    ]
    frames = [
        PhotoGridFrame(
            target_index=target.index,
            image_id=1000 + target.index,
            x=target.x,
            y=target.y,
            z=0,
        )
        for target in targets
    ]
    record = PhotoGridRecord(
        config_entry_id="bot",
        started_at=now,
        status="complete",
        bed_bounds={"x": (0, 500), "y": (0, 400)},
        footprint_width_mm=100,
        footprint_height_mm=100,
        calibration=calibration,
        targets=targets,
        frames=frames,
    )
    plant_x, plant_y = 200.0, 200.0
    measurements = []
    useful_cells = {(2, 2): list(range(18)), (2, 3): list(range(18, 36))}
    for target, frame in zip(targets, frames, strict=True):
        center_x = 50 + plant_x - target.x
        center_y = 50 + plant_y - target.y
        sectors = useful_cells.get((target.row, target.column), [])
        item = _measurement(
            tmp_path,
            frame.image_id,
            center_x,
            now + timedelta(seconds=target.index),
            center_y=center_y,
            size=(100, 100),
            boundary_sectors=sectors,
            center_visible=(target.row, target.column) == (2, 2),
            has_plant_evidence=bool(sectors),
            visible_fraction=0.25 if sectors else 0,
            transform_json=json.dumps(
                {
                    "pixels_per_mm_x": 1,
                    "pixels_per_mm_y": 1,
                    "rotation_degrees": 0,
                    "origin_location": "top_left",
                    "image_x": target.x,
                    "image_y": target.y,
                }
            ),
        )
        item = item.model_copy(
            update={
                "recorded_center_x": plant_x,
                "recorded_center_y": plant_y,
            }
        )
        measurements.append(item)

    assert build_plant_composite(
        measurements,
        output,
        overlay,
        grid_record=record,
        plants=[
            SimpleNamespace(
                id=7,
                name="Broccoli",
                openfarm_slug="broccoli",
                x=plant_x,
                y=plant_y,
                radius=30,
            ),
            SimpleNamespace(
                id=8,
                name="Lettuce",
                openfarm_slug="lettuce",
                x=plant_x + 100,
                y=plant_y,
                radius=24,
            ),
        ],
        proposed_radii={8: 32},
    )

    metadata = json.loads(output.with_suffix(".json").read_text())
    clean = cv2.imread(str(output))
    diagnostic = cv2.imread(str(overlay))
    assert metadata["tile_window"]["rows"] == 3
    assert metadata["tile_window"]["columns"] == 4
    assert clean.shape == diagnostic.shape
    assert metadata["standard_and_diagnostic_geometry_identical"] is True
    difference = cv2.absdiff(clean, diagnostic)
    ppm = metadata["pixels_per_mm"]
    min_x, min_y, _, _ = metadata["crop_mm"]
    target_px = (round(-min_x * ppm), round(-min_y * ppm))
    neighbour_px = (round((100 - min_x) * ppm), round(-min_y * ppm))
    target_patch = difference[
        target_px[1] - 12 : target_px[1] + 13,
        target_px[0] - 12 : target_px[0] + 13,
    ]
    neighbour_patch = difference[
        neighbour_px[1] - 10 : neighbour_px[1] + 11,
        neighbour_px[0] - 10 : neighbour_px[0] + 11,
    ]
    assert float(np.mean(target_patch)) > 3
    assert float(np.mean(neighbour_patch)) < 3
