from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from farmbot_vision import web
from farmbot_vision.calibration_store import FarmbotCalibrationInput
from farmbot_vision.photo_grid import (
    PhotoGridFrame,
    PhotoGridRecord,
    PhotoGridStore,
    PhotoGridTarget,
)
from farmbot_vision.photo_quality import inspect_photo_quality


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return encoded.tobytes()


def _garden() -> np.ndarray:
    image = np.full((240, 320, 3), (35, 65, 80), np.uint8)
    cv2.circle(image, (160, 120), 52, (28, 155, 42), -1)
    cv2.line(image, (10, 30), (300, 210), (45, 75, 48), 3)
    return image


def _washed_out() -> np.ndarray:
    image = np.full((240, 320, 3), (245, 250, 248), np.uint8)
    for x in range(0, 320, 20):
        cv2.line(image, (x, 0), (x, 239), (230, 245, 245), 1)
    for y in range(0, 240, 20):
        cv2.line(image, (0, y), (319, y), (230, 245, 245), 1)
    cv2.rectangle(image, (0, 145), (319, 156), (65, 70, 72), -1)
    return image


def _blocking_leaf(*, clear_plant: bool = False) -> np.ndarray:
    image = np.full((240, 320, 3), (35, 65, 80), np.uint8)
    # One close leaf crosses the top and right frame edges.
    cv2.ellipse(image, (285, 40), (180, 105), 20, 0, 360, (30, 145, 45), -1)
    if clear_plant:
        cv2.circle(image, (105, 160), 38, (25, 180, 40), -1)
    return image


def _record() -> PhotoGridRecord:
    target = PhotoGridTarget(index=0, row=0, column=0, x=300, y=250, z=0)
    return PhotoGridRecord(
        config_entry_id="entry-1",
        started_at=datetime.now(UTC),
        bed_bounds={"x": (0, 1000), "y": (0, 800)},
        footprint_width_mm=400,
        footprint_height_mm=300,
        calibration=FarmbotCalibrationInput(
            coordinate_scale=0.25,
            reference_width=1600,
            reference_height=1200,
        ),
        targets=[target],
        frames=[PhotoGridFrame(target_index=0, image_id=10, x=300, y=250, z=0)],
        indexed_targets=True,
    )


def test_quality_rule_separates_washout_close_leaf_and_normal_garden():
    assert inspect_photo_quality(_jpeg(_washed_out())).issue == "washed_out"
    assert inspect_photo_quality(_jpeg(_blocking_leaf())).issue == "leaf_obstruction"
    assert inspect_photo_quality(_jpeg(_garden())).issue == "usable"


def test_leaf_repair_plans_four_distinct_offset_coordinates():
    record = _record()
    targets = web._leaf_offset_targets(record, record.targets[0])

    assert len(targets) == 4
    assert len({(item.x, item.y) for item in targets}) == 4
    assert all((item.x, item.y) != (300, 250) for item in targets)
    assert {(item.x, item.y) for item in targets} == {
        (220, 250),
        (380, 250),
        (300, 190),
        (300, 310),
    }


@pytest.mark.asyncio
async def test_washed_out_original_is_excluded_deleted_and_retaken_once(monkeypatch, tmp_path):
    record = _record()
    store = PhotoGridStore(tmp_path / "grid.json")
    monkeypatch.setattr(web, "photo_grid_store", store)
    deleted = []
    calls = []

    async def quality_jpeg(_record, image_id):
        return _jpeg(_washed_out() if image_id == 10 else _garden())

    async def capture(_record, targets):
        calls.append([(item.x, item.y) for item in targets])
        return [PhotoGridFrame(target_index=0, image_id=20, x=300, y=250, z=0)]

    async def delete(_record, image_id, *, reason):
        deleted.append((image_id, reason))

    monkeypatch.setattr(web, "_quality_jpeg", quality_jpeg)
    monkeypatch.setattr(web, "_capture_quality_targets", capture)
    monkeypatch.setattr(web, "_delete_discarded_grid_image", delete)

    await web._photo_grid_quality_pass(record)
    await web._photo_grid_quality_pass(record)

    assert calls == [[(300.0, 250.0)]]
    assert deleted == [(10, "washed out")]
    assert record.excluded_image_ids == [10]
    assert [item.image_id for item in record.frames] == [20]
    assert record.quality_repairs[0].status == "complete"


@pytest.mark.asyncio
async def test_leaf_repair_keeps_original_and_layers_best_of_four(monkeypatch, tmp_path):
    record = _record()
    store = PhotoGridStore(tmp_path / "grid.json")
    monkeypatch.setattr(web, "photo_grid_store", store)
    deleted = []
    candidate_images = {
        21: _blocking_leaf(),
        22: _garden(),
        23: _blocking_leaf(clear_plant=True),
        24: _blocking_leaf(),
    }

    async def quality_jpeg(_record, image_id):
        return _jpeg(_blocking_leaf() if image_id == 10 else candidate_images[image_id])

    async def capture(_record, targets):
        return [
            PhotoGridFrame(
                target_index=target.index,
                image_id=21 + index,
                x=target.x,
                y=target.y,
                z=target.z,
            )
            for index, target in enumerate(targets)
        ]

    async def delete(_record, image_id, *, reason):
        deleted.append((image_id, reason))

    monkeypatch.setattr(web, "_quality_jpeg", quality_jpeg)
    monkeypatch.setattr(web, "_capture_quality_targets", capture)
    monkeypatch.setattr(web, "_delete_discarded_grid_image", delete)

    await web._photo_grid_quality_pass(record)

    assert [item.image_id for item in record.frames] == [10]
    assert [item.image_id for item in record.quality_overlay_frames] == [22]
    assert record.quality_overlay_frames[0].target_index == 0
    assert record.excluded_image_ids == [21, 23, 24]
    assert {item[0] for item in deleted} == {21, 23, 24}
    assert record.quality_repairs[0].candidate_image_ids == [21, 22, 23, 24]
    assert record.quality_repairs[0].selected_image_id == 22
