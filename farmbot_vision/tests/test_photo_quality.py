import asyncio
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
from farmbot_vision.photo_quality import inspect_photo_quality, with_neighbor_blur


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


def _large_canopy_with_visible_plant() -> np.ndarray:
    image = np.full((240, 320, 3), (35, 65, 80), np.uint8)
    # This edge-connected canopy is substantial, but smaller than the close
    # leaf gate and leaves a clearly visible second plant in the tile.
    cv2.ellipse(image, (285, 40), (160, 100), 20, 0, 360, (30, 145, 45), -1)
    cv2.circle(image, (105, 160), 38, (25, 180, 40), -1)
    return image


def _textured_garden() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = _garden()
    for _ in range(160):
        x, y = rng.integers((0, 0), (320, 240))
        colour = int(rng.integers(85, 155))
        cv2.circle(
            image,
            (int(x), int(y)),
            int(rng.integers(1, 4)),
            (colour - 25, colour - 10, colour),
            -1,
        )
    return image


def _blurry(image: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def _bright_but_detailed() -> np.ndarray:
    image = np.full((240, 320, 3), (185, 205, 220), np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(500):
        x, y = rng.integers((0, 0), (320, 240))
        colour = tuple(int(value) for value in rng.integers((120, 150, 170), (220, 235, 245)))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(1, 5)), colour, -1)
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
        quality_repair_enabled=True,
    )


def test_quality_rule_separates_washout_close_leaf_and_normal_garden():
    assert inspect_photo_quality(_jpeg(_washed_out())).issue == "washed_out"
    assert inspect_photo_quality(_jpeg(_blocking_leaf())).issue == "leaf_obstruction"
    assert inspect_photo_quality(_jpeg(_large_canopy_with_visible_plant())).issue == "usable"
    assert inspect_photo_quality(_jpeg(_blurry(_garden()))).issue == "blurry"
    assert inspect_photo_quality(_jpeg(_garden())).issue == "usable"
    assert inspect_photo_quality(_jpeg(_bright_but_detailed())).issue == "usable"


def test_quality_retry_settings_support_legacy_and_individual_values():
    legacy = _record()
    assert legacy.quality_retries_enabled
    selected = legacy.model_copy(
        update={
            "quality_repair_enabled": True,
            "quality_repair_blurry_enabled": False,
            "quality_repair_washed_out_enabled": True,
            "quality_repair_close_leaf_enabled": False,
        }
    )
    assert not selected.quality_retry_enabled("blurry")
    assert selected.quality_retry_enabled("washed_out")
    assert not selected.quality_retry_enabled("leaf_obstruction")


@pytest.mark.asyncio
async def test_disabled_quality_type_is_not_repaired(monkeypatch, tmp_path):
    record = _record().model_copy(
        update={
            "quality_repair_enabled": True,
            "quality_repair_blurry_enabled": False,
            "quality_repair_washed_out_enabled": True,
            "quality_repair_close_leaf_enabled": True,
        }
    )
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))
    fetched = []

    async def quality_jpeg(_record, image_id):
        fetched.append(image_id)
        return _jpeg(_blurry(_garden()))

    monkeypatch.setattr(web, "_quality_jpeg", quality_jpeg)

    await web._photo_grid_quality_pass(record)

    assert fetched == [10]
    assert record.quality_repairs == []
    assert record.frames[0].image_id == 10


def test_neighbor_comparison_finds_moderate_blur_without_a_universal_threshold():
    clear = inspect_photo_quality(_jpeg(_textured_garden()))
    moderate = inspect_photo_quality(_jpeg(_blurry(_textured_garden(), 1.0)))

    assert moderate.issue == "usable"
    assert with_neighbor_blur(moderate, [clear, clear]).issue == "blurry"
    assert with_neighbor_blur(moderate, [clear]).issue == "usable"


@pytest.mark.asyncio
async def test_grid_pass_uses_adjacent_cells_to_repair_moderate_blur(monkeypatch, tmp_path):
    record = _record()
    record.targets = [
        PhotoGridTarget(index=index, row=0, column=index, x=200 + index * 100, y=250, z=0)
        for index in range(3)
    ]
    record.frames = [
        PhotoGridFrame(
            target_index=index,
            image_id=10 + index,
            x=target.x,
            y=target.y,
            z=0,
        )
        for index, target in enumerate(record.targets)
    ]
    store = PhotoGridStore(tmp_path / "grid.json")
    monkeypatch.setattr(web, "photo_grid_store", store)
    calls = []

    async def quality_jpeg(_record, image_id):
        if image_id == 11:
            return _jpeg(_blurry(_textured_garden(), 1.0))
        return _jpeg(_textured_garden())

    async def capture(_record, targets):
        calls.append(targets[0].index)
        target = targets[0]
        return [
            PhotoGridFrame(
                target_index=target.index,
                image_id=30,
                x=target.x,
                y=target.y,
                z=target.z,
            )
        ]

    async def delete(_record, _image_id, *, reason):
        assert reason == "blurry"

    monkeypatch.setattr(web, "_quality_jpeg", quality_jpeg)
    monkeypatch.setattr(web, "_capture_quality_targets", capture)
    monkeypatch.setattr(web, "_delete_discarded_grid_image", delete)

    await web._photo_grid_quality_pass(record)

    assert calls == [1]
    assert [item.original_image_id for item in record.quality_repairs] == [11]
    assert sorted(item.image_id for item in record.frames) == [10, 12, 30]


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
async def test_blurry_original_is_excluded_deleted_and_retaken_once(monkeypatch, tmp_path):
    record = _record()
    store = PhotoGridStore(tmp_path / "grid.json")
    monkeypatch.setattr(web, "photo_grid_store", store)
    deleted = []
    calls = []

    async def quality_jpeg(_record, image_id):
        return _jpeg(_blurry(_garden()) if image_id == 10 else _garden())

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
    assert deleted == [(10, "blurry")]
    assert record.excluded_image_ids == [10]
    assert [item.image_id for item in record.frames] == [20]
    assert record.quality_repairs[0].issue == "blurry"
    assert record.quality_repairs[0].status == "complete"


@pytest.mark.asyncio
async def test_still_blurry_retake_is_discarded_without_another_attempt(monkeypatch, tmp_path):
    record = _record()
    store = PhotoGridStore(tmp_path / "grid.json")
    monkeypatch.setattr(web, "photo_grid_store", store)
    deleted = []
    calls = 0

    async def quality_jpeg(_record, _image_id):
        return _jpeg(_blurry(_garden()))

    async def capture(_record, _targets):
        nonlocal calls
        calls += 1
        return [PhotoGridFrame(target_index=0, image_id=20, x=300, y=250, z=0)]

    async def delete(_record, image_id, *, reason):
        deleted.append((image_id, reason))

    monkeypatch.setattr(web, "_quality_jpeg", quality_jpeg)
    monkeypatch.setattr(web, "_capture_quality_targets", capture)
    monkeypatch.setattr(web, "_delete_discarded_grid_image", delete)

    await web._photo_grid_quality_pass(record)
    await web._photo_grid_quality_pass(record)

    assert calls == 1
    assert {item[0] for item in deleted} == {10, 20}
    assert record.excluded_image_ids == [10, 20]
    assert record.frames == []
    assert record.quality_repairs[0].status == "failed"


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


@pytest.mark.asyncio
async def test_live_scout_analyses_each_frame_while_the_grid_is_still_capturing(
    monkeypatch, tmp_path
):
    """The scout must classify a cell's photo as soon as that cell is verified,
    not after the whole route finishes."""
    record = _record()
    record.targets = [
        PhotoGridTarget(index=index, row=0, column=index, x=200 + index * 100, y=250, z=0)
        for index in range(2)
    ]
    record.frames = []
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    async def quality_jpeg(_record, image_id):
        return _jpeg(_washed_out() if image_id == 11 else _garden())

    monkeypatch.setattr(web, "_quality_jpeg", quality_jpeg)
    scout = web._LiveGridScout(record)
    task = asyncio.create_task(scout.run())
    try:
        for index, target in enumerate(record.targets):
            record.frames.append(
                PhotoGridFrame(
                    target_index=target.index,
                    image_id=10 + index,
                    x=target.x,
                    y=target.y,
                    z=0,
                )
            )
            # Only this cell is verified so far; the scout must not need the
            # rest of the grid to reach a verdict about it.
            for _ in range(60):
                if any(item.target_index == target.index for item in record.cell_analysis):
                    break
                await asyncio.sleep(0.05)
        assert [item.target_index for item in record.cell_analysis] == [0, 1]
        assert [item.issue for item in record.cell_analysis] == ["usable", "washed_out"]
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_post_grid_pass_reuses_the_scout_inspection_instead_of_refetching(
    monkeypatch, tmp_path
):
    """Moving the check earlier must not mean paying for it twice."""
    record = _record()
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))
    fetched: list[int] = []

    async def quality_jpeg(_record, image_id):
        fetched.append(image_id)
        return _jpeg(_garden())

    monkeypatch.setattr(web, "_quality_jpeg", quality_jpeg)
    scout = web._LiveGridScout(record)
    task = asyncio.create_task(scout.run())
    try:
        for _ in range(60):
            if record.cell_analysis:
                break
            await asyncio.sleep(0.05)
    finally:
        task.cancel()
    assert fetched == [10]

    await web._photo_grid_quality_pass(record, scout)

    # No second download and no second decode for a frame already inspected.
    assert fetched == [10]
    assert record.quality_repairs == []
