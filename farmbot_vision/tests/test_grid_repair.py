from datetime import UTC, datetime, timedelta

import cv2
import numpy as np

from farmbot_vision.grid_repair import (
    GridRepairSettings,
    GridRepairSettingsStore,
    detect_latest_grid_run,
    looks_like_gantry_photo,
)
from farmbot_vision.models import InventoryImage


def _image(image_id: int, minute: int, x: float, y: float) -> InventoryImage:
    return InventoryImage.model_validate(
        {
            "id": image_id,
            "created_at": datetime(2026, 7, 26, 1, minute, tzinfo=UTC),
            "meta": {"x": x, "y": y, "z": 0},
        }
    )


def test_latest_grid_detects_missing_and_gantry_cells():
    images = [
        _image(1, 0, 0, 0),
        _image(2, 1, 0, 100),
        _image(3, 2, 0, 200),
        _image(4, 3, 100, 0),
        _image(5, 4, 100, 100),
        # (100, 200) is missing.
        _image(7, 6, 200, 0),
        _image(8, 7, 200, 100),
        _image(9, 8, 200, 200),
    ]
    run = detect_latest_grid_run(images, {5})
    assert run is not None
    assert run.expected_count == 9
    assert run.coverage == 8 / 9
    assert {(item.x, item.y, item.reason) for item in run.targets} == {
        (100, 100, "gantry"),
        (100, 200, "missing"),
    }


def test_gantry_photos_anywhere_on_grid_perimeter_are_not_repair_targets():
    images = [
        _image(1, 0, 0, 0),
        _image(2, 1, 0, 100),
        _image(3, 2, 0, 200),
        _image(4, 3, 100, 0),
        _image(5, 4, 100, 100),
        _image(6, 5, 100, 200),
    ]
    run = detect_latest_grid_run(images, {1, 3, 5, 6})
    assert run is not None
    assert run.targets == ()


def test_skips_newer_non_grid_cluster():
    grid = [
        _image(i + 1, i, x, y) for i, (x, y) in enumerate([(0, 0), (0, 100), (100, 0), (100, 100)])
    ]
    later = [
        item.model_copy(
            update={"id": 20 + index, "created_at": item.created_at + timedelta(hours=4)}
        )
        for index, item in enumerate(grid[:2])
    ]
    assert detect_latest_grid_run(grid + later).completed_at == grid[-1].created_at


def test_gantry_detector_finds_bright_vertical_rail():
    frame = np.full((240, 320, 3), (45, 70, 35), np.uint8)
    frame[:, 145:175] = (190, 190, 190)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok and looks_like_gantry_photo(encoded.tobytes())


def test_gantry_detector_rejects_plain_garden():
    frame = np.full((240, 320, 3), (35, 90, 35), np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok and not looks_like_gantry_photo(encoded.tobytes())


def test_gantry_detector_rejects_partial_bright_bed_edge():
    frame = np.full((240, 320, 3), (35, 90, 35), np.uint8)
    frame[:160, 145:175] = (190, 190, 190)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok and not looks_like_gantry_photo(encoded.tobytes())


def test_repair_settings_round_trip(tmp_path):
    store = GridRepairSettingsStore(tmp_path / "grid_repair.json")
    store.save(GridRepairSettings(enabled=True, repair_time="04:35"))
    assert store.load() == GridRepairSettings(enabled=True, repair_time="04:35")
