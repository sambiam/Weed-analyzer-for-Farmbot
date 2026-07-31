from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlencode, urljoin, urlsplit
from uuid import uuid4

import pytest
import yaml

import farmbot_vision.web as web
import farmbot_vision.weed_verifier as weed_verifier_module
from farmbot_vision.models import (
    BotList,
    Decision,
    Inventory,
    Measurement,
    SoilMeasurement,
    SoilMotionState,
    SoilPoint,
    SoilPointInventory,
    SoilSite,
    VisionRequestEvent,
)


@pytest.mark.asyncio
async def test_grid_repair_requires_advertised_v2_capability(monkeypatch):
    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                        "integration_version": "2.0.1",
                        "capabilities": [
                            "photo_grid_repair",
                            "verified_photo_grid_repair",
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(web.client, "list_bots", list_bots)
    try:
        with pytest.raises(
            web.HomeAssistantError,
            match="requires FarmBot integration V2.2.0",
        ):
            await web._require_grid_repair_capability()
    finally:
        web.settings.selected_config_entry_id = previous


@pytest.mark.asyncio
async def test_grid_repair_accepts_advertised_v2_capability(monkeypatch):
    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                        "integration_version": "2.0.2",
                        "capabilities": [
                            "photo_grid_repair",
                            "verified_photo_grid_repair",
                            "position_verified_photo_grid_repair",
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(web.client, "list_bots", list_bots)
    try:
        await web._require_grid_repair_capability()
    finally:
        web.settings.selected_config_entry_id = previous


@pytest.mark.asyncio
async def test_whole_grid_requires_illuminated_capture_capability(monkeypatch):
    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                        "integration_version": "2.1.0",
                        "capabilities": ["position_verified_photo_grid_repair"],
                    }
                ]
            }
        )

    monkeypatch.setattr(web.client, "list_bots", list_bots)
    try:
        with pytest.raises(web.HomeAssistantError, match="V2.2.0"):
            await web._require_grid_repair_capability(require_lighting=True)
    finally:
        web.settings.selected_config_entry_id = previous


@pytest.mark.asyncio
async def test_whole_grid_keeps_only_coordinate_verified_frames(monkeypatch, tmp_path):
    from farmbot_vision.calibration_store import FarmbotCalibrationInput
    from farmbot_vision.photo_grid import PhotoGridRecord, PhotoGridStore, PhotoGridTarget

    targets = [
        PhotoGridTarget(index=0, row=0, column=0, x=100, y=200, z=0),
        PhotoGridTarget(index=1, row=0, column=1, x=300, y=200, z=0),
    ]
    record = PhotoGridRecord(
        config_entry_id="entry-1",
        started_at=datetime.now(UTC),
        bed_bounds={"x": (0, 500), "y": (0, 400)},
        footprint_width_mm=250,
        footprint_height_mm=200,
        calibration=FarmbotCalibrationInput(
            coordinate_scale=0.25,
            reference_width=1000,
            reference_height=800,
        ),
        targets=targets,
    )

    async def start(_entry_id, _targets):
        return {"status": "queued", "repair_id": "grid-1", "message": "queued"}

    async def status(_entry_id, _repair_id):
        return {
            "status": "complete",
            "message": "capture complete",
            "frames": [
                {"image_id": 41, "x": 103, "y": 196, "z": 0},
                # Outside the app's independent 25 mm tolerance.
                {"image_id": 42, "x": 330, "y": 200, "z": 0},
            ],
        }

    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))
    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)
    missing = await web._capture_photo_grid_targets(record, targets)

    assert [frame.image_id for frame in record.frames] == [41]
    assert [target.index for target in missing] == [1]
    assert [frame.image_id for frame in web.photo_grid_store.load().frames] == [41]


@pytest.mark.asyncio
async def test_targeted_followup_uses_existing_capture_service_once(monkeypatch, tmp_path):
    from farmbot_vision.calibration_store import FarmbotCalibrationInput
    from farmbot_vision.photo_grid import (
        PhotoGridFrame,
        PhotoGridRecord,
        PhotoGridStore,
        PhotoGridTarget,
    )

    now = datetime.now(UTC)
    record = PhotoGridRecord(
        config_entry_id="entry-1",
        started_at=now,
        completed_at=now,
        status="complete",
        bed_bounds={"x": (0, 500), "y": (0, 400)},
        footprint_width_mm=100,
        footprint_height_mm=100,
        calibration=FarmbotCalibrationInput(
            coordinate_scale=1,
            reference_width=100,
            reference_height=100,
        ),
        targets=[PhotoGridTarget(index=0, row=0, column=0, x=50, y=50, z=0)],
        frames=[PhotoGridFrame(target_index=0, image_id=41, x=50, y=50, z=0)],
        indexed_targets=True,
    )
    inventory = Inventory.model_validate(
        {
            "device_id": "42",
            "generated_at": now,
            "plants": [
                {
                    "id": 7,
                    "name": "Lettuce",
                    "openfarm_slug": "lettuce",
                    "x": 250,
                    "y": 200,
                    "z": 0,
                    "radius": 20,
                    "plant_stage": "planted",
                }
            ],
            "weeds": [],
            "images": [],
            "curves": [],
            "camera_calibration": {"available": False},
        }
    )
    calls = []

    async def get_inventory(_request):
        return inventory

    async def start(_entry_id, targets):
        calls.append(targets)
        return {"status": "queued", "repair_id": "targeted-1", "message": "queued"}

    async def status(_entry_id, _repair_id):
        return {
            "status": "complete",
            "message": "targeted capture complete",
            "frames": [{"image_id": 99, "target_index": 1, "x": 250, "y": 200, "z": 0}],
        }

    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))
    monkeypatch.setattr(web.client, "inventory", get_inventory)
    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    await web._capture_targeted_plant_photos(record)
    await web._capture_targeted_plant_photos(record)

    assert len(calls) == 1
    assert calls[0] == [{"x": 250.0, "y": 200.0, "z": 0.0, "index": 1}]
    assert len(record.targeted_captures) == 1
    assert record.targeted_captures[0].status == "complete"
    assert record.targeted_captures[0].image_id == 99


def _photo_grid_record(targets, *, tmp_path):
    from farmbot_vision.calibration_store import FarmbotCalibrationInput
    from farmbot_vision.photo_grid import PhotoGridRecord

    return PhotoGridRecord(
        config_entry_id="entry-1",
        started_at=datetime.now(UTC),
        bed_bounds={"x": (0, 500), "y": (0, 400)},
        footprint_width_mm=250,
        footprint_height_mm=200,
        calibration=FarmbotCalibrationInput(
            coordinate_scale=0.25,
            reference_width=1000,
            reference_height=800,
        ),
        targets=targets,
        quality_repair_enabled=True,
    )


def test_grid_status_uses_the_canonical_6_by_9_plan(tmp_path):
    from farmbot_vision.photo_grid import PhotoGridFrame, PhotoGridTarget

    targets = [
        PhotoGridTarget(
            index=row * 9 + column,
            row=row,
            column=column,
            x=column * 100,
            y=row * 100,
            z=0,
        )
        for row in range(6)
        for column in range(9)
    ]
    record = _photo_grid_record(targets, tmp_path=tmp_path)
    record.status = "running"
    record.message = "Taking the next coordinate-verified photo"
    record.frames = [
        PhotoGridFrame(
            target_index=target.index,
            image_id=1000 + target.index,
            x=target.x,
            y=target.y,
            z=target.z,
        )
        for target in targets[:10]
    ]

    status = web._photo_grid_status(record)

    assert status["rows"] == 6
    assert status["columns"] == 9
    assert status["total"] == 54
    assert status["verified"] == 10
    assert status["percentage"] == 19
    assert [cell["state"] for cell in status["cells"][:12]] == [
        *(["verified"] * 10),
        "active",
        "unattempted",
    ]


def test_grid_status_marks_cells_without_a_photo_red(tmp_path):
    """A cell the run drove past without producing a photo is red; a cell that
    holds a flawed photo stays green inside with a red border."""
    from farmbot_vision.photo_grid import (
        PhotoGridFrame,
        PhotoGridQualityRepair,
        PhotoGridTarget,
    )

    targets = [
        PhotoGridTarget(index=index, row=0, column=index, x=index * 100, y=0, z=0)
        for index in range(4)
    ]
    record = _photo_grid_record(targets, tmp_path=tmp_path)
    record.status = "failed"
    record.frames = [
        PhotoGridFrame(target_index=0, image_id=10, x=0, y=0, z=0),
        PhotoGridFrame(target_index=1, image_id=11, x=100, y=0, z=0),
    ]
    record.quality_repairs.append(
        PhotoGridQualityRepair(
            target_index=1,
            issue="leaf_obstruction",
            original_image_id=11,
            status="failed",
            attempted_at=datetime.now(UTC),
        )
    )

    status = web._photo_grid_status(record)

    assert [cell["state"] for cell in status["cells"]] == [
        "verified",
        "verified",
        "missing",
        "missing",
    ]
    assert [cell["issue"] for cell in status["cells"]] == [None, "leaf_obstruction", None, None]
    assert status["missing"] == 2
    assert status["flagged"] == 1


def test_grid_status_keeps_the_red_border_while_a_flagged_cell_is_retried(tmp_path):
    """An in-progress repair turns the interior blue but must not clear the
    red border, which only a completed repair earns."""
    from farmbot_vision.photo_grid import (
        PhotoGridCellAnalysis,
        PhotoGridFrame,
        PhotoGridQualityRepair,
        PhotoGridTarget,
    )

    targets = [
        PhotoGridTarget(index=index, row=0, column=index, x=index * 100, y=0, z=0)
        for index in range(3)
    ]
    record = _photo_grid_record(targets, tmp_path=tmp_path)
    record.status = "quality_repair"
    record.frames = [
        PhotoGridFrame(target_index=index, image_id=10 + index, x=index * 100, y=0, z=0)
        for index in range(3)
    ]
    record.cell_analysis = [
        PhotoGridCellAnalysis(
            target_index=index,
            image_id=10 + index,
            issue="blurry" if index else "usable",
            analysed_at=datetime.now(UTC),
        )
        for index in range(3)
    ]
    record.quality_repairs = [
        PhotoGridQualityRepair(
            target_index=1,
            issue="blurry",
            original_image_id=11,
            status="attempting",
            attempted_at=datetime.now(UTC),
        ),
        PhotoGridQualityRepair(
            target_index=2,
            issue="blurry",
            original_image_id=12,
            status="complete",
            attempted_at=datetime.now(UTC),
        ),
    ]

    status = web._photo_grid_status(record)

    by_index = {cell["index"]: cell for cell in status["cells"]}
    assert (by_index[0]["state"], by_index[0]["issue"]) == ("verified", None)
    # Being retried: blue interior, red border retained.
    assert (by_index[1]["state"], by_index[1]["issue"]) == ("active", "blurry")
    # Verified as fixed: the border is earned back.
    assert (by_index[2]["state"], by_index[2]["issue"]) == ("verified", None)
    assert status["flagged"] == 1


def test_grid_status_respects_map_origin_and_rotate_map(tmp_path):
    """Cell layout must match the calibration tab's whole-map display
    orientation, not the raw gantry row/column order."""
    from farmbot_vision.calibration_store import FarmbotCalibrationInput
    from farmbot_vision.photo_grid import PhotoGridRecord, PhotoGridTarget

    targets = [
        PhotoGridTarget(
            index=row * 3 + column,
            row=row,
            column=column,
            x=column * 100,
            y=row * 100,
            z=0,
        )
        for row in range(2)
        for column in range(3)
    ]
    record = PhotoGridRecord(
        config_entry_id="entry-1",
        started_at=datetime.now(UTC),
        bed_bounds={"x": (0, 300), "y": (0, 200)},
        footprint_width_mm=250,
        footprint_height_mm=200,
        calibration=FarmbotCalibrationInput(
            coordinate_scale=0.25,
            reference_width=1000,
            reference_height=800,
            map_origin="bottom_right",
            rotate_map=True,
        ),
        targets=targets,
    )

    status = web._photo_grid_status(record)

    # rotate_map transposes the 2 rows x 3 columns lattice into 3 rows x 2
    # columns; bottom_right then mirrors both axes of that transposed grid, so
    # the physical (row 0, column 0) cell lands at the display's bottom-right.
    assert status["rows"] == 3
    assert status["columns"] == 2
    by_index = {cell["index"]: (cell["row"], cell["column"]) for cell in status["cells"]}
    assert by_index[0] == (2, 1)  # row=0,column=0 (physical top-left)
    assert by_index[2] == (0, 1)  # row=0,column=2 (physical top-right)
    assert by_index[3] == (2, 0)  # row=1,column=0 (physical bottom-left)
    assert by_index[5] == (0, 0)  # row=1,column=2 (physical bottom-right)


@pytest.mark.asyncio
async def test_worker_requeues_only_unverified_targets_after_batch_error(monkeypatch, tmp_path):
    """A batch error must not discard frames verified during that same batch's
    polling -- only targets still missing a verified frame get requeued."""
    from farmbot_vision.photo_grid import PhotoGridFrame, PhotoGridStore, PhotoGridTarget

    targets = [
        PhotoGridTarget(index=0, row=0, column=0, x=100, y=200, z=0),
        PhotoGridTarget(index=1, row=0, column=1, x=300, y=200, z=0),
    ]
    record = _photo_grid_record(targets, tmp_path=tmp_path)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    chunks_seen: list[list[int]] = []

    async def fake_capture(record_arg, chunk):
        chunks_seen.append([target.index for target in chunk])
        if len(chunks_seen) == 1:
            # Simulate target 0's frame being verified and merged into
            # record.frames during polling, moments before the batch as a
            # whole errors out (e.g. a dropped connection on a later poll).
            record_arg.frames = [PhotoGridFrame(target_index=0, image_id=41, x=103, y=196, z=0)]
        raise web.HomeAssistantError("simulated batch failure")

    monkeypatch.setattr(web, "_capture_photo_grid_targets", fake_capture)

    async def no_analysis(_record):
        return None

    async def no_scout(_scout):
        return None

    monkeypatch.setattr(web, "_analyse_completed_photo_grid", no_analysis)
    monkeypatch.setattr(web._LiveGridScout, "run", no_scout)

    await web._photo_grid_worker(record)

    # The already-verified target must never be resent to the bot.
    assert chunks_seen[0] == [0, 1]
    assert all(0 not in chunk for chunk in chunks_seen[1:])
    assert [frame.target_index for frame in record.frames] == [0]
    assert record.status == "failed"


@pytest.mark.asyncio
async def test_batch_start_retries_busy_rejection_then_succeeds(monkeypatch, tmp_path):
    from farmbot_vision.photo_grid import PhotoGridTarget

    targets = [PhotoGridTarget(index=0, row=0, column=0, x=100, y=200, z=0)]
    record = _photo_grid_record(targets, tmp_path=tmp_path)

    attempts = {"count": 0}

    async def start(_entry_id, _targets):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return {"status": "rejected", "message": "FarmBot is busy with another task"}
        return {"status": "queued", "repair_id": "grid-1", "message": "queued"}

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.asyncio, "sleep", fake_sleep)

    result = await web._start_photo_grid_batch(record, targets)

    assert result["repair_id"] == "grid-1"
    assert attempts["count"] == 3
    # Backoff grows with each retry.
    assert sleeps == [
        web.PHOTO_GRID_BATCH_START_RETRY_BASE_SECONDS,
        web.PHOTO_GRID_BATCH_START_RETRY_BASE_SECONDS * 2,
    ]


@pytest.mark.asyncio
async def test_batch_start_does_not_retry_a_non_busy_rejection(monkeypatch, tmp_path):
    from farmbot_vision.photo_grid import PhotoGridTarget

    targets = [PhotoGridTarget(index=0, row=0, column=0, x=100, y=200, z=0)]
    record = _photo_grid_record(targets, tmp_path=tmp_path)

    async def start(_entry_id, _targets):
        return {"status": "rejected", "message": "Calibration is missing"}

    slept = False

    async def fake_sleep(_seconds):
        nonlocal slept
        slept = True

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.asyncio, "sleep", fake_sleep)

    with pytest.raises(web.HomeAssistantError, match="Calibration is missing"):
        await web._start_photo_grid_batch(record, targets)
    assert not slept


def test_chunk_size_never_exceeds_the_integration_target_cap():
    """Regression guard for the 2.3.1 outage.

    Home Assistant validates `start_vision_grid_repair`'s schema
    (`vol.Length(min=1, max=12)` on `targets`) before the handler runs, so an
    oversized batch is refused with a bare HTTP 400 and captures nothing at
    all. Raising the chunk size past the cap silently lost 75 of 77
    coordinates; only the short final remainder was small enough to pass.
    """
    from farmbot_vision.photo_grid import (
        PHOTO_GRID_CHUNK_SIZE,
        PHOTO_GRID_MAX_TARGETS_PER_CALL,
    )

    assert PHOTO_GRID_MAX_TARGETS_PER_CALL == 12
    assert 1 <= PHOTO_GRID_CHUNK_SIZE <= PHOTO_GRID_MAX_TARGETS_PER_CALL


@pytest.mark.asyncio
async def test_rejected_batch_is_split_and_retried_in_smaller_calls(monkeypatch, tmp_path):
    """A batch refused outright must degrade into smaller calls, not vanish.

    Mirrors the real failure: the integration refuses any batch above a cap,
    so the app must keep halving rather than stranding every coordinate.
    """
    from farmbot_vision.photo_grid import PhotoGridStore, PhotoGridTarget

    targets = [
        PhotoGridTarget(index=index, row=0, column=index, x=100 * index, y=200, z=0)
        for index in range(4)
    ]
    record = _photo_grid_record(targets, tmp_path=tmp_path)
    accepted: list[int] = []

    async def start(_entry_id, payload):
        if len(payload) > 2:
            raise web.HomeAssistantError("non-retryable Home Assistant response 400")
        accepted.append(len(payload))
        return {"status": "queued", "repair_id": f"grid-{len(accepted)}", "message": "queued"}

    async def status(_entry_id, repair_id):
        first = repair_id == "grid-1"
        return {
            "status": "complete",
            "message": "capture complete",
            "frames": [
                {"image_id": 10, "x": 0, "y": 200, "z": 0},
                {"image_id": 11, "x": 100, "y": 200, "z": 0},
            ]
            if first
            else [
                {"image_id": 12, "x": 200, "y": 200, "z": 0},
                {"image_id": 13, "x": 300, "y": 200, "z": 0},
            ],
        }

    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))
    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    missing = await web._capture_photo_grid_targets(record, targets)

    assert accepted == [2, 2]
    assert missing == []
    assert sorted(frame.target_index for frame in record.frames) == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_split_batch_reports_unverified_targets_when_every_half_fails(monkeypatch, tmp_path):
    """If splitting cannot help, the targets come back as unverified rather
    than raising past the worker's per-batch accounting."""
    from farmbot_vision.photo_grid import PhotoGridStore, PhotoGridTarget

    targets = [
        PhotoGridTarget(index=index, row=0, column=index, x=100 * index, y=200, z=0)
        for index in range(2)
    ]
    record = _photo_grid_record(targets, tmp_path=tmp_path)

    async def start(_entry_id, _payload):
        raise web.HomeAssistantError("non-retryable Home Assistant response 400")

    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))
    monkeypatch.setattr(web.client, "start_grid_repair", start)

    missing = await web._capture_photo_grid_targets(record, targets)

    assert [target.index for target in missing] == [0, 1]
    assert record.frames == []


@pytest.mark.asyncio
async def test_worker_stops_early_when_a_pass_verifies_no_new_frames(monkeypatch, tmp_path):
    from farmbot_vision.photo_grid import PhotoGridStore, PhotoGridTarget

    targets = [PhotoGridTarget(index=0, row=0, column=0, x=100, y=200, z=0)]
    record = _photo_grid_record(targets, tmp_path=tmp_path)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    calls = {"count": 0}

    async def fake_capture(_record_arg, chunk):
        calls["count"] += 1
        # Never verifies anything, so every pass would be identical.
        return list(chunk)

    monkeypatch.setattr(web, "_capture_photo_grid_targets", fake_capture)

    async def no_analysis(_record):
        return None

    async def no_scout(_scout):
        return None

    monkeypatch.setattr(web, "_analyse_completed_photo_grid", no_analysis)
    monkeypatch.setattr(web._LiveGridScout, "run", no_scout)

    await web._photo_grid_worker(record)

    # The stall guard must stop after the first pass instead of running all
    # PHOTO_GRID_WORKER_MAX_PASSES passes against a target that can't verify.
    assert calls["count"] == 1
    assert record.status == "failed"
    assert "after 1 pass" in record.message


@pytest.mark.asyncio
async def test_legacy_grid_repair_recheck_route_is_removed(monkeypatch):
    """Repair is part of the canonical grid run and has no manual endpoint."""
    import base64 as b64
    from types import SimpleNamespace

    import cv2
    import numpy as np

    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"
    web.grid_repair_state.update(run=None, checked_at=None, error="", message="")

    def _image(image_id: int, minute: int, x: float, y: float) -> dict:
        return {
            "id": image_id,
            "created_at": f"2026-07-27T01:{minute:02d}:00+00:00",
            "x": x,
            "y": y,
            "z": 0,
        }

    async def inventory(_request):
        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": datetime.now(UTC),
                "plants": [],
                "images": [
                    _image(1, 0, 0, 0),
                    _image(2, 1, 0, 100),
                    _image(3, 2, 0, 200),
                    _image(4, 3, 100, 0),
                    _image(5, 4, 100, 100),
                    # (100, 200) is missing.
                ],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    plain_frame = np.full((120, 160, 3), (35, 90, 35), np.uint8)
    _, encoded = cv2.imencode(".jpg", plain_frame)
    plain_jpeg_base64 = b64.b64encode(encoded.tobytes()).decode()

    async def image(_request, _max_bytes):
        return SimpleNamespace(image_base64=plain_jpeg_base64)

    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "image", image)
    try:
        status, headers, _ = await asgi_request("/grid-repair/recheck", method="POST")
    finally:
        web.settings.selected_config_entry_id = previous

    assert status == 404
    assert b"location" not in headers


@pytest.mark.asyncio
async def test_grid_repair_queues_only_one_verified_target_at_a_time(monkeypatch):
    """Large repairs no longer rely on the legacy twelve-target batch limit."""
    import base64 as b64
    from types import SimpleNamespace

    import cv2
    import numpy as np

    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"
    web.grid_repair_state.update(run=None, checked_at=None, error="", message="")

    # A 6x6 grid (36 expected cells) with a 3x4 block plus one extra cell
    # missing (13 missing, exceeding the 12-per-call cap) while every row and
    # column keeps at least one present image, so all 6 x/y axis positions are
    # still detected -- an entirely absent row/column wouldn't register as
    # "missing" at all, since the grid axes are derived from present images.
    xs = [i * 100 for i in range(6)]
    ys = [i * 100 for i in range(6)]
    missing_cells = {(x, y) for x in xs[3:] for y in ys[2:]} | {(xs[2], ys[5])}
    present_cells = [(x, y) for x in xs for y in ys if (x, y) not in missing_cells]

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                        "integration_version": "2.0.2",
                        "capabilities": [
                            "photo_grid_repair",
                            "verified_photo_grid_repair",
                            "position_verified_photo_grid_repair",
                        ],
                    }
                ]
            }
        )

    async def inventory(_request):
        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": datetime.now(UTC),
                "plants": [],
                "images": [
                    {
                        "id": index,
                        "created_at": f"2026-07-27T01:{index % 60:02d}:00+00:00",
                        "x": x,
                        "y": y,
                        "z": 0,
                    }
                    for index, (x, y) in enumerate(present_cells)
                ],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    plain_frame = np.full((120, 160, 3), (35, 90, 35), np.uint8)
    _, encoded = cv2.imencode(".jpg", plain_frame)
    plain_jpeg_base64 = b64.b64encode(encoded.tobytes()).decode()

    async def image(_request, _max_bytes):
        return SimpleNamespace(image_base64=plain_jpeg_base64)

    captured_targets = []

    async def start_grid_repair(_entry_id, targets):
        captured_targets.append(targets)
        return {"status": "queued", "repair_id": "r1", "message": "Photo-grid repair queued"}

    monkeypatch.setattr(web.client, "list_bots", list_bots)
    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "image", image)
    monkeypatch.setattr(web.client, "start_grid_repair", start_grid_repair)
    try:
        result = await web.start_photo_grid_repair()
    finally:
        if web.grid_repair_task is not None:
            web.grid_repair_task.cancel()
            await web.asyncio.gather(web.grid_repair_task, return_exceptions=True)
            web.grid_repair_task = None
        web.settings.selected_config_entry_id = previous

    assert len(captured_targets[0]) == 1
    assert result["status"] == "queued"
    assert "13 cells remaining" in result["message"]


@pytest.mark.asyncio
async def test_grid_repair_worker_continues_after_each_processed_image(monkeypatch):
    from farmbot_vision.grid_repair import GridRun, RepairTarget

    now = datetime.now(UTC)

    def repair_run(*targets):
        return GridRun(
            started_at=now,
            completed_at=now,
            images=(),
            expected_count=4,
            coverage=0.5,
            targets=targets,
        )

    first = RepairTarget(100, 200, 0, "missing")
    second = RepairTarget(300, 400, 0, "missing")
    initial = repair_run(first, second)
    after_first = repair_run(second)
    complete = repair_run()
    inspections = iter([after_first, complete])
    queued = []

    async def status(_entry_id, repair_id):
        return {
            "status": "complete",
            "repair_id": repair_id,
            "message": "Processed target image verified",
        }

    async def inspect(*, force=False):
        assert force is True
        return next(inspections)

    async def queue_one(run, **_kwargs):
        queued.append(run.targets[0])
        return (
            {
                "status": "queued",
                "repair_id": "r2",
                "message": "Photo-grid repair queued",
            },
            run.targets[0],
        )

    monkeypatch.setattr(web.settings, "selected_config_entry_id", "entry-1")
    monkeypatch.setattr(web.client, "grid_repair_status", status)
    monkeypatch.setattr(web, "inspect_photo_grid", inspect)
    monkeypatch.setattr(web, "_queue_one_repair_target", queue_one)

    await web._photo_grid_repair_worker(
        session_id="session-1",
        run=initial,
        active_repair_id="r1",
    )

    assert queued == [second]
    assert web.grid_repair_state["status"] == "complete"
    assert "every cell" in str(web.grid_repair_state["message"])


@pytest.mark.asyncio
async def test_grid_repair_worker_moves_on_after_six_failed_camera_attempts(monkeypatch):
    from farmbot_vision.grid_repair import GridRun, RepairTarget

    now = datetime.now(UTC)

    def repair_run(*targets):
        return GridRun(
            started_at=now,
            completed_at=now,
            images=(),
            expected_count=4,
            coverage=0.5,
            targets=targets,
        )

    first = RepairTarget(100, 200, 0, "missing")
    second = RepairTarget(300, 400, 0, "missing")
    initial = repair_run(first, second)
    after_failure = repair_run(first, second)
    only_failed_cell_left = repair_run(first)
    inspections = iter([after_failure, only_failed_cell_left])
    statuses = iter(
        [
            {
                "status": "failed",
                "repair_id": "r1",
                "message": "No image after 6 attempts",
            },
            {
                "status": "complete",
                "repair_id": "r2",
                "message": "Processed target image verified",
            },
        ]
    )
    queued = []

    async def status(_entry_id, _repair_id):
        return next(statuses)

    async def inspect(*, force=False):
        assert force is True
        return next(inspections)

    async def start_grid_repair(_entry_id, payload):
        queued.append(payload)
        return {
            "status": "queued",
            "repair_id": "r2",
            "message": "Photo-grid repair queued",
        }

    monkeypatch.setattr(web.settings, "selected_config_entry_id", "entry-1")
    monkeypatch.setattr(web.client, "grid_repair_status", status)
    monkeypatch.setattr(web.client, "start_grid_repair", start_grid_repair)
    monkeypatch.setattr(web, "inspect_photo_grid", inspect)

    await web._photo_grid_repair_worker(
        session_id="session-1",
        run=initial,
        active_repair_id="r1",
    )

    assert queued == [[{"x": 300, "y": 400, "z": 0}]]
    assert web.grid_repair_state["status"] == "failed"
    assert "1 cell could not be captured after six camera attempts" in str(
        web.grid_repair_state["message"]
    )


@pytest.mark.asyncio
async def test_grid_repair_credits_each_new_photo_and_retires_the_gantry_frame(
    monkeypatch, tmp_path
):
    """A completed cell must stop being requested, and its gantry photo must go."""
    from farmbot_vision.grid_repair import GridRun, RepairCaptureStore, RepairTarget

    now = datetime.now(UTC)
    target = RepairTarget(100, 100, 0, "gantry", 5)
    initial = GridRun(
        started_at=now,
        completed_at=now,
        images=(),
        expected_count=4,
        coverage=1,
        targets=(target,),
    )
    repaired = GridRun(
        started_at=now,
        completed_at=now,
        images=(),
        expected_count=4,
        coverage=1,
        targets=(),
    )
    deleted: list[int] = []

    async def status(_entry_id, repair_id):
        return {
            "status": "complete",
            "repair_id": repair_id,
            "message": "Processed target image verified",
            "frames": [{"image_id": 30, "x": 100, "y": 100, "z": 0}],
        }

    async def inspect(*, force=False):
        return repaired

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                        "integration_version": "2.1.0",
                        "capabilities": ["vision_image_deletion"],
                    }
                ]
            }
        )

    async def delete_image(_entry_id, image_id):
        deleted.append(image_id)
        return {"status": "deleted", "image_id": image_id}

    store = RepairCaptureStore(tmp_path / "captures.json")
    monkeypatch.setattr(web, "repair_capture_store", store)
    monkeypatch.setattr(web.settings, "selected_config_entry_id", "entry-1")
    monkeypatch.setattr(web.client, "grid_repair_status", status)
    monkeypatch.setattr(web.client, "list_bots", list_bots)
    monkeypatch.setattr(web.client, "delete_image", delete_image)
    monkeypatch.setattr(web, "inspect_photo_grid", inspect)

    await web._photo_grid_repair_worker(session_id="session-1", run=initial, active_repair_id="r1")

    assert deleted == [5]
    assert store.load().image_ids == [30]
    assert store.load().run_started_at == now
    assert web.grid_repair_state["status"] == "complete"


@pytest.mark.asyncio
async def test_grid_repair_gives_up_on_a_cell_that_stays_unusable(monkeypatch, tmp_path):
    """A cell that photographs the gantry every time must not loop forever."""
    from farmbot_vision.grid_repair import GridRun, RepairCaptureStore, RepairTarget

    now = datetime.now(UTC)
    target = RepairTarget(100, 100, 0, "gantry", 5)
    stuck = GridRun(
        started_at=now,
        completed_at=now,
        images=(),
        expected_count=4,
        coverage=1,
        targets=(target,),
    )
    frame_ids = iter(range(30, 40))
    queued = []

    async def status(_entry_id, repair_id):
        return {
            "status": "complete",
            "repair_id": repair_id,
            "message": "Processed target image verified",
            "frames": [{"image_id": next(frame_ids), "x": 100, "y": 100, "z": 0}],
        }

    async def inspect(*, force=False):
        return stuck

    async def queue_one(run, **_kwargs):
        queued.append(run.targets[0])
        return ({"status": "queued", "repair_id": "r2", "message": "queued"}, run.targets[0])

    async def delete_replaced(_image_id):
        return None

    monkeypatch.setattr(web, "repair_capture_store", RepairCaptureStore(tmp_path / "c.json"))
    monkeypatch.setattr(web.settings, "selected_config_entry_id", "entry-1")
    monkeypatch.setattr(web.client, "grid_repair_status", status)
    monkeypatch.setattr(web, "inspect_photo_grid", inspect)
    monkeypatch.setattr(web, "_queue_one_repair_target", queue_one)
    monkeypatch.setattr(web, "_delete_replaced_gantry_image", delete_replaced)

    await web._photo_grid_repair_worker(session_id="session-1", run=stuck, active_repair_id="r1")

    assert queued == [target]
    assert web.grid_repair_state["status"] == "failed"
    assert "1 cell still had no usable photo after 2 fresh captures" in str(
        web.grid_repair_state["message"]
    )


@pytest.mark.asyncio
async def test_dashboard_removes_legacy_gantry_repair_controls(
    monkeypatch,
):
    from farmbot_vision.grid_repair import GridRun, RepairTarget
    from farmbot_vision.models import InventoryImage

    now = datetime.now(UTC)
    images = tuple(
        InventoryImage.model_validate({"id": index + 1, "created_at": now, "x": x, "y": y, "z": 0})
        for index, (x, y) in enumerate(
            [
                (0, 0),
                (0, 100),
                (0, 200),
                (100, 0),
                (100, 100),
                (100, 200),
                (200, 0),
                (200, 100),
                (200, 200),
            ]
        )
    )
    run = GridRun(
        started_at=now,
        completed_at=now,
        images=images,
        expected_count=9,
        coverage=1,
        targets=(RepairTarget(100, 100, 0, "gantry", 5),),
    )

    async def inspect(*, force=False):
        return run

    monkeypatch.setattr(web.settings, "selected_config_entry_id", "debug-entry")
    monkeypatch.setattr(web, "inspect_photo_grid", inspect)
    monkeypatch.setitem(web.grid_repair_state, "gantry_image_ids", (2, 5))

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert "Repair photo grid" not in html
    assert "View gantry photos" not in html
    assert "id=gantry-modal" not in html
    assert "Grid status" in html
    assert "info-card" in html
    assert "analysis-card" in html
    assert "Photo analysis" in html
    assert "Last job" not in html
    assert "Last run:" not in html
    assert "Calibration warnings" not in html
    assert html.count("name=quality_repair_blurry_enabled") == 1
    assert html.count("name=quality_repair_washed_out_enabled") == 1
    assert html.count("name=quality_repair_close_leaf_enabled") == 1
    assert "name=quality_repair_enabled" not in html
    assert "Uses the saved scale" not in html
    assert html.count('action="analysis/clear-measurements"') == 1
    assert html.count('action="analysis/clear-recommendations"') == 1
    assert html.count('action="analysis/clear-weeds"') == 1


def _review_measurement(**updates) -> Measurement:
    values = {
        "measurement_id": uuid4(),
        "config_entry_id": "review-bot",
        "plant_id": 812,
        "crop_slug": "lettuce",
        "image_id": 19,
        "image_timestamp": datetime.now(UTC),
        "current_radius_mm": 40,
        "typical_canopy_radius_mm": 45,
        "maximum_accepted_canopy_radius_mm": 50,
        "recommended_protection_radius_mm": 70,
        "confidence": 0.42,
        "decision": Decision.RECOMMENDED,
        "reason": "safe radius increase",
        "algorithm_version": "test",
    }
    values.update(updates)
    return Measurement(**values)


async def asgi_request(
    path: str,
    *,
    method: str = "GET",
    query_string: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
    form: dict[str, str] | None = None,
    raw_body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, dict[bytes, bytes], bytes]:
    messages: list[dict] = []
    body = bytearray()
    request_body = urlencode(form).encode() if form is not None else (raw_body or b"")
    request_headers = list(headers or [])
    if raw_body is not None and content_type is not None:
        request_headers += [
            (b"content-type", content_type.encode()),
            (b"content-length", str(len(request_body)).encode()),
        ]
    if form is not None:
        request_headers += [
            (b"content-type", b"application/x-www-form-urlencoded"),
            (b"content-length", str(len(request_body)).encode()),
        ]

    async def receive() -> dict:
        return {"type": "http.request", "body": request_body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)
        if message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    encoded_path = path.encode("ascii")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": encoded_path,
        "query_string": query_string,
        "headers": request_headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    await web.app(scope, receive, send)
    response = next(message for message in messages if message["type"] == "http.response.start")
    return response["status"], dict(response["headers"]), bytes(body)


@pytest.mark.asyncio
async def test_soil_height_page_lists_points_and_warns_below_three(monkeypatch):
    async def soil_points(_entry_id):
        return SoilPointInventory(
            device_id="42",
            generated_at=datetime.now(UTC),
            points=[
                SoilPoint(
                    id=70,
                    name="Clear soil",
                    x=100,
                    y=200,
                    z=-400,
                    updated_at=datetime(2026, 7, 1, tzinfo=UTC),
                ),
            ],
            motion=SoilMotionState(
                connected=True,
                busy=False,
                locked=False,
                position={"x": 0, "y": 0, "z": 0},
                z_direction=-1,
                axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
            ),
        )

    monkeypatch.setattr(web.settings, "selected_config_entry_id", "soil-entry")

    async def safe_sites(_entry_id, _baseline):
        inventory = await soil_points(_entry_id)
        return inventory, [
            SoilSite(
                point_id=70,
                point_name="Clear soil",
                expected_x=100,
                expected_y=200,
                expected_z=-400,
                point_updated_at=datetime(2026, 7, 1, tzinfo=UTC),
                capture_x=175,
                capture_y=200,
                relocation_distance_mm=75,
                clearance_mm=20,
            )
        ]

    monkeypatch.setattr(web.soil_jobs, "safe_sites", safe_sites)
    status, _, body = await asgi_request("/soil-height")
    assert status == 200
    assert b"Clear soil" in body
    assert b"Measure selected" in body
    assert b"Fewer than three stale soil points" in body
    assert b"replace the assigned stale point" in body


@pytest.mark.asyncio
async def test_soil_apply_is_human_approved_and_audited(monkeypatch):
    measurement = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="soil-entry",
        point_id=72,
        point_name="Soil 72",
        expected_x=100,
        expected_y=200,
        old_z_mm=-400,
        point_updated_at=datetime(2026, 7, 1, tzinfo=UTC),
        capture_x=125,
        capture_y=225,
        relocation_distance_mm=35.36,
        proposed_z_mm=-395,
        confidence=0.9,
        uncertainty_mm=3,
        status="valid",
        reason="passed",
    )
    web.database.save_soil_measurement(measurement)
    seen = {}

    async def apply_soil_height(request):
        seen.update(request.model_dump(mode="json"))
        return {"status": "applied", "message": "updated"}

    monkeypatch.setattr(web.client, "apply_soil_height", apply_soil_height)
    result = await web._apply_soil_measurement(str(measurement.measurement_id))
    assert result["status"] == "applied"
    assert seen["apply"] is True
    assert seen["human_approved"] is True
    assert web.database.soil_measurement(str(measurement.measurement_id))["status"] == "applied"


@pytest.mark.asyncio
async def test_root_and_duplicate_leading_slash_routes():
    status, _, body = await asgi_request("/")
    assert status == 200
    assert b"FarmBot Vision" in body
    assert b'action="analysis/clear-recommendations"' in body
    assert b'action="analysis/clear-weeds"' in body
    assert b'action="analysis/clear-measurements"' in body
    assert b"gridCroppedFootprint" in body
    assert b"gridTileBounds" in body
    assert b"strokeGridTiles" in body
    assert b"tile(s) need a new grid run at this rotation" in body
    assert b"X-FarmBot-Oriented-Width" in body

    for path in ("//", "///"):
        status, _, body = await asgi_request(path)
        assert status == 200
        assert b"FarmBot Vision" in body


@pytest.mark.asyncio
async def test_duplicate_slashes_reach_health_and_settings():
    status, _, body = await asgi_request("//api/health", query_string=b"check=//")
    assert status == 200
    assert json.loads(body)["status"] == "ok"

    status, _, body = await asgi_request("///settings")
    assert status == 200
    assert b"FarmBot calibration" in body
    assert b"id=map_origin" in body
    assert b"id=rotate_map" in body
    assert b"Rotate map (swap X and Y)" in body
    assert b"id=date_from" in body
    assert b"newest photo at each FarmBot coordinate" in body
    assert b"id=zoom-in" in body
    assert b"id=zoom-out" in body
    assert b"id=zoom-reset" in body
    assert b"calibrationTileBounds" in body
    assert b"strokeCalibrationTiles" in body
    assert b"existing tile(s) were captured too far apart" in body
    assert b"Copy both FarmBot camera offsets unchanged" in body
    assert b"All photos are still shown at their reported size" in body
    assert b"No loaded photos match the selected FarmBot camera resolution" not in body

    status, _, body = await asgi_request("/health")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


@pytest.mark.asyncio
async def test_normal_path_is_unchanged_and_query_string_survives():
    status, _, body = await asgi_request("/api/health", query_string=b"check=//")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


@pytest.mark.asyncio
async def test_post_duplicate_path_works(monkeypatch: pytest.MonkeyPatch):
    async def fake_run(*args, **kwargs):
        return {"accepted": True}

    monkeypatch.setattr(web.jobs, "run", fake_run)
    status, headers, _ = await asgi_request("//analyse", method="POST")
    assert status == 303
    assert headers[b"location"] == b"./"


@pytest.mark.asyncio
async def test_approval_json_reports_rejection_without_recording_a_false_success(monkeypatch):
    measurement = _review_measurement()
    web.database.save_measurements([measurement])
    calls = []

    async def rejected(request):
        calls.append(request)
        return {"status": "rejected", "message": "FarmBot declined this change"}

    monkeypatch.setattr(web.client, "apply_radius", rejected)
    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body) == {
        "status": "rejected",
        "message": "FarmBot declined this change",
    }
    assert calls[0].human_approved is True
    decisions = [
        row
        for row in web.database.recent_decisions()
        if row["measurement_id"] == str(measurement.measurement_id)
    ]
    assert decisions == []


@pytest.mark.asyncio
async def test_approval_json_records_applied_and_html_post_still_redirects(monkeypatch):
    json_measurement = _review_measurement()
    html_measurement = _review_measurement()
    web.database.save_measurements([json_measurement, html_measurement])

    async def applied(_request):
        return {"status": "applied", "message": "radius updated"}

    monkeypatch.setattr(web.client, "apply_radius", applied)
    status, _, body = await asgi_request(
        f"/recommendations/{json_measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 200
    assert json.loads(body)["status"] == "applied"
    decisions = [
        row
        for row in web.database.recent_decisions()
        if row["measurement_id"] == str(json_measurement.measurement_id)
    ]
    assert [row["action"] for row in decisions] == ["applied"]

    status, headers, _ = await asgi_request(
        f"/recommendations/{html_measurement.measurement_id}/approve", method="POST"
    )
    assert status == 303
    assert b"location" in headers


@pytest.mark.asyncio
async def test_rejecting_a_recommendation_unlinks_its_image_from_the_plant():
    # A photo taken at the wrong coordinate can end up analysed for a plant it
    # never actually shows. Rejecting the bad recommendation must stop that
    # image from being treated as evidence for this plant without deleting it
    # (other plants, and the whole-garden mosaic, may still reference it).
    measurement = _review_measurement(plant_id=4001, image_id=77)
    web.database.save_measurements([measurement])
    assert web.database.unlinked_image_ids("review-bot", 4001) == set()

    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/reject",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body)["status"] == "rejected"
    assert web.database.unlinked_image_ids("review-bot", 4001) == {77}
    # A different plant that legitimately used the same image is unaffected.
    assert web.database.unlinked_image_ids("review-bot", 4002) == set()


@pytest.mark.asyncio
async def test_keeping_a_flagged_removal_unlinks_its_image_from_the_plant(monkeypatch):
    measurement = _review_measurement(
        plant_id=4010,
        image_id=88,
        vegetation_absent=True,
        absent_observations=2,
        current_radius_mm=40,
        recommended_protection_radius_mm=0,
    )
    web.database.save_measurements([measurement])

    status, _, body = await asgi_request(
        f"/removals/{measurement.measurement_id}/keep",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body)["status"] == "rejected"
    assert web.database.unlinked_image_ids("review-bot", 4010) == {88}


@pytest.mark.asyncio
async def test_dashboard_recommendation_row_carries_plant_photo_attributes():
    measurement = _review_measurement(
        plant_id=4020,
        image_id=99,
        recorded_center_x=123.4,
        recorded_center_y=567.8,
    )
    web.database.save_measurements([measurement])
    web.database.unlink_plant_images("review-bot", 4020, [55])

    status, _, body = await asgi_request("/")
    assert status == 200
    html = body.decode()
    assert 'data-plant-id="4020"' in html
    assert 'data-plant-x="123.4"' in html
    assert 'data-plant-y="567.8"' in html
    assert 'data-unlinked-images="[55]"' in html


@pytest.mark.asyncio
async def test_uncertain_measurement_is_manually_reviewable_and_applicable(monkeypatch):
    measurement = _review_measurement(
        decision=Decision.UNCERTAIN,
        reason="overlap lowers confidence for automation",
    )
    web.database.save_measurements([measurement])
    calls = []

    async def applied(request):
        calls.append(request)
        return {"status": "applied", "message": "radius verified"}

    monkeypatch.setattr(web.client, "apply_radius", applied)
    status, _, dashboard = await asgi_request("/")
    assert status == 200
    html = dashboard.decode()
    assert str(measurement.measurement_id) in html
    assert "Not reviewable" not in html
    assert "Apply radius" in html

    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 200
    assert json.loads(body)["status"] == "applied"
    assert calls[0].human_approved is True


@pytest.mark.asyncio
async def test_manual_approval_can_apply_a_radius_reduction(monkeypatch):
    measurement = _review_measurement(
        current_radius_mm=100,
        recommended_protection_radius_mm=90,
        confidence=0.5,
        decision=Decision.UNCERTAIN,
        reason="large radius reduction requires human review",
    )
    web.database.save_measurements([measurement])
    calls = []

    async def applied(request):
        calls.append(request)
        return {"status": "applied", "message": "radius updated"}

    monkeypatch.setattr(web.client, "apply_radius", applied)
    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body)["status"] == "applied"
    assert calls[0].recommended_radius_mm == 90
    assert calls[0].human_approved is True


@pytest.mark.asyncio
async def test_dashboard_modal_uses_artifact_manifest_and_pending_rows(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    overlay = artifact_dir / "review-overlay.jpg"
    mask = artifact_dir / "review-mask.png"
    overlay.write_bytes(b"overlay")
    mask.write_bytes(b"mask")
    measurement = _review_measurement(
        overlay_path=str(overlay),
        mask_path=str(mask),
        artifact_paths=[str(overlay), str(mask)],
    )
    web.database.save_measurements([measurement])
    weed_id = str(uuid4())
    web.database.save_weed_detection(
        detection_id=weed_id,
        config_entry_id="dashboard-bot",
        image_id=42,
        image_timestamp=datetime.now(UTC),
        x=10,
        y=20,
        z=0,
        area_mm2=80,
        radius_mm=15,
        confidence=0.9,
        overlay_path=str(overlay),
        review_path=str(overlay),
    )
    monkeypatch.setattr(web.settings, "data_dir", tmp_path)

    status, _, body = await asgi_request("/")
    html = body.decode()
    assert status == 200
    assert "id=overlay-modal" in html
    assert "artifact/review-overlay.jpg" in html
    assert "artifact/review-mask.png" not in html
    assert "Previous weed" in html and "Next weed" in html
    assert "Previous image" in html and "Next image" in html
    assert "Reject as" in html
    for label in ("Crop", "Fallen leaf", "Mushroom", "Moss", "Soil", "Hardware"):
        assert label in html
    assert "Review / training label" not in html
    assert '<button class=review-action data-url="weeds/' not in html

    # The modal still relies on the deliberately restricted artifact route.
    status, _, served = await asgi_request("/artifact/review-overlay.jpg")
    assert status == 200
    assert served == b"overlay"

    web.database.record_decision(str(measurement.measurement_id), "applied", {})
    assert str(measurement.measurement_id) not in {
        row["measurement_id"] for row in web.database.pending_measurements()
    }


@pytest.mark.asyncio
async def test_dashboard_weed_modal_offers_unknown_and_close_up_controls(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    overlay = artifact_dir / "closeup-overlay.jpg"
    overlay.write_bytes(b"overlay")
    web.database.save_weed_detection(
        detection_id=str(uuid4()),
        config_entry_id="dashboard-bot",
        image_id=7,
        image_timestamp=datetime.now(UTC),
        x=10,
        y=20,
        z=0,
        area_mm2=80,
        radius_mm=15,
        confidence=0.9,
        overlay_path=str(overlay),
        review_path=str(overlay),
    )
    monkeypatch.setattr(web.settings, "data_dir", tmp_path)

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert "id=weed-modal-unknown" in html
    assert "id=weed-modal-closeup" in html
    assert "id=weed-modal-zoom-level" in html
    assert "data-weed-label=fallen_leaf" in html
    assert "teaches the verifier that it is not a weed" in html


@pytest.mark.asyncio
async def test_fallen_leaf_label_rejects_detection_and_records_hard_negative():
    detection_id = str(uuid4())
    web.database.save_weed_detection(
        detection_id=detection_id,
        config_entry_id="fallen-leaf-bot",
        image_id=8,
        image_timestamp=datetime.now(UTC),
        x=40,
        y=70,
        z=0,
        area_mm2=55,
        radius_mm=12,
        confidence=0.85,
        features={"strong_green_fraction": 0.9},
        overlay_path=None,
    )

    status, _, body = await asgi_request(f"/weeds/{detection_id}/label/fallen_leaf", method="POST")

    assert status == 200
    assert json.loads(body)["status"] == "applied"
    assert web.database.weed_detection(detection_id)["status"] == "rejected"
    sample = next(
        row for row in web.database.weed_training_samples() if row["detection_id"] == detection_id
    )
    assert sample["label"] == "fallen_leaf"
    with web.database.connection:
        web.database.connection.execute(
            "DELETE FROM weed_training_samples WHERE detection_id=?", (detection_id,)
        )


@pytest.mark.asyncio
async def test_dismiss_weed_suppresses_position_without_recording_a_label():
    detection_id = str(uuid4())
    web.database.save_weed_detection(
        detection_id=detection_id,
        config_entry_id="unknown-bot",
        image_id=11,
        image_timestamp=datetime.now(UTC),
        x=100,
        y=200,
        z=0,
        area_mm2=60,
        radius_mm=14,
        confidence=0.8,
        overlay_path=None,
    )

    status, _, body = await asgi_request(f"/weeds/{detection_id}/dismiss", method="POST")

    assert status == 200
    assert json.loads(body)["status"] == "dismissed"
    # Neither accepted nor rejected: it leaves the queue, teaches the verifier
    # nothing, and never comes back at the same position.
    assert web.database.weed_detection(detection_id)["status"] == "dismissed"
    assert detection_id not in {
        row["detection_id"] for row in web.database.pending_weed_detections()
    }
    assert web.database.weed_training_samples() == []
    assert web.database.has_terminal_weed_detection_near("unknown-bot", 100, 200, 20) is True


@pytest.mark.asyncio
async def test_dismiss_weed_rejects_an_unknown_detection():
    status, _, _ = await asgi_request(f"/weeds/{uuid4()}/dismiss", method="POST")

    assert status == 404


@pytest.mark.asyncio
async def test_weed_settings_page_exposes_pipeline_training_and_automation_controls():
    status, _, body = await asgi_request("/weed-settings")
    html = body.decode()

    assert status == 200
    assert "How big a weed to look for" in html
    assert "What counts as green foliage" in html
    assert "What shape a weed may be" in html
    assert "Known crop protection" in html
    assert "Multi-image confirmation" in html
    assert "Learned visual verifier" in html
    assert 'name="automatic_creation_confidence"' in html
    assert 'action="weed-model/train"' in html
    assert 'action="weed-model/clear"' in html
    assert 'action="weed-model/export"' in html
    assert 'action="weed-model/import"' in html
    assert "Clear all training images" in html
    assert "Most informative to label next" in html
    assert 'name="candidate_recall_boost"' in html


@pytest.mark.asyncio
async def test_every_weed_setting_is_explained_and_has_a_slider_or_toggle():
    """A threshold nobody can interpret cannot be calibrated, so none may ship bare."""
    from farmbot_vision.weed_settings import WeedSettings

    status, _, body = await asgi_request("/weed-settings")
    html = body.decode()

    assert status == 200
    for name, field in WeedSettings.model_fields.items():
        assert f'name="{name}"' in html, f"{name} is missing from the settings form"
        if field.annotation is bool:
            assert f'<input type=checkbox name="{name}"' in html, f"{name} needs a checkbox"
        else:
            assert f'type=range aria-hidden=true tabindex=-1 data-sync="{name}"' in html, (
                f"{name} needs a slider paired with its number box"
            )
    # One tooltip per setting, plus the one introducing the convention.
    assert html.count("class=hint") >= len(WeedSettings.model_fields)


@pytest.mark.asyncio
async def test_weed_settings_offer_visual_pickers_for_colour_shape_and_size():
    status, _, body = await asgi_request("/weed-settings")
    html = body.decode()

    assert status == 200
    # Hue is chosen against a colour band and can be sampled from a real leaf.
    assert "class=hue-band" in html
    assert "type=color id=hue-picker" in html
    assert "Centre the range on this colour" in html
    # Area limits are shown as the physical width of a blob, not bare mm².
    assert "id=size-blob-min" in html
    assert "mm across" in html
    # Solidity and circularity are shown as shapes.
    assert "class=shape-figs" in html
    assert "Low solidity" in html
    assert "High circularity" in html


@pytest.mark.asyncio
async def test_editing_a_training_tag_saves_and_redirects_back_to_the_settings_page(monkeypatch):
    """The nested sample route needs two levels up, or the redirect 404s."""
    detection_id = str(uuid4())
    web.database.save_weed_detection(
        detection_id=detection_id,
        config_entry_id="tag-bot",
        image_id=7,
        image_timestamp=datetime.now(UTC),
        x=10,
        y=20,
        z=0,
        area_mm2=80,
        radius_mm=15,
        confidence=0.9,
        overlay_path=None,
    )
    assert web.database.label_weed_detection(detection_id, "weed")
    manual = web.weed_settings_store.load().model_copy(update={"automatic_retraining": False})
    monkeypatch.setattr(web.weed_settings_store, "load", lambda: manual)

    status, headers, _ = await asgi_request(
        f"/weed-model/samples/{detection_id}",
        method="POST",
        form={"label": "mulch_soil"},
    )

    assert status == 303
    location = headers[b"location"].decode()
    assert location.split("?")[0] == "../../weed-settings"
    assert "Updated training tag to mulch/soil" in unquote(location)

    # The redirect target must resolve to the settings page, not /weed-model/....
    resolved = urljoin(f"http://testserver/weed-model/samples/{detection_id}", location)
    assert urlsplit(resolved).path == "/weed-settings"

    labels = {
        sample["detection_id"]: sample["label"] for sample in web.database.weed_training_samples()
    }
    assert labels[detection_id] == "mulch_soil"


def _store_labelled_detection(detection_id: str, label: str, image_id: int = 11) -> None:
    web.database.save_weed_detection(
        detection_id=detection_id,
        config_entry_id="export-bot",
        image_id=image_id,
        image_timestamp=datetime.now(UTC),
        x=10,
        y=20,
        z=0,
        area_mm2=80,
        radius_mm=15,
        confidence=0.9,
        overlay_path=None,
        features={name: 0.3 for name in weed_verifier_module.FEATURE_NAMES},
    )
    assert web.database.label_weed_detection(detection_id, label)


@pytest.mark.asyncio
async def test_training_bundle_exports_labels_with_their_features():
    detection_id = str(uuid4())
    _store_labelled_detection(detection_id, "moss")

    status, headers, body = await asgi_request("/weed-model/export")
    bundle = json.loads(body)

    assert status == 200
    assert b"attachment" in headers[b"content-disposition"]
    exported = {sample["detection_id"]: sample for sample in bundle["samples"]}
    assert exported[detection_id]["label"] == "moss"
    # Features are what training consumes, so they are the payload that makes
    # a bundle portable between installs.
    assert exported[detection_id]["features"]["strong_green_fraction"] == 0.3
    assert exported[detection_id]["image_id"] == 11
    # Crops are omitted unless explicitly requested.
    assert "crop_base64" not in exported[detection_id]
    assert bundle["feature_names"] == list(weed_verifier_module.FEATURE_NAMES)


@pytest.mark.asyncio
async def test_a_bundle_round_trips_back_into_an_empty_install(monkeypatch):
    detection_id = str(uuid4())
    _store_labelled_detection(detection_id, "mushroom")
    manual = web.weed_settings_store.load().model_copy(update={"automatic_retraining": False})
    monkeypatch.setattr(web.weed_settings_store, "load", lambda: manual)

    _, _, body = await asgi_request("/weed-model/export")
    web.database.clear_weed_training_samples()
    assert web.database.weed_training_samples() == []

    boundary = "----bundleboundary"
    payload = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="bundle"; filename="bundle.json"\r\n'
            "Content-Type: application/json\r\n\r\n"
        ).encode()
        + body
        + f"\r\n--{boundary}--\r\n".encode()
    )

    status, headers, _ = await asgi_request(
        "/weed-model/import",
        method="POST",
        raw_body=payload,
        content_type=f"multipart/form-data; boundary={boundary}",
    )

    assert status == 303
    assert "Imported" in unquote(headers[b"location"].decode())
    restored = {sample["detection_id"]: sample for sample in web.database.weed_training_samples()}
    assert restored[detection_id]["label"] == "mushroom"
    assert restored[detection_id]["features"]["strong_green_fraction"] == 0.3


@pytest.mark.asyncio
async def test_importing_a_bundle_with_no_usable_samples_is_rejected():
    boundary = "----emptyboundary"
    payload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="bundle"; filename="bundle.json"\r\n'
        "Content-Type: application/json\r\n\r\n"
        '{"samples": [{"label": "not_a_label", "features": {}, "detection_id": "x"}]}'
        f"\r\n--{boundary}--\r\n"
    ).encode()

    status, _, _ = await asgi_request(
        "/weed-model/import",
        method="POST",
        raw_body=payload,
        content_type=f"multipart/form-data; boundary={boundary}",
    )

    assert status == 422


@pytest.mark.asyncio
async def test_the_suggested_threshold_can_be_applied_in_one_click(monkeypatch):
    monkeypatch.setattr(web.weed_verifier, "reload", lambda: None)
    monkeypatch.setattr(type(web.weed_verifier), "suggested_threshold", property(lambda self: 0.78))
    saved: list[float] = []
    monkeypatch.setattr(
        web.weed_settings_store,
        "save",
        lambda values: saved.append(values.visual_verifier_minimum_confidence),
    )

    status, headers, _ = await asgi_request("/weed-model/apply-threshold", method="POST")

    assert status == 303
    assert saved == [0.78]
    assert "0.78" in unquote(headers[b"location"].decode())


@pytest.mark.asyncio
async def test_best_guess_names_categories_in_plain_language(monkeypatch):
    monkeypatch.setattr(
        web.weed_verifier,
        "explain",
        lambda features: [("fallen_leaf", 0.7), ("weed", 0.3)],
    )

    guess = web._verifier_label_guess({"strong_green_fraction": 0.4})

    assert guess[0]["label"] == "fallen leaf"
    assert guess[0]["probability"] == 0.7


@pytest.mark.asyncio
async def test_canopy_settings_page_exposes_fusion_and_automation_controls():
    status, _, body = await asgi_request("/canopy-settings")
    html = body.decode()

    assert status == 200
    assert "Multi-image canopy fusion" in html
    assert "name=always_fuse_when_available" in html
    assert "name=minimum_angular_coverage" in html
    assert "name=automatic_requires_reliable_fusion" in html
    assert "name=save_diagnostics" in html


@pytest.mark.asyncio
async def test_dashboard_plant_view_uses_only_clean_and_mask_composites(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    clean = artifact_dir / "plant-composite.jpg"
    composite_overlay = artifact_dir / "plant-composite-overlay.jpg"
    diagnostic_overlay = artifact_dir / "frame-overlay.jpg"
    raw_mask = artifact_dir / "frame-mask.png"
    for path in (clean, composite_overlay, diagnostic_overlay, raw_mask):
        path.write_bytes(b"image")
    measurement = _review_measurement(
        composite_path=str(clean),
        composite_overlay_path=str(composite_overlay),
        overlay_path=str(diagnostic_overlay),
        mask_path=str(raw_mask),
        artifact_paths=[str(diagnostic_overlay), str(raw_mask)],
        recorded_center_x=100,
        recorded_center_y=200,
    )
    web.database.save_measurements([measurement])
    monkeypatch.setattr(web.settings, "data_dir", tmp_path)

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert 'data-composite-clean="artifact/plant-composite.jpg"' in html
    assert 'data-composite-overlay="artifact/plant-composite-overlay.jpg"' in html
    assert "Standard view" in html
    assert "Diagnostic mask" in html
    assert "artifact/frame-overlay.jpg" not in html
    assert "artifact/frame-mask.png" not in html


@pytest.mark.asyncio
async def test_dashboard_plant_without_composite_never_reuses_frame_artifacts(
    tmp_path, monkeypatch
):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    diagnostic_overlay = artifact_dir / "unrelated-frame-overlay.jpg"
    raw_mask = artifact_dir / "unrelated-frame-mask.png"
    diagnostic_overlay.write_bytes(b"overlay")
    raw_mask.write_bytes(b"mask")
    measurement = _review_measurement(
        composite_path=None,
        composite_overlay_path=None,
        overlay_path=str(diagnostic_overlay),
        mask_path=str(raw_mask),
        artifact_paths=[str(diagnostic_overlay), str(raw_mask)],
        recorded_center_x=100,
        recorded_center_y=200,
    )
    web.database.save_measurements([measurement])
    monkeypatch.setattr(web.settings, "data_dir", tmp_path)

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert 'data-artifacts="[]"' in html
    assert "artifact/unrelated-frame-overlay.jpg" not in html
    assert "artifact/unrelated-frame-mask.png" not in html


@pytest.mark.asyncio
async def test_dashboard_plant_without_composite_shows_its_source_and_frame_artifacts(
    tmp_path, monkeypatch
):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    source = artifact_dir / "source-photo.jpg"
    overlay = artifact_dir / "source-overlay.jpg"
    source.write_bytes(b"source")
    overlay.write_bytes(b"overlay")
    measurement = _review_measurement(
        source_image_path=str(source),
        artifact_paths=[str(overlay)],
        recorded_center_x=100,
        recorded_center_y=200,
    )
    web.database.save_measurements([measurement])
    monkeypatch.setattr(web.settings, "data_dir", tmp_path)

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert "artifact/source-photo.jpg" in html
    assert "artifact/source-overlay.jpg" in html


@pytest.mark.asyncio
async def test_missing_plant_table_shows_crop_and_center_coordinates():
    measurement = _review_measurement(
        vegetation_absent=True,
        center_misaligned=True,
        recorded_center_x=123.4,
        recorded_center_y=567.8,
        recommended_center_px=(130.5, 575.25),
        absent_observations=2,
    )
    web.database.save_measurements([measurement])

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert "<th>Crop</th>" in html
    assert "Recorded center (X, Y mm)" in html
    assert "Move center to (X, Y mm)" in html
    assert "<td>lettuce</td>" in html
    assert "<td>X 123.4, Y 567.8</td>" in html
    assert "<td>X 130.5, Y 575.2</td>" in html
    assert "<th>Plant</th><th>Absent looks</th>" not in html


@pytest.mark.asyncio
async def test_removal_approval_uses_fresh_inventory_radius(monkeypatch):
    measurement = _review_measurement(
        vegetation_absent=True,
        absent_observations=2,
        current_radius_mm=40,
        recommended_protection_radius_mm=0,
    )
    web.database.save_measurements([measurement])
    calls = []

    async def inventory(_request):
        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": datetime.now(UTC),
                "plants": [
                    {
                        "id": measurement.plant_id,
                        "name": "Lettuce",
                        "openfarm_slug": "lettuce",
                        "x": 100,
                        "y": 200,
                        "radius": 55,
                        "plant_stage": "planted",
                    }
                ],
                "images": [],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    async def apply_removal(request):
        calls.append(request)
        return {"status": "applied", "message": "Plant removed"}

    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "apply_removal", apply_removal)

    status, _, body = await asgi_request(
        f"/removals/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body)["status"] == "applied"
    assert calls[0].expected_current_radius_mm == 55
    assert calls[0].human_approved is True


@pytest.mark.asyncio
async def test_event_listener_targets_new_image_and_uses_configured_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    async def events():
        yield VisionRequestEvent(config_entry_id="entry-1", device_id="device_42", image_id=99)

    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        return {"accepted": True}

    monkeypatch.setattr(web.client, "vision_events", events)
    monkeypatch.setattr(web.jobs, "run", fake_run)
    monkeypatch.setattr(web.settings, "mode", web.OperatingMode.RECOMMEND)
    await web.event_listener()
    assert calls == [
        {
            "entry_id": "entry-1",
            "mode": web.OperatingMode.RECOMMEND,
            "plant_ids": [],
            "image_ids": [99],
            "trigger": "new_image",
            "queue_if_busy": True,
        }
    ]


@pytest.mark.asyncio
async def test_event_listener_batches_burst_images_into_one_job(
    monkeypatch: pytest.MonkeyPatch,
):
    async def events():
        for image_id in (97, 98, 99, 98):
            yield VisionRequestEvent(
                config_entry_id="entry-1",
                device_id="device_42",
                image_id=image_id,
            )

    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        return {"accepted": True}

    monkeypatch.setattr(web.client, "vision_events", events)
    monkeypatch.setattr(web.jobs, "run", fake_run)
    monkeypatch.setattr(web.settings, "mode", web.OperatingMode.RECOMMEND)

    await web.event_listener()

    assert len(calls) == 1
    assert calls[0]["image_ids"] == [97, 98, 99]
    assert calls[0]["plant_ids"] == []


@pytest.mark.asyncio
async def test_completed_grid_hands_off_verified_images_and_deduplicates_late_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from farmbot_vision.photo_grid import PhotoGridFrame, PhotoGridStore, PhotoGridTarget

    record = _photo_grid_record(
        [PhotoGridTarget(index=0, row=0, column=0, x=100, y=200, z=0)],
        tmp_path=tmp_path,
    )
    record.frames = [
        PhotoGridFrame(target_index=0, image_id=10, x=100, y=200, z=0),
        PhotoGridFrame(target_index=1, image_id=11, x=300, y=200, z=0),
    ]
    record.quality_overlay_frames = [PhotoGridFrame(target_index=1, image_id=12, x=300, y=200, z=0)]
    record.excluded_image_ids = [11]
    store = PhotoGridStore(tmp_path / "grid.json")
    store.save(record)
    monkeypatch.setattr(web, "photo_grid_store", store)
    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        return {"accepted": True, "status": "idle", "images_processed": 2}

    monkeypatch.setattr(web.jobs, "run", fake_run)
    monkeypatch.setattr(web.settings, "mode", web.OperatingMode.RECOMMEND)

    await web._analyse_completed_photo_grid(record)

    assert calls[0]["image_ids"] == [10, 12]
    assert calls[0]["trigger"] == "photo_grid"
    assert store.load().analysis_handoff_image_ids == [10, 12]
    assert (
        await web._eligible_vision_event(VisionRequestEvent(config_entry_id="entry-1", image_id=10))
        is None
    )
    assert (
        await web._eligible_vision_event(VisionRequestEvent(config_entry_id="entry-1", image_id=11))
        is None
    )


@pytest.mark.asyncio
async def test_grid_analyzes_quality_cleared_frames_before_slow_repairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from farmbot_vision.photo_grid import (
        PhotoGridCellAnalysis,
        PhotoGridFrame,
        PhotoGridStore,
        PhotoGridTarget,
    )

    targets = [
        PhotoGridTarget(index=0, row=0, column=0, x=100, y=200, z=0),
        PhotoGridTarget(index=1, row=0, column=1, x=300, y=200, z=0),
    ]
    record = _photo_grid_record(targets, tmp_path=tmp_path)
    record.frames = [
        PhotoGridFrame(target_index=0, image_id=10, x=100, y=200, z=0),
        PhotoGridFrame(target_index=1, image_id=11, x=300, y=200, z=0),
    ]
    record.cell_analysis = [
        PhotoGridCellAnalysis(
            target_index=0,
            image_id=10,
            issue="usable",
            analysed_at=datetime.now(UTC),
        ),
        PhotoGridCellAnalysis(
            target_index=1,
            image_id=11,
            issue="leaf_obstruction",
            analysed_at=datetime.now(UTC),
        ),
    ]
    store = PhotoGridStore(tmp_path / "grid.json")
    store.save(record)
    monkeypatch.setattr(web, "photo_grid_store", store)

    async def empty_capture(*_args):
        return []

    monkeypatch.setattr(web, "_capture_photo_grid_targets", empty_capture)

    calls = []

    async def fake_analysis(_record, image_ids=None):
        calls.append(("analysis", image_ids))

    async def fake_quality(_record, _scout=None):
        calls.append(("quality", None))

    async def fake_targeted(_record):
        calls.append(("targeted", None))

    async def no_scout(_scout):
        return None

    monkeypatch.setattr(web, "_analyse_completed_photo_grid", fake_analysis)
    monkeypatch.setattr(web, "_photo_grid_quality_pass", fake_quality)
    monkeypatch.setattr(web, "_capture_targeted_plant_photos", fake_targeted)
    monkeypatch.setattr(web._LiveGridScout, "run", no_scout)

    await web._photo_grid_worker(record)

    assert calls[:2] == [("analysis", [10]), ("quality", None)]


@pytest.mark.asyncio
async def test_startup_auto_selects_only_loaded_farmbot(monkeypatch: pytest.MonkeyPatch):
    async def list_bots():
        return BotList.model_validate(
            {"bots": [{"config_entry_id": "entry-1", "device_id": "42", "name": "FarmBot"}]}
        )

    monkeypatch.setattr(web.settings, "selected_config_entry_id", "")
    monkeypatch.setattr(web.client, "list_bots", list_bots)
    await web.resolve_config_entry()
    assert web.settings.selected_config_entry_id == "entry-1"


@pytest.mark.asyncio
async def test_ingress_html_uses_relative_links_without_logging_session(
    caplog: pytest.LogCaptureFixture,
):
    ingress_path = "/api/hassio_ingress/temporary-session-id/"
    status, _, body = await asgi_request(
        "/",
        headers=[(b"x-ingress-path", ingress_path.encode("ascii"))],
    )
    html = body.decode()
    assert status == 200
    assert f'<base href="{ingress_path}">' in html
    assert 'href="/settings"' not in html
    assert 'href="/api/health"' not in html
    assert 'action="/analyse"' not in html
    assert "//" not in html.replace("http://", "")
    assert ingress_path not in caplog.text

    _, _, settings_body = await asgi_request(
        "/settings",
        headers=[(b"x-ingress-path", ingress_path.encode("ascii"))],
    )
    settings_html = settings_body.decode()
    assert "fetch('api/vision/images" in settings_html
    assert "fetch(url).then" in settings_html
    assert "X-FarmBot-Oriented-Width" in settings_html
    assert "f.action='calibration'" in settings_html
    assert 'href="/settings"' not in settings_html


@pytest.mark.asyncio
async def test_mobile_review_uses_one_responsive_geometry_for_standard_and_mask_views():
    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert ">Standard view</button>" in html
    assert ">Diagnostic mask</button>" in html
    assert "id=plant-photo-canvas" not in html
    assert "id=plant-modal-without-overlay" not in html
    assert "id=plant-modal-with-overlay" not in html
    assert ".overlay-modal img{display:block;width:100%;height:auto" in html
    assert "@media (max-width:600px)" in html
    assert "if(plantComposite) showPlantComposite(!showPhoto);" in html


def test_direct_asgi_middleware_normalizes_scope_without_touching_query():
    captured: dict = {}

    async def downstream(scope, receive, send):
        captured.update(scope)

    middleware = web.NormalizeIngressPathMiddleware(downstream)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "path": "//api/health",
        "raw_path": b"//api/health",
        "query_string": b"token=//preserve",
        "headers": [(b"x-ingress-path", b"/api/hassio_ingress/session")],
    }
    import asyncio

    asyncio.run(middleware(scope, receive, send))
    assert captured["path"] == "/api/health"
    assert captured["raw_path"] == b"/api/health"
    assert captured["query_string"] == b"token=//preserve"
    assert captured["headers"] == scope["headers"]


@pytest.mark.asyncio
async def test_save_calibration_farmbot_values_branch():
    from farmbot_vision.models import OriginLocation

    response = await web.save_calibration(
        entry_id="botFB",
        rotation=-31.9,
        offset_x=0,
        offset_y=0,
        origin_location="top_right",
        coordinate_scale=0.242,
        reference_width=2592,
        reference_height=1944,
    )
    assert response.status_code == 303
    calibration = web.database.active_calibration("botFB")
    assert calibration is not None
    assert calibration.source == "manual"
    assert calibration.origin_location == OriginLocation.TOP_RIGHT
    assert calibration.rotation_degrees == -31.9
    width = web.settings.resolution.width
    assert calibration.pixels_per_mm_x == pytest.approx((1 / 0.242) * width / 2592)


@pytest.mark.asyncio
async def test_save_calibration_persists_to_data_store():
    """Saved FarmBot inputs are written verbatim to the durable /data store."""
    await web.save_calibration(
        entry_id="botStore",
        rotation=12.0,
        offset_x=3,
        offset_y=-4,
        origin_location="bottom_left",
        map_origin="top_right",
        rotate_map=True,
        coordinate_scale=0.3,
        reference_width=2592,
        reference_height=1944,
    )
    stored = web.calibration_store.get("botStore")
    assert stored is not None
    assert stored.coordinate_scale == 0.3
    assert stored.rotation_degrees == 12.0
    assert stored.offset_x_mm == 3
    assert str(stored.origin_location) == "bottom_left"
    assert str(stored.map_origin) == "top_right"
    assert stored.rotate_map is True


@pytest.mark.asyncio
async def test_calibration_grid_filters_dates_and_keeps_newest_per_location(monkeypatch, tmp_path):
    from farmbot_vision.photo_grid import PhotoGridStore

    now = datetime.now(UTC)

    async def inventory(_request):
        def image(image_id, minutes_ago, x, y):
            return {
                "id": image_id,
                "created_at": now - timedelta(minutes=minutes_ago),
                "x": x,
                "y": y,
                "z": 0,
            }

        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": now,
                "plants": [],
                "weeds": [],
                "images": [
                    image(1, 20, 100, 200),
                    image(2, 10, 104, 203),
                    image(3, 5, 400, 500),
                    image(4, 120, 700, 800),
                ],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    async def soil_points(_entry_id):
        raise web.HomeAssistantError("motion bounds unavailable")

    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "soil_points", soil_points)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    response = await web.vision_images(
        entry_id="bot-grid",
        date_from=now - timedelta(hours=1),
        date_to=now,
    )
    payload = json.loads(response.body)

    assert [image["id"] for image in payload["images"]] == [3, 2]
    assert payload["bed_bounds"] is None


@pytest.mark.asyncio
async def test_latest_grid_keeps_snapshotted_weed_markers_when_inventory_is_offline(
    monkeypatch, tmp_path
):
    from farmbot_vision.calibration_store import FarmbotCalibrationInput
    from farmbot_vision.photo_grid import (
        KnownMapPoint,
        PhotoGridRecord,
        PhotoGridStore,
        PhotoGridTarget,
    )

    store = PhotoGridStore(tmp_path / "grid.json")
    store.save(
        PhotoGridRecord(
            config_entry_id="bot-grid",
            started_at=datetime.now(UTC),
            bed_bounds={"x": (0, 1000), "y": (0, 800)},
            footprint_width_mm=500,
            footprint_height_mm=400,
            calibration=FarmbotCalibrationInput(
                coordinate_scale=0.25,
                reference_width=2592,
                reference_height=1944,
            ),
            targets=[PhotoGridTarget(index=0, row=0, column=0, x=250, y=200, z=0)],
            known_points=[KnownMapPoint(id=91, kind="weed", name="", x=125, y=275, radius=22)],
        )
    )

    async def inventory(_request):
        raise web.HomeAssistantError("offline")

    monkeypatch.setattr(web, "photo_grid_store", store)
    monkeypatch.setattr(web.client, "inventory", inventory)

    response = await web.latest_calibrated_photo_grid()
    payload = json.loads(response.body)

    assert payload["weeds"] == [
        {
            "id": 91,
            "kind": "weed",
            "name": "",
            "x": 125.0,
            "y": 275.0,
            "radius": 22.0,
        }
    ]


@pytest.mark.asyncio
async def test_calibration_grid_prefers_live_axis_bounds(monkeypatch):
    from types import SimpleNamespace

    now = datetime.now(UTC)

    async def inventory(_request):
        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": now,
                "plants": [],
                "weeds": [],
                "images": [
                    {
                        "id": 1,
                        "created_at": now,
                        "x": 100,
                        "y": 200,
                        "z": 0,
                    }
                ],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    async def soil_points(_entry_id):
        return SoilPointInventory(
            device_id="42",
            generated_at=now,
            points=[],
            motion=SoilMotionState(
                connected=True,
                busy=False,
                locked=False,
                position={"x": 0, "y": 0, "z": 0},
                z_direction=-1,
                axis_bounds={"x": (0, 4180), "y": (0, 2164), "z": (-400, 0)},
            ),
        )

    stale_record = SimpleNamespace(
        config_entry_id="bot-grid",
        bed_bounds={"x": (0, 1000), "y": (0, 800)},
    )
    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "soil_points", soil_points)
    monkeypatch.setattr(web.photo_grid_store, "load", lambda: stale_record)

    response = await web.vision_images(
        entry_id="bot-grid",
        date_from=now - timedelta(hours=1),
        date_to=now,
    )
    payload = json.loads(response.body)

    assert payload["bed_bounds"] == {"x": [0.0, 4180.0], "y": [0.0, 2164.0]}
    assert payload["bed_bounds_source"] == "live FarmBot axes"


@pytest.mark.asyncio
async def test_calibration_image_reports_actual_dimensions(monkeypatch, tmp_path):
    import base64
    from types import SimpleNamespace

    calls = 0

    async def image(_request, _max_bytes):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            image_base64=base64.b64encode(b"jpeg bytes").decode(),
            width=960,
            height=720,
            oriented_width=1280,
            oriented_height=960,
        )

    monkeypatch.setattr(web.client, "image", image)
    monkeypatch.setattr(web, "image_file_cache", web.ImageFileCache(tmp_path / "image-cache"))
    web._vision_image_locks.clear()
    response = await web.vision_image(entry_id="bot-grid", image_id=123)

    assert response.headers["x-farmbot-processed-width"] == "960"
    assert response.headers["x-farmbot-processed-height"] == "720"
    assert response.headers["x-farmbot-oriented-width"] == "1280"
    assert response.headers["x-farmbot-oriented-height"] == "960"
    assert response.headers["cache-control"] == "private, max-age=86400, immutable"
    assert response.body == b"jpeg bytes"

    status, headers, body = await asgi_request(
        "/api/vision/image/123.jpg",
        query_string=b"entry_id=bot-grid",
    )
    assert status == 200
    assert headers[b"cache-control"] == b"private, max-age=86400, immutable"
    assert body == b"jpeg bytes"
    assert calls == 1


@pytest.mark.asyncio
async def test_save_calibration_rejects_nonpositive_scale():
    with pytest.raises((web.HTTPException, ValueError)):
        await web.save_calibration(
            entry_id="botFB",
            coordinate_scale=0,
            reference_width=2592,
            reference_height=1944,
        )


@pytest.mark.asyncio
async def test_save_calibration_rejects_unknown_origin():
    with pytest.raises(web.HTTPException) as exc:
        await web.save_calibration(
            entry_id="botFB",
            origin_location="middle",
            coordinate_scale=0.242,
            reference_width=2592,
            reference_height=1944,
        )
    assert exc.value.status_code == 400


def test_app_config_uses_default_ingress_entry():
    config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    assert config["ingress"] is True
    assert config["ingress_port"] == 8099
    assert config["panel_icon"] == "mdi:sprout"
    assert config["panel_title"] == "FarmBot Vision"
    assert config["homeassistant_api"] is True
    assert "ingress_entry" not in config
