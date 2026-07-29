"""End-to-end route behaviour of the photo-grid worker.

These cover what the hardware logs exposed: a 77-cell grid that was cut into
twelve-cell calls, each of which drove back out to a staging position and
toggled the lighting, plus cells photographed a second time without any
recorded failure.
"""

from datetime import UTC, datetime

import pytest

from farmbot_vision import web
from farmbot_vision.calibration_store import FarmbotCalibrationInput
from farmbot_vision.photo_grid import (
    PHOTO_GRID_CONTINUOUS_MAX_TARGETS,
    PhotoGridRecord,
    PhotoGridStore,
    PhotoGridTarget,
)

BED_COLUMNS = 11
BED_ROWS = 7
X_SPACING = 373.0
Y_SPACING = 294.0


def _route(count=BED_COLUMNS * BED_ROWS, columns=BED_COLUMNS):
    """A canonical serpentine route with the real bed's spacing."""
    targets = []
    for index in range(count):
        row, position = divmod(index, columns)
        column = position if row % 2 == 0 else columns - 1 - position
        targets.append(
            PhotoGridTarget(
                index=index,
                row=row,
                column=column,
                x=235.0 + X_SPACING * column,
                y=195.0 + Y_SPACING * row,
                z=-1.0,
            )
        )
    return targets


def _record(targets, *, tmp_path, chunk_size=PHOTO_GRID_CONTINUOUS_MAX_TARGETS):
    return PhotoGridRecord(
        config_entry_id="entry-1",
        started_at=datetime.now(UTC),
        bed_bounds={"x": (0, 4180), "y": (0, 2164)},
        footprint_width_mm=450,
        footprint_height_mm=400,
        calibration=FarmbotCalibrationInput(
            coordinate_scale=0.25,
            reference_width=1800,
            reference_height=1600,
        ),
        targets=targets,
        chunk_size=chunk_size,
        indexed_targets=True,
    )


def _frames(targets):
    return [
        {
            "image_id": 100 + target.index,
            "target_index": target.index,
            "x": target.x,
            "y": target.y,
            "z": target.z,
        }
        for target in targets
    ]


@pytest.mark.asyncio
async def test_whole_bed_grid_is_sent_as_one_continuous_call(monkeypatch, tmp_path):
    """With a continuous-capable integration the 77-cell route is one call, so
    the run's lighting and its return to staging happen once, not seven times."""
    targets = _route()
    record = _record(targets, tmp_path=tmp_path)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    calls = []

    async def start(_entry_id, payload):
        calls.append(payload)
        return {"status": "queued", "repair_id": f"grid-{len(calls)}", "message": "queued"}

    async def status(_entry_id, _repair_id):
        return {"status": "complete", "message": "done", "frames": _frames(targets)}

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    await web._photo_grid_worker(record)

    assert len(calls) == 1
    assert [item["index"] for item in calls[0]] == list(range(77))
    # Exactly one capture operation per cell, each at its canonical coordinate.
    assert len(record.frames) == 77
    assert sorted(frame.target_index for frame in record.frames) == list(range(77))
    assert record.status == "complete"


@pytest.mark.asyncio
async def test_forced_small_batches_still_walk_the_canonical_route_once(monkeypatch, tmp_path):
    """A deliberately small batch limit must slice the route, not rebuild it."""
    targets = _route()
    record = _record(targets, tmp_path=tmp_path, chunk_size=5)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    batches = []

    async def start(_entry_id, payload):
        batches.append(payload)
        return {"status": "queued", "repair_id": f"grid-{len(batches)}", "message": "queued"}

    async def status(_entry_id, repair_id):
        payload = batches[int(repair_id.rsplit("-", 1)[1]) - 1]
        indexes = {item["index"] for item in payload}
        return {
            "status": "complete",
            "message": "done",
            "frames": _frames([t for t in targets if t.index in indexes]),
        }

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    await web._photo_grid_worker(record)

    rejoined = [item for batch in batches for item in batch]
    # Concatenating the batches reproduces the canonical route exactly: no
    # omissions, no overlaps, no renumbering and no reordering. Nothing but
    # grid cells is ever sent, so no staging, home or park coordinate can
    # appear between the first and last cell of the route.
    assert [item["index"] for item in rejoined] == list(range(77))
    assert [(item["x"], item["y"]) for item in rejoined] == [(t.x, t.y) for t in targets]
    assert len(rejoined) == len({item["index"] for item in rejoined}) == 77
    for previous, following in zip(batches, batches[1:], strict=False):
        assert following[0]["index"] == previous[-1]["index"] + 1
    # A five-cell limit lands boundaries mid-row without disturbing the route.
    assert batches[1][0]["index"] == 5
    assert record.status == "complete"


@pytest.mark.asyncio
async def test_grid_route_moves_only_along_rows_and_between_adjacent_rows(monkeypatch, tmp_path):
    """The coordinates actually sent contain no excursion out of the grid."""
    targets = _route()
    record = _record(targets, tmp_path=tmp_path, chunk_size=5)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    batches = []

    async def start(_entry_id, payload):
        batches.append(payload)
        return {"status": "queued", "repair_id": f"grid-{len(batches)}", "message": "queued"}

    async def status(_entry_id, repair_id):
        payload = batches[int(repair_id.rsplit("-", 1)[1]) - 1]
        indexes = {item["index"] for item in payload}
        return {
            "status": "complete",
            "message": "done",
            "frames": _frames([t for t in targets if t.index in indexes]),
        }

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    await web._photo_grid_worker(record)

    sent = [item for batch in batches for item in batch]
    legs = list(zip(sent, sent[1:], strict=False))
    lateral = [leg for leg in legs if leg[0]["y"] == leg[1]["y"]]
    transitions = [leg for leg in legs if leg[0]["y"] != leg[1]["y"]]

    assert len(lateral) == BED_ROWS * (BED_COLUMNS - 1)
    assert len(transitions) == BED_ROWS - 1
    assert all(abs(b["x"] - a["x"]) == X_SPACING for a, b in lateral)
    # Every row change steps to the adjacent row at the same end of the grid.
    assert all(a["x"] == b["x"] and abs(b["y"] - a["y"]) == Y_SPACING for a, b in transitions)

    route = sum(abs(b["x"] - a["x"]) + abs(b["y"] - a["y"]) for a, b in legs)
    assert route == BED_ROWS * (BED_COLUMNS - 1) * X_SPACING + (BED_ROWS - 1) * Y_SPACING
    assert route == 27874.0


@pytest.mark.asyncio
async def test_a_failed_cell_is_retried_without_recapturing_successful_cells(monkeypatch, tmp_path):
    targets = _route(count=6, columns=3)
    record = _record(targets, tmp_path=tmp_path)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    sent = []

    async def start(_entry_id, payload):
        sent.append([item["index"] for item in payload])
        return {"status": "queued", "repair_id": f"grid-{len(sent)}", "message": "queued"}

    async def status(_entry_id, repair_id):
        attempt = int(repair_id.rsplit("-", 1)[1])
        requested = set(sent[attempt - 1])
        # Cell 3's camera fails on the first pass and succeeds on the retry.
        captured = [
            target
            for target in targets
            if target.index in requested and not (attempt == 1 and target.index == 3)
        ]
        return {
            "status": "complete" if attempt > 1 else "failed",
            "message": "done" if attempt > 1 else "Captured 5 of 6 photo-grid cells",
            "frames": _frames(captured),
        }

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    await web._photo_grid_worker(record)

    # The retry pass carries only the failed cell; no successful cell is sent
    # twice, and the retry does not create a second grid-plan entry.
    assert sent == [[0, 1, 2, 3, 4, 5], [3]]
    assert len(record.targets) == 6
    assert sorted(frame.target_index for frame in record.frames) == [0, 1, 2, 3, 4, 5]
    assert record.status == "complete"


@pytest.mark.asyncio
async def test_an_upload_timeout_is_never_reported_as_a_captured_cell(monkeypatch, tmp_path):
    targets = _route(count=3, columns=3)
    record = _record(targets, tmp_path=tmp_path)
    monkeypatch.setattr(web, "photo_grid_store", PhotoGridStore(tmp_path / "grid.json"))

    passes = {"count": 0}

    async def start(_entry_id, _payload):
        passes["count"] += 1
        return {"status": "queued", "repair_id": "grid-1", "message": "queued"}

    async def status(_entry_id, _repair_id):
        # Cell 2's image never finishes uploading, on any pass.
        return {
            "status": "failed",
            "message": "FarmBot did not produce a processed image at X 981.0, Y 195.0",
            "frames": _frames([t for t in targets if t.index != 2]),
        }

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    await web._photo_grid_worker(record)

    # Bounded retries (the stall guard stops once a pass verifies nothing new),
    # then an honest incomplete result naming the exact cell.
    assert 1 < passes["count"] <= web.PHOTO_GRID_WORKER_MAX_PASSES
    assert sorted(frame.target_index for frame in record.frames) == [0, 1]
    assert record.status == "failed"
    assert "1 of 3 coordinates" in record.message
    assert "981,195" in record.message


@pytest.mark.asyncio
async def test_an_interrupted_run_leaves_enough_state_to_resume(monkeypatch, tmp_path):
    """A run cut short must persist which cells succeeded, so resuming it does
    not re-photograph them."""
    targets = _route(count=6, columns=3)
    record = _record(targets, tmp_path=tmp_path)
    store = PhotoGridStore(tmp_path / "grid.json")
    monkeypatch.setattr(web, "photo_grid_store", store)

    async def start(_entry_id, _payload):
        return {"status": "queued", "repair_id": "grid-1", "message": "queued"}

    async def status(_entry_id, _repair_id):
        return {
            "status": "failed",
            "message": "FarmBot lost its MQTT connection",
            "frames": _frames(targets[:4]),
        }

    monkeypatch.setattr(web.client, "start_grid_repair", start)
    monkeypatch.setattr(web.client, "grid_repair_status", status)

    unverified = await web._capture_photo_grid_targets(record, list(targets))

    assert [target.index for target in unverified] == [4, 5]
    reloaded = store.load()
    assert sorted(frame.target_index for frame in reloaded.frames) == [0, 1, 2, 3]
    verified = {frame.target_index for frame in reloaded.frames}
    assert [t.index for t in reloaded.targets if t.index not in verified] == [4, 5]


@pytest.mark.asyncio
async def test_targets_omit_index_for_an_integration_that_cannot_accept_it(monkeypatch, tmp_path):
    """Older service schemas reject unknown target keys outright."""
    targets = [PhotoGridTarget(index=7, row=0, column=7, x=100, y=200, z=0)]
    record = _record(targets, tmp_path=tmp_path)
    record.indexed_targets = False
    sent = []

    async def start(_entry_id, payload):
        sent.append(payload)
        return {"status": "queued", "repair_id": "grid-1", "message": "queued"}

    monkeypatch.setattr(web.client, "start_grid_repair", start)

    await web._start_photo_grid_batch(record, targets)
    assert sent == [[{"x": 100.0, "y": 200.0, "z": 0.0}]]

    record.indexed_targets = True
    await web._start_photo_grid_batch(record, targets)
    assert sent[1] == [{"x": 100.0, "y": 200.0, "z": 0.0, "index": 7}]
