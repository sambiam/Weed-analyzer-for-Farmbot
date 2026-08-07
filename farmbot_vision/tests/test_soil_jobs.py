from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

import farmbot_vision.soil_jobs as soil_jobs_module
from farmbot_vision.database import Database
from farmbot_vision.models import (
    SoilCaptureStatus,
    SoilGridPlan,
    SoilMeasurement,
    SoilMotionState,
    SoilPoint,
    SoilPointInventory,
    SoilStereoCalibration,
)
from farmbot_vision.soil_height import SoilCalibrationQualityError, SoilFrame, SoilHeightError
from farmbot_vision.soil_jobs import SoilJobManager
from farmbot_vision.soil_settings import SoilSettings, SoilSettingsStore
from farmbot_vision.zones import ZoneStore


def _calibration() -> SoilStereoCalibration:
    return SoilStereoCalibration(
        config_entry_id="bot-soil",
        point_id=10,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=400,
        z_direction=-1,
        inverse_depth_slope=0.02,
        inverse_depth_intercept=0.001,
        residual_mm=2.5,
        processed_width=1280,
        processed_height=960,
        source_width=2592,
        source_height=1944,
        source_image_ids=list(range(1, 10)),
        camera_signature="signature",
    )


def test_soil_geometry_accepts_widescreen_calibration_and_rejects_measurement_drift():
    widescreen = (1280, 720, 1280, 720)
    SoilJobManager._validate_soil_geometry(widescreen, None)
    SoilJobManager._validate_soil_geometry(widescreen, widescreen)
    with pytest.raises(SoilHeightError, match="soil-height calibration"):
        SoilJobManager._validate_soil_geometry(widescreen, (1280, 960, 1280, 960))


def test_soil_capture_status_accepts_integration_batch_id():
    batch_id = uuid4()
    status = SoilCaptureStatus.model_validate(
        {
            "capture_id": str(uuid4()),
            "batch_id": str(batch_id),
            "status": "running",
            "message": "capturing",
            "frames": [],
        }
    )
    assert status.batch_id == batch_id


@pytest.mark.asyncio
async def test_automatic_acceptance_requires_confidence_and_height_margin(tmp_path):
    database = Database(tmp_path / "vision.db")
    store = SoilSettingsStore(tmp_path / "soil_settings.json")
    store.save(
        SoilSettings(
            automatic_acceptance_enabled=True,
            automatic_acceptance_confidence_percent=85,
            automatic_acceptance_margin_mm=10,
        )
    )
    applied = []

    class Client:
        async def apply_soil_height(self, request):
            applied.append(request)
            return {"status": "applied", "message": "updated automatically"}

    manager = SoilJobManager(
        database,
        Client(),
        tmp_path,
        asyncio.Lock(),
        ZoneStore(tmp_path / "zones.json"),
        store,
    )
    now = datetime.now(UTC)

    def measurement(*, confidence=0.95, proposed=-395):
        return SoilMeasurement(
            measurement_id=uuid4(),
            config_entry_id="bot-soil",
            point_id=70,
            point_name="Soil 70",
            expected_x=100,
            expected_y=200,
            old_z_mm=-400,
            point_updated_at=now,
            capture_x=110,
            capture_y=210,
            relocation_distance_mm=14.1,
            proposed_z_mm=proposed,
            confidence=confidence,
            uncertainty_mm=2,
            status="valid",
            reason="passed",
        )

    accepted = measurement(confidence=0.85)
    low_confidence = measurement(confidence=0.849)
    large_change = measurement(proposed=-380)
    standalone = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="bot-soil",
        point_id=0,
        point_name="New soil point",
        expected_x=800,
        expected_y=900,
        old_z_mm=0,
        point_updated_at=None,
        capture_x=800,
        capture_y=900,
        relocation_distance_mm=0,
        proposed_z_mm=-390,
        confidence=0.85,
        uncertainty_mm=2,
        status="valid",
        reason="passed",
    )
    for item in (accepted, low_confidence, large_change, standalone):
        database.save_soil_measurement(item)
        await manager._automatically_apply(item)

    assert [item.measurement_id for item in applied] == [
        accepted.measurement_id,
        standalone.measurement_id,
    ]
    assert applied[0].human_approved is True
    assert applied[1].point_id is None
    assert applied[1].expected_updated_at is None
    assert database.soil_measurement(str(accepted.measurement_id))["status"] == "applied"
    assert database.soil_measurement(str(standalone.measurement_id))["status"] == "applied"
    assert database.soil_measurement(str(low_confidence.measurement_id))["status"] == "valid"
    assert database.soil_measurement(str(large_change.measurement_id))["status"] == "valid"


@pytest.mark.asyncio
async def test_automatic_write_rejection_keeps_result_pending_for_human_review(tmp_path):
    database = Database(tmp_path / "vision.db")
    store = SoilSettingsStore(tmp_path / "soil_settings.json")
    store.save(SoilSettings(automatic_acceptance_enabled=True))

    class Client:
        async def apply_soil_height(self, _request):
            return {"status": "rejected", "message": "Human approval is required"}

    manager = SoilJobManager(
        database,
        Client(),
        tmp_path,
        asyncio.Lock(),
        ZoneStore(tmp_path / "zones.json"),
        store,
    )
    measurement = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="bot-soil",
        point_id=70,
        point_name="Custom soil point",
        expected_x=4000,
        expected_y=600,
        old_z_mm=-400,
        point_updated_at=datetime.now(UTC),
        capture_x=4100,
        capture_y=600,
        relocation_distance_mm=100,
        proposed_z_mm=-395,
        confidence=0.95,
        uncertainty_mm=2,
        status="valid",
        reason="passed",
    )
    database.save_soil_measurement(measurement)

    await manager._automatically_apply(measurement)

    saved = database.soil_measurement(str(measurement.measurement_id))
    assert saved["status"] == "valid"
    decisions = database.connection.execute(
        "SELECT action FROM soil_decisions WHERE measurement_id=?",
        (str(measurement.measurement_id),),
    ).fetchall()
    assert [row["action"] for row in decisions] == ["automatic_apply_deferred"]


@pytest.mark.asyncio
async def test_capture_frames_uses_actual_1280x720_geometry(tmp_path, monkeypatch):
    capture_id = uuid4()
    image_requests = []
    start_calls = 0
    capture_items = [
        SimpleNamespace(
            image_id=index,
            x=100,
            y=200 + lateral,
            z=0,
            lateral_offset_mm=lateral,
            z_offset_mm=0,
            capture_attempt=1,
        )
        for index, lateral in enumerate((-15, 0, 15), start=1)
    ]

    class Client:
        async def start_soil_capture(self, _request):
            nonlocal start_calls
            start_calls += 1
            if start_calls == 1:
                return SimpleNamespace(
                    status="rejected", capture_id=None, message="FarmBot is busy"
                )
            return SimpleNamespace(status="queued", capture_id=capture_id, message="queued")

        async def soil_capture_status(self, _entry_id, _capture_id):
            return SimpleNamespace(
                status="complete",
                message="Captured 3 soil images",
                frames=capture_items,
            )

        async def image(self, request, _maximum_bytes):
            image_requests.append(request)
            item = capture_items[request.image_id - 1]
            # The downloaded image metadata is authoritative and may differ
            # slightly from the requested capture target.
            recorded_y = {1: 184.0, 2: 200.0, 3: 216.0}[request.image_id]
            return SimpleNamespace(
                image_id=request.image_id,
                full_metadata=True,
                width=1280,
                height=720,
                source_width=1280,
                source_height=720,
                image_base64="anBlZw==",
                meta=SimpleNamespace(x=item.x, y=recorded_y, z=item.z),
                processed_calibration=None,
            )

    manager = SoilJobManager(
        Database(tmp_path / "vision.db"),
        Client(),
        tmp_path,
        asyncio.Lock(),
        ZoneStore(tmp_path / "zones.json"),
    )
    monkeypatch.setattr(
        "farmbot_vision.soil_jobs.inspect_photo_quality",
        lambda _jpeg: SimpleNamespace(issue="usable"),
    )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("farmbot_vision.soil_jobs.asyncio.sleep", no_sleep)
    _capture_id, frames, _signature = await manager._capture_frames(
        config_entry_id="bot-soil",
        point_id=None,
        capture_x=100,
        capture_y=200,
        capture_z=0,
        baseline_mm=15,
        z_offsets_mm=[0],
    )
    assert {(frame.processed_width, frame.processed_height) for frame in frames} == {(1280, 720)}
    assert [frame.y for frame in frames] == [184.0, 200.0, 216.0]
    assert {(request.max_width, request.max_height) for request in image_requests} == {(1280, 960)}
    assert start_calls == 2

    image_requests.clear()
    await manager._capture_frames(
        config_entry_id="bot-soil",
        point_id=70,
        capture_x=100,
        capture_y=200,
        capture_z=0,
        baseline_mm=15,
        z_offsets_mm=[0],
        expected_geometry=(1280, 720, 1280, 720),
    )
    assert {(request.max_width, request.max_height) for request in image_requests} == {(1280, 720)}
    assert start_calls == 3


def test_soil_records_round_trip_and_restart_interrupts_jobs(tmp_path):
    path = tmp_path / "vision.db"
    database = Database(path)
    calibration = database.save_soil_calibration(_calibration())
    assert calibration.calibration_id is not None
    assert database.active_soil_calibration("bot-soil") == calibration

    measurement = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="bot-soil",
        point_id=10,
        point_name="Soil 10",
        expected_x=100,
        expected_y=200,
        old_z_mm=-400,
        proposed_z_mm=-397,
        confidence=0.91,
        uncertainty_mm=4,
        status="valid",
        reason="passed",
        calibration_id=calibration.calibration_id,
        frame_ids=[20, 21, 22],
        metrics={"valid_coverage": 0.4},
        artifact_paths=["/data/artifacts/soil.png"],
    )
    database.save_soil_measurement(measurement)
    loaded = database.soil_measurement(str(measurement.measurement_id))
    assert loaded["frame_ids"] == [20, 21, 22]
    assert loaded["metrics"]["valid_coverage"] == 0.4

    database.start_soil_job("soil-job", "bot-soil", "measurement", [10])
    database.connection.close()
    restarted = Database(path)
    job = restarted.soil_job("soil-job")
    assert job["status"] == "interrupted"
    assert job["completed_at"] is not None


def test_soil_height_change_log_reports_user_and_automatic_applies(tmp_path):
    database = Database(tmp_path / "vision.db")
    calibration = database.save_soil_calibration(_calibration())

    approved = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="bot-soil",
        point_id=10,
        point_name="Soil 10",
        expected_x=100,
        expected_y=200,
        old_z_mm=-400,
        capture_x=100,
        capture_y=200,
        proposed_z_mm=-397,
        confidence=0.91,
        status="applied",
        reason="passed",
        calibration_id=calibration.calibration_id,
    )
    database.save_soil_measurement(approved)
    database.record_soil_decision(str(approved.measurement_id), "approve", {"status": "applied"})

    auto_approved = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="bot-soil",
        point_id=11,
        point_name="Soil 11",
        expected_x=300,
        expected_y=400,
        old_z_mm=-410,
        capture_x=300,
        capture_y=400,
        proposed_z_mm=-402,
        confidence=0.97,
        status="applied",
        reason="passed",
        calibration_id=calibration.calibration_id,
    )
    database.save_soil_measurement(auto_approved)
    database.record_soil_decision(
        str(auto_approved.measurement_id), "automatic_approve", {"status": "applied"}
    )

    # A rejection must not appear in the applied change log.
    rejected = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="bot-soil",
        point_id=12,
        point_name="Soil 12",
        expected_x=500,
        expected_y=600,
        old_z_mm=-420,
        capture_x=500,
        capture_y=600,
        proposed_z_mm=-415,
        confidence=0.6,
        status="rejected",
        reason="user declined",
        calibration_id=calibration.calibration_id,
    )
    database.save_soil_measurement(rejected)
    database.record_soil_decision(str(rejected.measurement_id), "reject", {"status": "rejected"})

    log = database.soil_height_change_log("bot-soil")
    methods_by_point = {(entry["point_name"], entry["method"]): entry for entry in log}
    assert ("Soil 10", "user") in methods_by_point
    assert ("Soil 11", "automatic") in methods_by_point
    assert "Soil 12" not in {entry["point_name"] for entry in log}
    assert methods_by_point[("Soil 11", "automatic")]["new_z_mm"] == -402


def test_soil_points_use_nearest_neighbour_order():
    points = [
        SoilPoint(id=1, name="far", x=100, y=100, z=0),
        SoilPoint(id=2, name="near", x=10, y=0, z=0),
        SoilPoint(id=3, name="middle", x=30, y=0, z=0),
    ]
    ordered = SoilJobManager._nearest_order(points, 0, 0)
    assert [point.id for point in ordered] == [2, 3, 1]


def test_custom_measurement_automatically_replaces_nearby_or_creates_new_point():
    now = datetime.now(UTC)
    inventory = SoilPointInventory(
        device_id="42",
        generated_at=now,
        points=[SoilPoint(id=70, name="Nearby", x=100, y=100, z=-400, updated_at=now)],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 0, "y": 0, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
        ),
    )

    replacement = SoilJobManager._custom_measurement_target(inventory, 250, 100)
    standalone = SoilJobManager._custom_measurement_target(inventory, 500, 500)

    assert replacement["point_id"] == 70
    assert replacement["relocation_distance_mm"] == 150
    assert standalone["point_id"] == 0
    assert standalone["point_updated_at"] is None


@pytest.mark.asyncio
async def test_confirmed_grid_is_replanned_under_measurement_lock(tmp_path, monkeypatch):
    database = Database(tmp_path / "vision.db")
    manager = SoilJobManager(
        database,
        object(),
        tmp_path,
        asyncio.Lock(),
        ZoneStore(tmp_path / "zones.json"),
    )
    inventory = SoilPointInventory(
        device_id="42",
        generated_at=datetime.now(UTC),
        points=[],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 0, "y": 0, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
        ),
    )
    calls = []

    async def measurement_grid(*args, **kwargs):
        calls.append((args, kwargs))
        return inventory, SoilGridPlan(
            spacing_mm=500,
            maximum_deviation_mm=80,
            clear_soil_margin_mm=75,
            points=[],
        )

    async def stale_preview(*_args, **_kwargs):
        raise AssertionError("ordinary cached safe sites must not authorize a grid run")

    monkeypatch.setattr(manager, "measurement_grid", measurement_grid)
    monkeypatch.setattr(manager, "safe_sites", stale_preview)
    await manager._run_measurements(
        job_id="grid-job",
        config_entry_id="bot-soil",
        point_ids=[],
        custom_point_id=None,
        custom_x=None,
        custom_y=None,
        capture_z=0,
        baseline_mm=15,
        job_kind="measurement_grid",
        grid_spacing_mm=500,
        grid_maximum_deviation_mm=80,
    )

    assert len(calls) == 1
    assert calls[0][1]["spacing_mm"] == 500
    assert calls[0][1]["maximum_deviation_mm"] == 80
    assert manager.current["status"] == "failed"
    assert "no eligible clear-soil" in manager.current["message"]


@pytest.mark.asyncio
async def test_calibration_rechecks_clear_site_immediately_before_capture(tmp_path, monkeypatch):
    database = Database(tmp_path / "vision.db")
    manager = SoilJobManager(
        database,
        object(),
        tmp_path,
        asyncio.Lock(),
        ZoneStore(tmp_path / "zones.json"),
    )
    inventory = SoilPointInventory(
        device_id="42",
        generated_at=datetime.now(UTC),
        points=[],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 0, "y": 0, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
        ),
    )
    captured = False

    async def safe_sites(_entry_id, _baseline, *, clear_soil_margin_mm=75):
        return inventory, []

    async def capture_frames(**_kwargs):
        nonlocal captured
        captured = True
        raise AssertionError("unsafe calibration capture was started")

    monkeypatch.setattr(manager, "safe_sites", safe_sites)
    monkeypatch.setattr(manager, "_capture_frames", capture_frames)
    await manager._run_calibration(
        job_id="calibration-job",
        config_entry_id="bot-soil",
        point_id=10,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=400,
    )

    assert captured is False
    assert manager.current["status"] == "failed"
    assert "plant- and weed-free" in manager.current["message"]


@pytest.mark.asyncio
async def test_custom_calibration_does_not_resolve_a_soil_point(tmp_path, monkeypatch):
    database = Database(tmp_path / "vision.db")
    manager = SoilJobManager(
        database,
        object(),
        tmp_path,
        asyncio.Lock(),
        ZoneStore(tmp_path / "zones.json"),
    )
    inventory = SoilPointInventory(
        device_id="42",
        generated_at=datetime.now(UTC),
        points=[],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 0, "y": 0, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
        ),
    )
    captured = {}

    async def safe_sites(_entry_id, _baseline, *, clear_soil_margin_mm=75):
        return inventory, []

    async def capture_frames(**kwargs):
        captured.update(kwargs)
        raise SoilHeightError("capture path reached")

    monkeypatch.setattr(manager, "safe_sites", safe_sites)
    monkeypatch.setattr(manager, "_capture_frames", capture_frames)
    await manager._run_calibration(
        job_id="custom-calibration-job",
        config_entry_id="bot-soil",
        point_id=None,
        capture_x=320,
        capture_y=450,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=400,
    )

    assert captured["point_id"] is None
    assert captured["capture_x"] == 320
    assert captured["capture_y"] == 450
    assert manager.current["message"] == "capture path reached"


@pytest.mark.asyncio
async def test_calibration_quality_failure_offers_and_accepts_an_override(tmp_path, monkeypatch):
    database = Database(tmp_path / "vision.db")
    manager = SoilJobManager(
        database, object(), tmp_path, asyncio.Lock(), ZoneStore(tmp_path / "zones.json")
    )
    inventory = SoilPointInventory(
        device_id="42",
        generated_at=datetime.now(UTC),
        points=[],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 0, "y": 0, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
        ),
    )
    frames = [
        SoilFrame(
            image_id=index,
            jpeg=b"",
            x=100,
            y=200,
            z=0,
            lateral_offset_mm=0,
            z_offset_mm=0,
            processed_width=640,
            processed_height=480,
            source_width=640,
            source_height=480,
        )
        for index in range(1, 4)
    ]
    forced_calls = []

    def fake_fit_calibration(**kwargs):
        forced_calls.append(kwargs.get("force", False))
        if not kwargs.get("force", False):
            raise SoilCalibrationQualityError(
                "calibration imagery failed quality gates at 0 mm (1/3 pairs passed: "
                "pair 1 passed; pair 2 failed (coverage 0.05 < 0.15); "
                "pair 3 failed (plane support 0.20 < 0.50))"
            )
        return _calibration().model_copy(
            update={
                "quality_override": True,
                "quality_warnings": ["0 mm gate bypassed -- accepted by override"],
            }
        )

    async def safe_sites(_entry_id, _baseline, *, clear_soil_margin_mm=75):
        return inventory, []

    async def capture_frames(**_kwargs):
        return uuid4(), frames, "sig"

    monkeypatch.setattr(manager, "safe_sites", safe_sites)
    monkeypatch.setattr(manager, "_capture_frames", capture_frames)
    monkeypatch.setattr(soil_jobs_module, "fit_calibration", fake_fit_calibration)

    await manager._run_calibration(
        job_id="cal-job",
        config_entry_id="bot-soil",
        point_id=None,
        capture_x=100,
        capture_y=200,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=300,
    )

    assert manager.current["status"] == "failed"
    assert "failed quality gates" in manager.current["message"]
    assert "coverage 0.05" in manager.current["detail"]
    assert manager.pending_override_job_id == "cal-job"
    persisted = database.soil_job("cal-job")
    assert persisted["status"] == "failed"
    assert "coverage 0.05" in persisted["detail"]
    assert database.active_soil_calibration("bot-soil") is None

    override_job_id = manager.start_calibration_override()
    await manager.task

    assert forced_calls == [False, True]
    assert manager.current["status"] == "complete"
    assert manager.pending_override_job_id is None
    saved = database.active_soil_calibration("bot-soil")
    assert saved is not None
    assert saved.quality_override is True
    assert database.soil_job(override_job_id)["status"] == "complete"

    # A fresh calibration attempt drops the stale override, closing the window.
    manager.start_calibration(
        config_entry_id="bot-soil",
        point_id=None,
        capture_x=100,
        capture_y=200,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=300,
    )
    assert manager.pending_override_job_id is None
    await manager.close()


@pytest.mark.asyncio
async def test_measurement_run_uses_one_capture_batch_and_finishes_it_once(tmp_path, monkeypatch):
    database = Database(tmp_path / "vision.db")
    database.save_soil_calibration(_calibration())
    finished = []

    class Client:
        async def finish_soil_capture_batch(self, entry_id, batch_id):
            finished.append((entry_id, batch_id))
            return {"status": "complete", "message": "restored"}

    manager = SoilJobManager(
        database,
        Client(),
        tmp_path,
        asyncio.Lock(),
        ZoneStore(tmp_path / "zones.json"),
    )
    now = datetime.now(UTC)
    inventory = SoilPointInventory(
        device_id="42",
        generated_at=now,
        points=[],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 0, "y": 0, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
        ),
    )
    sites = [
        SimpleNamespace(
            point_id=point_id,
            point_name=f"Soil {point_id}",
            expected_x=x,
            expected_y=100,
            expected_z=-400,
            point_updated_at=now,
            capture_x=x,
            capture_y=100,
            relocation_distance_mm=0,
        )
        for point_id, x in ((1, 100), (2, 200))
    ]
    batch_ids = []

    async def safe_sites(_entry_id, _baseline, *, clear_soil_margin_mm=75):
        return inventory, sites

    async def capture_frames(**kwargs):
        batch_ids.append(kwargs["batch_id"])
        return uuid4(), [], "signature"

    def analyse(_frames, _calibration, **_kwargs):
        return SimpleNamespace(
            measurement_id=uuid4(),
            artifacts={},
            proposed_z_mm=-399,
            confidence=0.9,
            uncertainty_mm=2,
            valid=True,
            reason="passed",
            metrics={},
        )

    monkeypatch.setattr(manager, "safe_sites", safe_sites)
    monkeypatch.setattr(manager, "_capture_frames", capture_frames)
    monkeypatch.setattr(soil_jobs_module, "analyse_soil_height", analyse)

    await manager._run_measurements(
        job_id="measurement-job",
        config_entry_id="bot-soil",
        point_ids=[1, 2],
        custom_point_id=None,
        custom_x=None,
        custom_y=None,
        capture_z=0,
        baseline_mm=15,
    )

    assert len(batch_ids) == 2
    assert batch_ids[0] == batch_ids[1]
    assert finished == [("bot-soil", str(batch_ids[0]))]
    assert manager.current["status"] == "complete"

    batch_ids.clear()
    finished.clear()

    async def communication_failure(**kwargs):
        batch_ids.append(kwargs["batch_id"])
        raise soil_jobs_module.HomeAssistantError("malformed FarmBot integration response")

    monkeypatch.setattr(manager, "_capture_frames", communication_failure)
    await manager._run_measurements(
        job_id="failed-communication-job",
        config_entry_id="bot-soil",
        point_ids=[1, 2],
        custom_point_id=None,
        custom_x=None,
        custom_y=None,
        capture_z=0,
        baseline_mm=15,
    )

    assert len(batch_ids) == 1
    assert finished == [("bot-soil", str(batch_ids[0]))]
    assert manager.current["status"] == "failed"
