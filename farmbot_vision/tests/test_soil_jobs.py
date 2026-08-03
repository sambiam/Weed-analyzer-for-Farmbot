from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

import farmbot_vision.soil_jobs as soil_jobs_module
from farmbot_vision.database import Database
from farmbot_vision.models import (
    SoilMeasurement,
    SoilMotionState,
    SoilPoint,
    SoilPointInventory,
    SoilStereoCalibration,
)
from farmbot_vision.soil_height import SoilCalibrationQualityError, SoilFrame, SoilHeightError
from farmbot_vision.soil_jobs import SoilJobManager
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


@pytest.mark.asyncio
async def test_capture_frames_uses_actual_1280x720_geometry(tmp_path, monkeypatch):
    capture_id = uuid4()
    image_requests = []
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
            return SimpleNamespace(
                image_id=request.image_id,
                full_metadata=True,
                width=1280,
                height=720,
                source_width=1280,
                source_height=720,
                image_base64="anBlZw==",
                meta=SimpleNamespace(x=item.x, y=item.y, z=item.z),
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
    assert {(request.max_width, request.max_height) for request in image_requests} == {(1280, 960)}

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


def test_soil_points_use_nearest_neighbour_order():
    points = [
        SoilPoint(id=1, name="far", x=100, y=100, z=0),
        SoilPoint(id=2, name="near", x=10, y=0, z=0),
        SoilPoint(id=3, name="middle", x=30, y=0, z=0),
    ]
    ordered = SoilJobManager._nearest_order(points, 0, 0)
    assert [point.id for point in ordered] == [2, 3, 1]


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
