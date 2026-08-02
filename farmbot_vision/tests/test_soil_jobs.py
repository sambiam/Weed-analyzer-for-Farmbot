from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from farmbot_vision.database import Database
from farmbot_vision.models import (
    SoilMeasurement,
    SoilMotionState,
    SoilPoint,
    SoilPointInventory,
    SoilStereoCalibration,
)
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
