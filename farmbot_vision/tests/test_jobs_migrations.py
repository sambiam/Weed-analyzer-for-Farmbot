from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from farmbot_vision.database import Database
from farmbot_vision.models import (
    Calibration,
    Decision,
    Measurement,
    OperatingMode,
)
from farmbot_vision.safety import decide
from farmbot_vision.settings import Settings


def _measurement(**kwargs) -> Measurement:
    base = dict(
        measurement_id=uuid4(),
        plant_id=1,
        crop_slug="lettuce",
        image_id=2,
        image_timestamp=datetime.now(UTC),
        current_radius_mm=100,
        typical_canopy_radius_mm=80,
        maximum_accepted_canopy_radius_mm=90,
        recommended_protection_radius_mm=140,
        confidence=0.95,
        decision=Decision.OBSERVED,
        reason="test",
        algorithm_version="test",
    )
    base.update(kwargs)
    return Measurement(**base)


def test_automatic_application_impossible_without_calibration():
    # An uncalibrated measurement can never become APPLIED or RECOMMENDED.
    uncalibrated = _measurement(calibrated=False)
    for mode in OperatingMode:
        result = decide(uncalibrated, mode, Settings())
        assert result.decision == Decision.OBSERVED
        assert result.decision not in (Decision.APPLIED, Decision.RECOMMENDED)


def test_calibrated_auto_radius_can_apply():
    result = decide(_measurement(calibrated=True), OperatingMode.AUTO_RADIUS, Settings())
    assert result.decision == Decision.APPLIED


def test_existing_data_survives_migration(tmp_path):
    # Simulate a v1 database (only migration 1 applied) carrying real rows,
    # then let the current code migrate it and confirm the rows are intact.
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    from farmbot_vision.database import MIGRATIONS

    connection.executescript(MIGRATIONS[0])
    connection.execute("INSERT INTO schema_version(version) VALUES (1)")
    connection.execute(
        "INSERT INTO calibrations(config_entry_id,source,pixels_per_mm_x,pixels_per_mm_y,"
        "rotation_degrees,offset_x_mm,offset_y_mm,uncertainty_mm) VALUES(?,?,?,?,?,?,?,?)",
        ("bot", "manual", 1.0, 1.0, 0, 0, 0, 10),
    )
    connection.execute(
        "INSERT INTO measurements(measurement_id,plant_id,crop_slug,image_id,image_timestamp,"
        "current_radius_mm,typical_canopy_radius_mm,maximum_accepted_canopy_radius_mm,"
        "recommended_protection_radius_mm,confidence,transform_json,algorithm_version,decision,"
        "reason,ambiguous,applied) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "m1",
            1,
            "lettuce",
            5,
            "2026-01-01T00:00:00+00:00",
            100,
            80,
            90,
            140,
            0.9,
            "{}",
            "old",
            "observed",
            "legacy",
            0,
            0,
        ),
    )
    connection.commit()
    connection.close()

    database = Database(path)  # runs migration 2
    assert database.stats()["measurements"] == 1
    calibration = database.active_calibration("bot")
    assert calibration is not None
    # New columns exist and default sensibly on the migrated row.
    row = database.connection.execute(
        "SELECT calibrated, analysis_resolution FROM measurements WHERE measurement_id='m1'"
    ).fetchone()
    assert row["calibrated"] == 1
    assert row["analysis_resolution"] is None


def test_derived_calibration_does_not_clobber_manual(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    manual = database.save_calibration(
        "bot",
        Calibration(
            source="manual",
            pixels_per_mm_x=1.0,
            pixels_per_mm_y=1.0,
            processed_width=960,
            processed_height=720,
        ),
    )
    database.record_calibration(
        "bot",
        Calibration(
            source="reference_scaled",
            pixels_per_mm_x=0.5,
            pixels_per_mm_y=0.5,
            processed_width=960,
            processed_height=720,
        ),
    )
    active = database.active_calibration("bot")
    assert active.source == "manual"
    assert active.version_id == manual.version_id


def test_measurement_provenance_round_trips(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    measurement = _measurement(
        analysis_resolution="960x720",
        processed_width=960,
        processed_height=720,
        calibration_source="processed_image",
        contract_version="farmbot-vision-v2",
    )
    database.save_measurements([measurement])
    row = database.recent_measurements()[0]
    assert row["analysis_resolution"] == "960x720"
    assert row["calibration_source"] == "processed_image"
    assert row["processed_width"] == 960


def test_removal_artifacts_migrate_persist_and_count_distinct_images(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    timestamp = datetime(2026, 7, 23, tzinfo=UTC)
    present = _measurement(
        config_entry_id="bot-1",
        image_id=1,
        image_timestamp=timestamp,
        artifact_paths=["/data/artifacts/one-overlay.jpg", "/data/artifacts/one-mask.png"],
    )
    first_absent = _measurement(
        config_entry_id="bot-1",
        image_id=2,
        image_timestamp=timestamp.replace(hour=1),
        vegetation_absent=True,
        absent_observations=1,
        typical_canopy_radius_mm=0,
        maximum_accepted_canopy_radius_mm=0,
        recommended_protection_radius_mm=0,
    )
    same_image_again = _measurement(
        config_entry_id="bot-1",
        image_id=2,
        image_timestamp=timestamp.replace(hour=2),
        vegetation_absent=True,
        absent_observations=1,
        typical_canopy_radius_mm=0,
        maximum_accepted_canopy_radius_mm=0,
        recommended_protection_radius_mm=0,
    )
    database.save_measurements([present, first_absent, same_image_again])

    columns = {row[1] for row in database.connection.execute("PRAGMA table_info(measurements)")}
    assert {"artifact_paths_json", "vegetation_absent", "absent_observations"} <= columns
    saved = database.measurement(str(present.measurement_id))
    assert saved is not None
    assert json.loads(saved["artifact_paths_json"]) == present.artifact_paths
    assert database.has_present_measurement("bot-1", 1) is True
    # Re-analysing one photo can replace/refine its result, but it is not an
    # independent observation that can advance the removal confirmation gate.
    assert database.absent_streak("bot-1", 1) == 1


@pytest.mark.asyncio
async def test_second_run_is_rejected_while_locked(tmp_path):
    # Sequential processing: a second run is refused while one holds the lock.
    from farmbot_vision.jobs import JobManager

    database = Database(tmp_path / "db.sqlite")
    manager = JobManager(Settings(), database, client=None)
    await manager.lock.acquire()
    try:
        result = await manager.run(entry_id="bot")
        assert result["accepted"] is False
        assert "already running" in result["reason"]
    finally:
        manager.lock.release()


def test_resource_gate_blocks_low_memory(tmp_path, monkeypatch):
    from farmbot_vision import jobs as jobs_module
    from farmbot_vision.jobs import JobManager

    database = Database(tmp_path / "db.sqlite")
    manager = JobManager(Settings(minimum_free_memory_mb=999999), database, client=None)

    class _Mem:
        available = 10 * 1024 * 1024

    monkeypatch.setattr(jobs_module.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(jobs_module.psutil, "cpu_percent", lambda interval=0.1: 1.0)
    available, reason = manager.resources_available()
    assert available is False
    assert "free memory" in reason


@pytest.mark.asyncio
async def test_new_photo_job_processes_only_the_target_image(tmp_path, monkeypatch):
    import numpy as np
    from conftest import vision_image_dict

    from farmbot_vision.jobs import JobManager
    from farmbot_vision.models import Inventory, VisionImage

    class Client:
        def __init__(self):
            self.image_ids = []
            self.statuses = []

        async def inventory(self, _request):
            return Inventory.model_validate(
                {
                    "device_id": "42",
                    "generated_at": "2026-07-20T00:00:00+00:00",
                    "plants": [],
                    "images": [
                        {
                            "id": image_id,
                            "created_at": f"2026-07-20T00:00:0{image_id}+00:00",
                            "processed": True,
                            "meta": {"x": 0, "y": 0, "z": 0},
                        }
                        for image_id in (1, 2)
                    ],
                    "curves": [],
                    "camera_calibration": {"available": False},
                }
            )

        async def image(self, request, _max_bytes):
            self.image_ids.append(request.image_id)
            return VisionImage.model_validate(
                vision_image_dict(np.zeros((240, 320, 3), np.uint8), image_id=request.image_id)
            )

        async def report_status(self, status):
            self.statuses.append(status)

    client = Client()
    manager = JobManager(Settings(data_dir=tmp_path), Database(tmp_path / "db.sqlite"), client)
    monkeypatch.setattr(manager, "resources_available", lambda: (True, "resources available"))
    result = await manager.run(
        entry_id="entry-1", image_ids=[2], trigger="new_image", queue_if_busy=True
    )
    assert result["accepted"] is True
    assert result["images_processed"] == 1
    assert client.image_ids == [2]
    assert len(list((tmp_path / "artifacts").glob("*-mask.png"))) == 1
    assert client.statuses[-1].app_version == "0.5.0"


@pytest.mark.asyncio
async def test_calibrated_job_persists_overlay_vegetation_and_ownership_artifacts(tmp_path, monkeypatch):
    import numpy as np
    from conftest import vision_image_dict

    from farmbot_vision.jobs import JobManager
    from farmbot_vision.models import Inventory, VisionImage

    image = np.zeros((240, 320, 3), np.uint8)
    import cv2

    cv2.circle(image, (160, 120), 24, (20, 210, 30), -1)

    class Client:
        async def inventory(self, _request):
            return Inventory.model_validate(
                {
                    "device_id": "42",
                    "generated_at": "2026-07-20T00:00:00+00:00",
                    "plants": [
                        {
                            "id": 21,
                            "name": "Lettuce",
                            "openfarm_slug": "lettuce",
                            "x": 0,
                            "y": 0,
                            "radius": 20,
                            "plant_stage": "planted",
                            "planted_at": "2026-07-01T00:00:00+00:00",
                        }
                    ],
                    "images": [
                        {
                            "id": 9,
                            "created_at": "2026-07-20T00:00:00+00:00",
                            "processed": True,
                            "meta": {"x": 0, "y": 0, "z": 0},
                        }
                    ],
                    "curves": [],
                    "camera_calibration": {"available": False},
                }
            )

        async def image(self, _request, _max_bytes):
            return VisionImage.model_validate(
                vision_image_dict(
                    image,
                    image_id=9,
                    processed_calibration={
                        "available": True,
                        "pixels_per_mm_x": 1,
                        "pixels_per_mm_y": 1,
                        "basis": "processed_image",
                        "width": 320,
                        "height": 240,
                    },
                )
            )

        async def report_status(self, _status):
            pass

    database = Database(tmp_path / "db.sqlite")
    manager = JobManager(Settings(data_dir=tmp_path), database, Client())
    monkeypatch.setattr(manager, "resources_available", lambda: (True, "resources available"))
    result = await manager.run(entry_id="bot-1", image_ids=[9])

    assert result["accepted"] is True
    row = database.recent_measurements()[0]
    artifact_paths = [Path(path) for path in row["artifact_paths"]]
    assert len(artifact_paths) == 3
    assert {path.suffix for path in artifact_paths} == {".jpg", ".png"}
    assert any(path.name.endswith("-overlay.jpg") for path in artifact_paths)
    assert any(path.name.endswith("-mask.png") for path in artifact_paths)
    assert any("-plant-21-mask.png" in path.name for path in artifact_paths)
    assert all(path.is_file() for path in artifact_paths)
