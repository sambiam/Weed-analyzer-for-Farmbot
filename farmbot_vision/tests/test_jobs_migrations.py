from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from farmbot_vision import __version__
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
        "SELECT calibrated,analysis_resolution,recorded_center_x,recorded_center_y "
        "FROM measurements WHERE measurement_id='m1'"
    ).fetchone()
    assert row["calibrated"] == 1
    assert row["analysis_resolution"] is None
    assert row["recorded_center_x"] is None
    assert row["recorded_center_y"] is None


def test_measurement_preserves_recorded_and_recommended_centers(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    measurement = _measurement(
        vegetation_absent=True,
        center_misaligned=True,
        recorded_center_x=125.5,
        recorded_center_y=480.25,
        recommended_center_px=(140.0, 500.0),
    )

    database.save_measurements([measurement])
    row = database.measurement(str(measurement.measurement_id))

    assert row["recorded_center_x"] == 125.5
    assert row["recorded_center_y"] == 480.25
    assert row["recommended_center_x"] == 140.0
    assert row["recommended_center_y"] == 500.0


def test_fused_canopy_provenance_round_trips_and_overrides_consolidation(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    first = _measurement(
        image_id=1,
        fused_canopy=True,
        fused_typical_radius_mm=100,
        fused_maximum_radius_mm=120,
        fused_recommended_radius_mm=145,
        fused_confidence=0.91,
        fusion_view_count=2,
        fusion_angular_coverage=0.94,
        fusion_corroborated_fraction=0.7,
        fusion_disagreement_mm=8,
        fusion_reliable=True,
        fusion_diagnostic_path="/data/artifacts/fusion.jpg",
    )
    second = _measurement(image_id=2)
    database.save_measurements([first, second])

    row = database.measurement(str(first.measurement_id))
    consolidated = database.pending_measurements()[0]

    assert row["fused_maximum_radius_mm"] == 120
    assert row["fusion_reliable"] == 1
    assert consolidated["fused_canopy"] == 1
    assert consolidated["maximum_accepted_canopy_radius_mm"] == 120
    assert consolidated["recommended_protection_radius_mm"] == 145
    assert consolidated["fusion_diagnostic_path"].endswith("fusion.jpg")


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


def test_pending_views_are_consolidated_with_outlier_resistance(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    timestamp = datetime(2026, 7, 26, tzinfo=UTC)
    measurements = [
        _measurement(
            config_entry_id="bot",
            image_id=101,
            image_timestamp=timestamp,
            maximum_accepted_canopy_radius_mm=100,
            recommended_protection_radius_mm=130,
            confidence=0.9,
            visible_fraction=1,
        ),
        _measurement(
            config_entry_id="bot",
            image_id=102,
            image_timestamp=timestamp.replace(hour=1),
            maximum_accepted_canopy_radius_mm=105,
            recommended_protection_radius_mm=135,
            confidence=0.85,
            visible_fraction=0.95,
        ),
        _measurement(
            config_entry_id="bot",
            image_id=103,
            image_timestamp=timestamp.replace(hour=2),
            maximum_accepted_canopy_radius_mm=600,
            recommended_protection_radius_mm=630,
            confidence=0.3,
            visible_fraction=0.2,
        ),
    ]
    database.save_measurements(measurements)

    rows = database.pending_measurements()

    assert len(rows) == 1
    assert rows[0]["measurement_count"] == 3
    assert rows[0]["recommended_protection_radius_mm"] in {130, 135}
    assert rows[0]["recommended_protection_radius_mm"] < 200


def test_reanalysis_supersedes_the_old_review_row_without_deleting_history(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    first = _measurement(config_entry_id="bot", decision=Decision.RECOMMENDED)
    second = _measurement(
        config_entry_id="bot",
        plant_id=first.plant_id,
        image_id=first.image_id,
        decision=Decision.RECOMMENDED,
    )
    database.save_measurements([first])
    database.save_measurements([second])
    pending_ids = {row["measurement_id"] for row in database.pending_measurements()}
    assert str(first.measurement_id) not in pending_ids
    assert str(second.measurement_id) in pending_ids
    assert database.measurement(str(first.measurement_id)) is not None


def test_clear_pending_review_items_preserves_history(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    measurement = _measurement(config_entry_id="clear-bot")
    database.save_measurements([measurement])
    weed_id = str(uuid4())
    database.save_weed_detection(
        detection_id=weed_id,
        config_entry_id="clear-bot",
        image_id=measurement.image_id,
        image_timestamp=measurement.image_timestamp,
        x=100,
        y=200,
        z=0,
        area_mm2=100,
        radius_mm=12,
        confidence=0.9,
        overlay_path=None,
    )
    observing_weed_id = str(uuid4())
    database.save_weed_detection(
        detection_id=observing_weed_id,
        config_entry_id="clear-bot",
        image_id=measurement.image_id + 1,
        image_timestamp=measurement.image_timestamp,
        x=120,
        y=220,
        z=0,
        area_mm2=80,
        radius_mm=10,
        confidence=0.7,
        overlay_path=None,
        status="observing",
    )
    rejected_weed_id = str(uuid4())
    database.save_weed_detection(
        detection_id=rejected_weed_id,
        config_entry_id="clear-bot",
        image_id=measurement.image_id + 2,
        image_timestamp=measurement.image_timestamp,
        x=140,
        y=240,
        z=0,
        area_mm2=60,
        radius_mm=8,
        confidence=0.6,
        overlay_path=None,
        status="rejected",
    )

    assert database.clear_pending_measurements() == 1
    assert database.pending_measurements() == []
    assert database.measurement(str(measurement.measurement_id)) is not None
    decision = database.connection.execute(
        "SELECT action FROM decisions WHERE measurement_id=?",
        (str(measurement.measurement_id),),
    ).fetchone()
    assert decision["action"] == "superseded"

    assert database.clear_pending_weed_detections() == 2
    assert database.pending_weed_detections() == []
    assert database.weed_detection(weed_id)["status"] == "superseded"
    assert database.weed_detection(observing_weed_id)["status"] == "superseded"
    assert database.weed_detection(rejected_weed_id)["status"] == "rejected"
    assert database.clear_pending_weed_detections() == 0


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
    assert {
        "artifact_paths_json",
        "vegetation_absent",
        "absent_observations",
        "composite_overlay_path",
    } <= columns
    saved = database.measurement(str(present.measurement_id))
    assert saved is not None
    assert json.loads(saved["artifact_paths_json"]) == present.artifact_paths
    assert database.has_present_measurement("bot-1", 1) is True
    # Re-analysing one photo can replace/refine its result, but it is not an
    # independent observation that can advance the removal confirmation gate.
    assert database.absent_streak("bot-1", 1) == 1


def test_rejected_weed_position_suppresses_future_detections_after_restart(tmp_path):
    path = tmp_path / "db.sqlite"
    database = Database(path)
    rejected_id = str(uuid4())
    duplicate_id = str(uuid4())
    common = {
        "config_entry_id": "bot-1",
        "image_id": 10,
        "image_timestamp": datetime(2026, 7, 23, tzinfo=UTC),
        "z": 0,
        "area_mm2": 100,
        "radius_mm": 30,
        "confidence": 0.9,
        "overlay_path": None,
    }
    database.save_weed_detection(
        detection_id=rejected_id,
        x=100,
        y=200,
        **common,
    )
    database.save_weed_detection(
        detection_id=duplicate_id,
        x=105,
        y=203,
        **common,
    )

    assert database.reject_weed_detection(rejected_id, 20) is True
    assert database.pending_weed_detections() == []
    database.connection.close()

    restarted = Database(path)
    assert restarted.has_weed_detection_near("bot-1", 103, 202, 20) is True
    # The stored rejected radius extends the suppression area beyond the
    # current detection's duplicate tolerance.
    assert restarted.has_weed_detection_near("bot-1", 140, 200, 20) is True
    assert restarted.has_weed_detection_near("bot-1", 151, 200, 20) is False
    assert restarted.has_weed_detection_near("bot-2", 103, 202, 20) is False


def test_weed_candidate_tracking_and_training_labels_survive_restart(tmp_path):
    path = tmp_path / "db.sqlite"
    database = Database(path)
    first = database.observe_weed_candidate(
        config_entry_id="bot-1",
        image_id=1,
        seen_at=datetime(2026, 7, 23, tzinfo=UTC),
        x=100,
        y=200,
        confidence=0.7,
        match_distance_mm=25,
        max_gap_hours=72,
    )
    repeated_image = database.observe_weed_candidate(
        config_entry_id="bot-1",
        image_id=1,
        seen_at=datetime(2026, 7, 23, tzinfo=UTC),
        x=103,
        y=201,
        confidence=0.8,
        match_distance_mm=25,
        max_gap_hours=72,
    )
    second = database.observe_weed_candidate(
        config_entry_id="bot-1",
        image_id=2,
        seen_at=datetime(2026, 7, 24, tzinfo=UTC),
        x=104,
        y=202,
        confidence=0.9,
        match_distance_mm=25,
        max_gap_hours=72,
    )
    assert first["observations"] == repeated_image["observations"] == 1
    assert second["id"] == first["id"]
    assert second["observations"] == 2

    detection_id = str(uuid4())
    database.save_weed_detection(
        detection_id=detection_id,
        config_entry_id="bot-1",
        image_id=2,
        image_timestamp=datetime(2026, 7, 24, tzinfo=UTC),
        x=104,
        y=202,
        z=0,
        area_mm2=90,
        radius_mm=15,
        confidence=0.85,
        heuristic_confidence=0.8,
        verifier_confidence=0.9,
        features={"strong_green_fraction": 0.8},
        crop_path="/data/artifacts/candidate.jpg",
        overlay_path=None,
        observation_count=2,
        candidate_track_id=first["id"],
    )
    assert database.label_weed_detection(detection_id, "weed") is True
    database.connection.close()

    restarted = Database(path)
    sample = restarted.weed_training_samples()[0]
    assert sample["label"] == "weed"
    assert sample["features"]["strong_green_fraction"] == 0.8
    assert restarted.weed_detection(detection_id)["observation_count"] == 2


def test_weed_training_samples_can_be_relabelled_and_cleared(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    detection_id = str(uuid4())
    database.save_weed_detection(
        detection_id=detection_id,
        config_entry_id="bot-1",
        image_id=1,
        image_timestamp=datetime(2026, 7, 24, tzinfo=UTC),
        x=104,
        y=202,
        z=0,
        area_mm2=90,
        radius_mm=15,
        confidence=0.85,
        features={"strong_green_fraction": 0.8},
        crop_path="/data/artifacts/candidate.jpg",
        overlay_path=None,
    )

    assert database.label_weed_detection(detection_id, "weed") is True
    assert database.update_weed_training_sample_label(detection_id, "crop") is True
    assert database.weed_training_samples()[0]["label"] == "crop"
    assert database.clear_weed_training_samples() == ["/data/artifacts/candidate.jpg"]
    assert database.weed_training_samples() == []


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


@pytest.mark.asyncio
async def test_approved_radius_creates_curve_when_plant_has_none(tmp_path):
    from farmbot_vision.jobs import JobManager
    from farmbot_vision.models import Inventory

    class Client:
        def __init__(self):
            self.requests = []

        async def upsert_curve(self, request):
            self.requests.append(request)
            return {"status": "applied", "curve_id": 99, "message": "Curve created"}

    inventory = Inventory.model_validate(
        {
            "device_id": "42",
            "generated_at": "2026-07-25T00:00:00+00:00",
            "plants": [
                {
                    "id": 1,
                    "name": "Thyme",
                    "openfarm_slug": "thyme",
                    "x": 0,
                    "y": 0,
                    "radius": 100,
                    "plant_stage": "planted",
                    "planted_at": "2026-07-01T00:00:00+00:00",
                }
            ],
            "images": [],
            "curves": [],
            "camera_calibration": {"available": False},
        }
    )
    client = Client()
    manager = JobManager(Settings(), Database(tmp_path / "db.sqlite"), client)
    result = await manager._update_curve_after_radius(
        "entry",
        inventory,
        _measurement(
            config_entry_id="entry",
            crop_slug="thyme",
            current_radius_mm=100,
            recommended_protection_radius_mm=120,
            plant_age_days=24,
        ),
        human_approved=True,
    )
    assert result["status"] == "applied"
    assert client.requests[0].curve_id is None
    assert client.requests[0].assign_to_plant_ids == [1]


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
    assert client.statuses[-1].app_version == __version__


@pytest.mark.asyncio
async def test_calibrated_job_persists_overlay_vegetation_and_ownership_artifacts(
    tmp_path, monkeypatch
):
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
