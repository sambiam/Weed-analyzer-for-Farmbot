from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

import farmbot_vision.soil_legacy_repair as repair_module
from farmbot_vision.database import Database
from farmbot_vision.models import (
    SoilMeasurement,
    SoilMotionState,
    SoilPoint,
    SoilPointInventory,
    SoilStereoCalibration,
)
from farmbot_vision.soil_legacy_repair import LegacySoilRepairManager


def _calibration() -> SoilStereoCalibration:
    return SoilStereoCalibration(
        config_entry_id="legacy-entry",
        point_id=70,
        capture_z=0,
        baseline_mm=15,
        reference_distance_mm=550,
        z_direction=-1,
        inverse_depth_slope=0.001,
        inverse_depth_intercept=0.0001,
        residual_mm=3.7,
        processed_width=640,
        processed_height=480,
        source_width=640,
        source_height=480,
        source_image_ids=list(range(1, 10)),
        camera_signature="legacy-signature",
    )


def _inventory(z: float = -469) -> SoilPointInventory:
    return SoilPointInventory(
        device_id="42",
        generated_at=datetime.now(UTC),
        points=[
            SoilPoint(
                id=70,
                name="Legacy soil",
                x=100,
                y=200,
                z=z,
                updated_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 100, "y": 200, "z": 0},
            z_direction=-1,
            axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-600, 0)},
        ),
    )


@pytest.mark.asyncio
async def test_legacy_repair_stages_only_v2_then_applies_after_confirmation(tmp_path, monkeypatch):
    database = Database(tmp_path / "vision.db")
    calibration = database.save_soil_calibration(_calibration())
    legacy_id = uuid4()
    database.save_soil_measurement(
        SoilMeasurement(
            measurement_id=legacy_id,
            config_entry_id="legacy-entry",
            point_id=70,
            point_name="Legacy soil",
            expected_x=100,
            expected_y=200,
            old_z_mm=-469,
            point_updated_at=datetime(2026, 7, 1, tzinfo=UTC),
            capture_x=100,
            capture_y=200,
            proposed_z_mm=-469,
            confidence=0.8,
            uncertainty_mm=3.7,
            status="valid",
            reason="legacy result",
            calibration_id=calibration.calibration_id,
            frame_ids=[10, 11, 12],
            algorithm_version="soil-stereo-v2",
        )
    )
    database.save_soil_measurement(
        SoilMeasurement(
            measurement_id=uuid4(),
            config_entry_id="legacy-entry",
            point_id=71,
            point_name="Current soil",
            expected_x=300,
            expected_y=400,
            old_z_mm=-500,
            proposed_z_mm=-501,
            confidence=0.9,
            status="valid",
            reason="current result",
            algorithm_version="soil-stereo-v3",
        )
    )
    applied = []

    class Client:
        async def soil_points(self, _entry_id):
            return _inventory()

        async def apply_soil_height(self, request):
            applied.append(request)
            return {"status": "applied", "message": "corrected"}

    manager = LegacySoilRepairManager(database, Client())

    async def refit(_calibration):
        return _calibration

    async def frames(_row, _calibration):
        return [SimpleNamespace()]

    monkeypatch.setattr(manager, "_refit_calibration", refit)
    monkeypatch.setattr(manager, "_measurement_frames", frames)
    monkeypatch.setattr(
        repair_module,
        "analyse_soil_height",
        lambda _frames, _calibration: SimpleNamespace(
            valid=True,
            proposed_z_mm=-505.0,
            confidence=0.91,
            uncertainty_mm=4.2,
            reason="passed",
        ),
    )

    await manager.scan_now("legacy-entry")
    pending = database.pending_legacy_soil_repairs("legacy-entry")
    assert [item["legacy_measurement_id"] for item in pending] == [str(legacy_id)]
    assert pending[0]["delta_mm"] == pytest.approx(-36)
    assert database.legacy_soil_repair_summary("legacy-entry")["eligible"] == 1
    assert not applied

    # The durable classification makes a second scan a no-op and prevents the
    # migration from becoming a reusable reprocessing path.
    await manager.scan_now("legacy-entry")
    assert len(database.pending_legacy_soil_repairs("legacy-entry")) == 1

    await manager.apply_selected("legacy-entry", [str(legacy_id)])
    assert len(applied) == 1
    assert applied[0].recommended_z_mm == -505
    assert applied[0].human_approved is True
    assert database.legacy_soil_repair(str(legacy_id))["state"] == "applied"
    assert database.legacy_soil_repair_summary("legacy-entry")["retired"] is True


def test_legacy_repair_rejection_is_terminal(tmp_path):
    database = Database(tmp_path / "vision.db")
    calibration = database.save_soil_calibration(_calibration())
    measurement_id = uuid4()
    database.save_soil_measurement(
        SoilMeasurement(
            measurement_id=measurement_id,
            config_entry_id="legacy-entry",
            point_id=70,
            point_name="Legacy soil",
            expected_x=100,
            expected_y=200,
            old_z_mm=-505,
            proposed_z_mm=-469,
            confidence=0.8,
            status="applied",
            reason="legacy result",
            calibration_id=calibration.calibration_id,
            frame_ids=[10, 11, 12],
            algorithm_version="soil-stereo-v2",
        )
    )
    database.save_legacy_soil_repair(
        legacy_measurement_id=str(measurement_id),
        config_entry_id="legacy-entry",
        source_status="applied",
        state="pending",
        old_proposed_z_mm=-469,
        repaired_z_mm=-505,
        confidence=0.9,
        uncertainty_mm=4,
        reason="staged",
    )
    manager = LegacySoilRepairManager(database, SimpleNamespace())

    manager.reject_selected("legacy-entry", [str(measurement_id)])

    assert database.legacy_soil_repair(str(measurement_id))["state"] == "rejected"
    assert database.legacy_soil_repair_summary("legacy-entry")["retired"] is True
