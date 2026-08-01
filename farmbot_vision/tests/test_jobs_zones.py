"""Zone enforcement inside the analysis job (automatic writes)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest
from conftest import vision_image_dict

from farmbot_vision.database import Database
from farmbot_vision.jobs import JobManager, limit_weed_radius_growth
from farmbot_vision.models import (
    AnalysisResult,
    Decision,
    Inventory,
    Measurement,
    OperatingMode,
    VisionImage,
)
from farmbot_vision.settings import Settings
from farmbot_vision.weed_settings import WeedSettings, WeedSettingsStore
from farmbot_vision.zones import Zone, ZoneKind, ZoneShape, ZoneStore

IMAGE_WIDTH, IMAGE_HEIGHT = 96, 72
TIMESTAMP = datetime(2026, 2, 1, tzinfo=UTC)


def test_known_weed_radius_growth_uses_smaller_rolling_24_hour_cap():
    target, ceiling = limit_weed_radius_growth(
        current_radius_mm=18,
        measured_radius_mm=60,
        rolling_baseline_radius_mm=15,
        maximum_growth_mm=20,
        maximum_growth_percent=40,
    )

    assert ceiling == pytest.approx(21)
    assert target == pytest.approx(21)

    # A second same-day view starts from the already widened radius but cannot
    # consume another 40%; the original rolling baseline still controls it.
    repeated, repeated_ceiling = limit_weed_radius_growth(21, 65, 15, 20, 40)
    assert repeated_ceiling == pytest.approx(21)
    assert repeated == pytest.approx(21)


def _image_payload() -> dict:
    """A processed image with its own calibration: 1 px/mm, no rotation.

    Garden coordinates therefore span X -48…48 mm and Y -36…36 mm around the
    photo centre at (0, 0).
    """
    return vision_image_dict(
        np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8),
        image_id=11,
        processed_calibration={
            "available": True,
            "basis": "processed_image",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "pixels_per_mm_x": 1.0,
            "pixels_per_mm_y": 1.0,
            "rotation_degrees": 0.0,
            "offset_x_mm": 0.0,
            "offset_y_mm": 0.0,
        },
    )


def _inventory() -> Inventory:
    return Inventory.model_validate(
        {
            "device_id": "42",
            "generated_at": TIMESTAMP.isoformat(),
            "plants": [
                {
                    "id": 1,
                    "name": "Thyme",
                    "openfarm_slug": "thyme",
                    "x": -20.0,
                    "y": 0.0,
                    "radius": 30.0,
                    "plant_stage": "planted",
                    "planted_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            "images": [
                {
                    "id": 11,
                    "created_at": TIMESTAMP.isoformat(),
                    "processed": True,
                    "meta": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
            ],
            "curves": [],
            "weeds": [],
            "camera_calibration": {"available": False},
        }
    )


class _Client:
    def __init__(self):
        self.created_weeds: list = []
        self.applied_radius: list = []

    async def inventory(self, _request):
        return _inventory()

    async def image(self, _request, _max_bytes):
        return VisionImage.model_validate(_image_payload())

    async def create_weed(self, request):
        self.created_weeds.append(request)
        return {"status": "applied", "message": "weed created"}

    async def apply_radius(self, request):
        self.applied_radius.append(request)
        return {"status": "applied", "message": "radius updated"}

    async def report_status(self, _status):
        return {"status": "ok"}

    async def upsert_curve(self, _request):
        return {"status": "applied", "curve_id": 5, "message": "curve updated"}


def _measurement() -> Measurement:
    return Measurement(
        measurement_id=uuid4(),
        plant_id=1,
        crop_slug="thyme",
        image_id=11,
        image_timestamp=TIMESTAMP,
        current_radius_mm=30,
        typical_canopy_radius_mm=100,
        maximum_accepted_canopy_radius_mm=120,
        recommended_protection_radius_mm=150,
        confidence=0.99,
        decision=Decision.APPLIED,
        reason="canopy grew",
        algorithm_version="test",
    )


def _analysis_result(
    weed_center_px: tuple[float, float], features: dict[str, float] | None = None
) -> AnalysisResult:
    return AnalysisResult(
        measurements=[_measurement()],
        weeds=[
            {
                "detection_id": uuid4(),
                "image_id": 11,
                "image_timestamp": TIMESTAMP,
                "center_px": weed_center_px,
                "area_mm2": 150.0,
                "radius_mm": 12.0,
                "confidence": 0.99,
                "features": features or {},
            }
        ],
    )


def _manager(
    tmp_path,
    monkeypatch,
    *,
    zones: list[Zone],
    weed_center_px,
    weed_features: dict[str, float] | None = None,
) -> JobManager:
    from farmbot_vision import jobs as jobs_module

    store = ZoneStore(tmp_path / "zones.json")
    for zone in zones:
        store.add(zone)
    weed_settings_store = WeedSettingsStore(tmp_path / "weed_settings.json")
    # This suite isolates zone gating, so deliberately lower the independent
    # temporal/verifier automation gates tested by the weed pipeline suite.
    weed_settings_store.save(
        WeedSettings(
            enabled=True,
            automatic_creation=True,
            temporal_confirmation_enabled=False,
            visual_verifier_required_for_automatic=False,
        )
    )
    manager = JobManager(
        Settings(data_dir=tmp_path, minimum_auto_confidence=0.5),
        Database(tmp_path / "db.sqlite"),
        _Client(),
        weed_settings_store,
        store,
    )
    monkeypatch.setattr(manager, "resources_available", lambda: (True, "ok"))
    # The engine and the safety gate are exercised elsewhere; here the job is
    # driven with a fixed result so only the zone decision varies.
    monkeypatch.setattr(
        jobs_module.ClassicalVisionEngine,
        "analyse",
        lambda *args, **kwargs: _analysis_result(weed_center_px, weed_features),
    )
    monkeypatch.setattr(jobs_module, "decide", lambda item, *args, **kwargs: item)
    return manager


def _exclusion_east() -> Zone:
    """Forbids everything from X = 0 mm eastwards."""
    return Zone(
        name="Concrete path",
        kind=ZoneKind.EXCLUSION,
        shape=ZoneShape.RECTANGLE,
        min_x=0,
        min_y=-500,
        max_x=500,
        max_y=500,
    )


@pytest.mark.asyncio
async def test_weed_inside_an_exclusion_zone_is_never_stored_or_created(tmp_path, monkeypatch):
    # Pixel (68, 36) is 20 mm east of the photo centre, inside the zone.
    manager = _manager(
        tmp_path, monkeypatch, zones=[_exclusion_east()], weed_center_px=(68.0, 36.0)
    )
    result = await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)

    assert result["accepted"] is True
    assert result["zone_blocked_weeds"] == 1
    assert manager.db.pending_weed_detections() == []
    assert manager.client.created_weeds == []


@pytest.mark.asyncio
async def test_weed_outside_the_exclusion_zone_is_still_created_automatically(
    tmp_path, monkeypatch
):
    # Pixel (28, 36) is 20 mm west of the photo centre, clear of the zone.
    manager = _manager(
        tmp_path, monkeypatch, zones=[_exclusion_east()], weed_center_px=(28.0, 36.0)
    )
    result = await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)

    assert result["zone_blocked_weeds"] == 0
    assert len(manager.client.created_weeds) == 1
    assert manager.client.created_weeds[0].x == pytest.approx(-20.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "safety_feature",
    ["crop_protection_overlap", "configured_maximum_area_exceeded"],
)
async def test_recall_only_candidate_is_reviewable_but_never_created_automatically(
    tmp_path, monkeypatch, safety_feature
):
    manager = _manager(
        tmp_path,
        monkeypatch,
        zones=[],
        weed_center_px=(28.0, 36.0),
        weed_features={safety_feature: 1.0},
    )

    await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)

    assert manager.client.created_weeds == []
    assert len(manager.db.pending_weed_detections()) == 1


@pytest.mark.asyncio
async def test_automatic_radius_growth_into_a_zone_stays_a_recommendation(tmp_path, monkeypatch):
    # The plant sits at X -20 mm, so a 150 mm radius reaches into the zone.
    manager = _manager(
        tmp_path, monkeypatch, zones=[_exclusion_east()], weed_center_px=(28.0, 36.0)
    )
    result = await manager.run(entry_id="bot", mode=OperatingMode.AUTO_RADIUS)

    assert result["zone_blocked_radius"] == 1
    assert manager.client.applied_radius == []
    row = next(iter(manager.db.pending_measurements()))
    assert row["decision"] == Decision.RECOMMENDED.value
    assert "radius growth blocked" in row["reason"]
    assert "Concrete path" in row["reason"]


@pytest.mark.asyncio
async def test_automatic_radius_growth_is_applied_when_no_zone_objects(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch, zones=[], weed_center_px=(28.0, 36.0))
    result = await manager.run(entry_id="bot", mode=OperatingMode.AUTO_RADIUS)

    assert result["zone_blocked_radius"] == 0
    assert len(manager.client.applied_radius) == 1
    assert manager.client.applied_radius[0].recommended_radius_mm == 150
