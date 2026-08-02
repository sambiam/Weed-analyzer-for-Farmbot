"""Verifier authority and repeated-evidence gates for known weed automation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest
from conftest import vision_image_dict

from farmbot_vision.database import Database
from farmbot_vision.jobs import JobManager
from farmbot_vision.models import AnalysisResult, Inventory, OperatingMode, VisionImage
from farmbot_vision.settings import Settings
from farmbot_vision.weed_settings import WeedSettings, WeedSettingsStore

TIMESTAMP = datetime(2026, 8, 1, tzinfo=UTC)
WIDTH, HEIGHT = 320, 240


def _inventory() -> Inventory:
    return Inventory.model_validate(
        {
            "device_id": "42",
            "generated_at": TIMESTAMP.isoformat(),
            "plants": [],
            "images": [
                {
                    "id": image_id,
                    "created_at": (TIMESTAMP + timedelta(minutes=index)).isoformat(),
                    "processed": True,
                    "meta": {"x": 0, "y": 0, "z": 0},
                }
                for index, image_id in enumerate((11, 12))
            ],
            "curves": [],
            "weeds": [{"id": 91, "x": 0, "y": 0, "z": 0, "radius": 15}],
            "camera_calibration": {"available": False},
        }
    )


class _Client:
    def __init__(self):
        self.radius_updates = []
        self.removals = []

    async def inventory(self, _request):
        return _inventory()

    async def image(self, request, _max_bytes):
        return VisionImage.model_validate(
            vision_image_dict(
                np.zeros((HEIGHT, WIDTH, 3), np.uint8),
                image_id=request.image_id,
                processed_calibration={
                    "available": True,
                    "basis": "processed_image",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "pixels_per_mm_x": 1,
                    "pixels_per_mm_y": 1,
                    "rotation_degrees": 0,
                    "offset_x_mm": 0,
                    "offset_y_mm": 0,
                },
            )
        )

    async def update_weed_radius(self, request):
        self.radius_updates.append(request)
        return {"status": "applied", "radius_mm": request.recommended_radius_mm}

    async def remove_weed(self, request):
        self.removals.append(request)
        return {"status": "applied", "weed_id": request.weed_id}

    async def report_status(self, _status):
        return {"status": "ok"}


def _analysis(image_id: int, observation_status: str, *, include_detection: bool) -> AnalysisResult:
    timestamp = TIMESTAMP + timedelta(minutes=image_id - 11)
    weeds = []
    if include_detection:
        weeds.append(
            {
                "detection_id": uuid4(),
                "image_id": image_id,
                "image_timestamp": timestamp,
                "center_px": (WIDTH / 2, HEIGHT / 2),
                "area_mm2": 600,
                "radius_mm": 30,
                "confidence": 0.96,
                "heuristic_confidence": 0.50,
                "verifier_confidence": 0.96,
                "features": {"known_weed_id": 91},
            }
        )
    return AnalysisResult(
        measurements=[],
        weeds=weeds,
        known_weed_observations=[
            {
                "weed_id": 91,
                "status": observation_status,
                "confidence": 0.96 if observation_status != "inconclusive" else 0,
                "verifier_confidence": 0.96 if observation_status == "present" else None,
                "verifier_evaluated": observation_status == "present",
                "reason": "test evidence",
            }
        ],
    )


def _manager(tmp_path, monkeypatch, settings: WeedSettings, result_factory) -> JobManager:
    from farmbot_vision import jobs as jobs_module

    store = WeedSettingsStore(tmp_path / "weed_settings.json")
    store.save(settings)
    manager = JobManager(
        Settings(data_dir=tmp_path),
        Database(tmp_path / "vision.sqlite"),
        _Client(),
        store,
    )
    monkeypatch.setattr(manager, "resources_available", lambda: (True, "ok"))

    def analyse(_engine, _image_bytes, image_id, *_args, **_kwargs):
        return result_factory(image_id)

    monkeypatch.setattr(jobs_module.ClassicalVisionEngine, "analyse", analyse)
    return manager


@pytest.mark.asyncio
async def test_no_final_detection_is_not_treated_as_known_weed_absence(tmp_path, monkeypatch):
    manager = _manager(
        tmp_path,
        monkeypatch,
        WeedSettings(
            enabled=True,
            automatic_removal=True,
            removal_min_consecutive_absent=2,
            visual_verifier_enabled=True,
            visual_verifier_shadow_mode=False,
        ),
        lambda image_id: _analysis(image_id, "inconclusive", include_detection=False),
    )

    await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)

    assert manager.client.removals == []
    track = manager.db.weed_track("bot", 91)
    assert track["status"] == "inconclusive"
    assert track["absent_observations"] == 0


@pytest.mark.asyncio
async def test_two_explicit_absent_images_can_remove_a_known_weed(tmp_path, monkeypatch):
    manager = _manager(
        tmp_path,
        monkeypatch,
        WeedSettings(
            enabled=True,
            automatic_removal=True,
            removal_min_consecutive_absent=2,
            visual_verifier_enabled=True,
            visual_verifier_shadow_mode=False,
        ),
        lambda image_id: _analysis(image_id, "absent", include_detection=False),
    )

    await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)

    assert len(manager.client.removals) == 1
    assert manager.client.removals[0].confidence == 0.96

    # Re-analysing the same historical images must not manufacture a longer
    # streak or send the destructive service call again while inventory syncs.
    await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)
    assert len(manager.client.removals) == 1


@pytest.mark.asyncio
async def test_radius_widening_requires_repeated_enforcing_verifier_acceptance(
    tmp_path, monkeypatch
):
    manager = _manager(
        tmp_path,
        monkeypatch,
        WeedSettings(
            enabled=True,
            automatic_radius_adjustment=True,
            radius_min_consecutive_present=2,
            visual_verifier_enabled=True,
            visual_verifier_shadow_mode=False,
            maximum_radius_growth_mm_per_day=100,
            maximum_radius_growth_percent_per_day=200,
        ),
        lambda image_id: _analysis(image_id, "present", include_detection=True),
    )

    await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)

    assert len(manager.client.radius_updates) == 1
    assert manager.client.radius_updates[0].recommended_radius_mm == 30
    assert manager.client.radius_updates[0].confidence == 0.96

    await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)
    assert len(manager.client.radius_updates) == 1


@pytest.mark.asyncio
async def test_shadow_verifier_cannot_widen_a_known_weed(tmp_path, monkeypatch):
    manager = _manager(
        tmp_path,
        monkeypatch,
        WeedSettings(
            enabled=True,
            automatic_radius_adjustment=True,
            radius_min_consecutive_present=1,
            visual_verifier_enabled=True,
            visual_verifier_shadow_mode=True,
        ),
        lambda image_id: _analysis(image_id, "present", include_detection=True),
    )

    await manager.run(entry_id="bot", mode=OperatingMode.RECOMMEND)

    assert manager.client.radius_updates == []
