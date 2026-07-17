from __future__ import annotations

import asyncio
import base64
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
import psutil

try:
    import resource
except ImportError:  # pragma: no cover - Windows development hosts
    resource = None

from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError, StaleRadiusError
from .models import (
    ApplyRadiusRequest,
    Calibration,
    CameraCalibration,
    Decision,
    InventoryRequest,
    OperatingMode,
    PlantSeed,
    VisionImageRequest,
    VisionStatus,
)
from .safety import decide
from .settings import Settings
from .vision import ClassicalVisionEngine, garden_to_pixel

LOGGER = logging.getLogger(__name__)


class JobManager:
    def __init__(self, settings: Settings, database: Database, client: HomeAssistantClient):
        self.settings = settings
        self.db = database
        self.client = client
        self.lock = asyncio.Lock()
        self.current: dict = {"status": "idle", "queue_length": 0, "progress": "Not run"}
        self.last: dict = {}

    def resources_available(self) -> tuple[bool, str]:
        memory_mb = psutil.virtual_memory().available / 1024 / 1024
        cpu = psutil.cpu_percent(interval=0.1)
        if memory_mb < self.settings.minimum_free_memory_mb:
            return False, f"free memory below {self.settings.minimum_free_memory_mb} MB"
        if cpu > self.settings.maximum_system_load_percent:
            return False, f"system CPU load above {self.settings.maximum_system_load_percent}%"
        return True, "resources available"

    async def run(
        self,
        entry_id: str | None = None,
        mode: OperatingMode | None = None,
        plant_ids: list[int] | None = None,
        trigger: str = "manual",
    ) -> dict:
        if self.lock.locked():
            self.current["queue_length"] = 1
            return {"accepted": False, "reason": "analysis already running"}
        entry_id = entry_id or self.settings.selected_config_entry_id
        mode = mode or self.settings.mode
        if not entry_id:
            return {"accepted": False, "reason": "select a FarmBot before analysis"}
        async with self.lock:
            return await self._run_locked(entry_id, mode, plant_ids or [], trigger)

    async def _run_locked(
        self, entry_id: str, mode: OperatingMode, plant_ids: list[int], trigger: str
    ) -> dict:
        job_id = uuid4()
        start_wall = datetime.now(UTC)
        start_cpu = time.process_time()
        self.current = {
            "id": str(job_id),
            "status": "running",
            "queue_length": 0,
            "progress": "Checking resources",
            "started_at": start_wall.isoformat(),
        }
        self.db.start_job(str(job_id), entry_id, trigger, mode.value, start_wall.isoformat())
        available, reason = self.resources_available()
        if not available:
            return await self._finish(
                entry_id, job_id, "warning", start_wall, start_cpu, [], reason
            )
        try:
            await self._status(entry_id, job_id, "running", "starting")
            inventory = await self.client.inventory(
                InventoryRequest(
                    config_entry_id=entry_id,
                    image_lookback_hours=self.settings.image_lookback_hours,
                )
            )
            self.current["progress"] = "Inventory loaded"
            calibration = self._calibration(entry_id, inventory.camera_calibration)
            if calibration is None:
                return await self._finish(
                    entry_id,
                    job_id,
                    "warning",
                    start_wall,
                    start_cpu,
                    [],
                    "calibration required; automatic updates refused",
                )
            engine = ClassicalVisionEngine(
                self.settings.safety_margin_mm, self.settings.calibration_uncertainty_mm
            )
            wanted = [p for p in inventory.plants if not plant_ids or p.id in plant_ids]
            all_measurements = []
            artifacts = self.settings.data_dir / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            for image_number, image_info in enumerate(
                sorted(inventory.images, key=lambda item: item.created_at)
            ):
                if not image_info.processed:
                    continue
                available, resource_reason = self.resources_available()
                if not available:
                    LOGGER.warning("Analysis paused: %s", resource_reason)
                    break
                self.current["progress"] = (
                    f"Processing image {image_number + 1}/{len(inventory.images)}"
                )
                response = await self.client.image(
                    VisionImageRequest(config_entry_id=entry_id, image_id=image_info.id),
                    self.settings.max_image_payload_bytes,
                )
                image_bytes = base64.b64decode(response.image_base64, validate=True)
                seeds = [
                    PlantSeed(
                        plant_id=plant.id,
                        crop_slug=plant.openfarm_slug,
                        center_px=garden_to_pixel(
                            plant.x,
                            plant.y,
                            response.meta.x,
                            response.meta.y,
                            response.width,
                            response.height,
                            calibration,
                        ),
                        current_radius_mm=plant.radius,
                        planted_at=plant.planted_at,
                    )
                    for plant in wanted
                ]
                previous_masks = {}
                for seed in seeds:
                    prior = decode_previous_mask(self.db.latest_mask_path(seed.plant_id))
                    if prior is not None:
                        previous_masks[seed.plant_id] = prior
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        engine.analyse,
                        image_bytes,
                        response.image_id,
                        response.meta.created_at,
                        seeds,
                        calibration,
                        previous_masks,
                    ),
                    timeout=60,
                )
                decided = [decide(item, mode, self.settings) for item in result.measurements]
                overlay_path = artifacts / f"{job_id}-{response.image_id}-overlay.jpg"
                if result.overlay_jpeg:
                    overlay_path.write_bytes(result.overlay_jpeg)
                ownership = None
                if result.mask:
                    ownership = cv2.imdecode(
                        np.frombuffer(result.mask, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                    )
                labelled = {seed.plant_id: index + 1 for index, seed in enumerate(seeds)}
                persisted = []
                for item in decided:
                    mask_path = artifacts / (
                        f"{job_id}-{response.image_id}-plant-{item.plant_id}-mask.png"
                    )
                    if ownership is not None:
                        cv2.imwrite(
                            str(mask_path),
                            (ownership == labelled[item.plant_id]).astype(np.uint8) * 255,
                        )
                    persisted.append(
                        item.model_copy(
                            update={
                                "overlay_path": str(overlay_path),
                                "mask_path": str(mask_path) if ownership is not None else None,
                            }
                        )
                    )
                decided = persisted
                self.db.save_measurements(decided)
                all_measurements.extend(decided)
                if mode == OperatingMode.AUTO_RADIUS:
                    for item in decided:
                        if item.decision != Decision.APPLIED:
                            continue
                        try:
                            await self.client.apply_radius(
                                ApplyRadiusRequest(
                                    config_entry_id=entry_id,
                                    plant_id=item.plant_id,
                                    measurement_id=item.measurement_id,
                                    expected_current_radius_mm=item.current_radius_mm,
                                    recommended_radius_mm=item.recommended_protection_radius_mm,
                                    confidence=item.confidence,
                                    apply=True,
                                )
                            )
                            self.db.record_decision(str(item.measurement_id), "applied", {})
                        except StaleRadiusError:
                            self.db.record_decision(str(item.measurement_id), "stale_radius", {})
                            await self.client.inventory(
                                InventoryRequest(
                                    config_entry_id=entry_id,
                                    image_lookback_hours=self.settings.image_lookback_hours,
                                )
                            )
            return await self._finish(
                entry_id, job_id, "idle", start_wall, start_cpu, all_measurements, "completed"
            )
        except Exception as exc:
            LOGGER.error("Analysis failed: %s", type(exc).__name__)
            return await self._finish(
                entry_id,
                job_id,
                "error",
                start_wall,
                start_cpu,
                [],
                f"analysis failed: {type(exc).__name__}",
            )

    def _calibration(self, entry_id: str, remote: CameraCalibration) -> Calibration | None:
        manual = self.db.active_calibration(entry_id)
        if remote.available:
            calibration = Calibration(
                source="farmbot",
                pixels_per_mm_x=remote.pixels_per_mm_x or 1,
                pixels_per_mm_y=remote.pixels_per_mm_y or 1,
                rotation_degrees=remote.rotation_degrees or 0,
                offset_x_mm=remote.offset_x_mm or 0,
                offset_y_mm=remote.offset_y_mm or 0,
                uncertainty_mm=self.settings.calibration_uncertainty_mm,
            )
            return self.db.save_calibration(entry_id, calibration)
        return manual

    async def _status(
        self,
        entry_id: str,
        job_id: UUID | None,
        status: str,
        message: str,
        measurements: list | None = None,
    ) -> None:
        measurements = measurements or []
        try:
            await self.client.report_status(
                VisionStatus(
                    config_entry_id=entry_id,
                    available=True,
                    status=status,
                    job_id=job_id,
                    last_completed_at=datetime.now(UTC) if status == "idle" else None,
                    plants_analysed=len(measurements),
                    recommendations=sum(m.decision == Decision.RECOMMENDED for m in measurements),
                    automatically_applied=sum(m.decision == Decision.APPLIED for m in measurements),
                    uncertain=sum(m.decision == Decision.UNCERTAIN for m in measurements),
                    message=message,
                )
            )
        except HomeAssistantError:
            LOGGER.warning("Could not report job status to Home Assistant")

    async def _finish(
        self,
        entry_id: str,
        job_id: UUID,
        status: str,
        start_wall: datetime,
        start_cpu: float,
        measurements: list,
        message: str,
    ) -> dict:
        if resource is not None:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        else:
            peak = psutil.Process().memory_info().rss / 1024 / 1024
        result = {
            "id": str(job_id),
            "status": status,
            "message": message,
            "completed_at": datetime.now(UTC).isoformat(),
            "plants_analysed": len(measurements),
            "recommendations": sum(m.decision == Decision.RECOMMENDED for m in measurements),
            "automatically_applied": sum(m.decision == Decision.APPLIED for m in measurements),
            "uncertain": sum(m.decision == Decision.UNCERTAIN for m in measurements),
            "cpu_seconds": round(time.process_time() - start_cpu, 3),
            "peak_memory_mb": round(peak, 1),
            "duration_seconds": round((datetime.now(UTC) - start_wall).total_seconds(), 3),
        }
        self.last = result
        self.current = {"status": "idle", "queue_length": 0, "progress": message}
        self.db.finish_job(str(job_id), result)
        await self._status(entry_id, job_id, status, message, measurements)
        return {"accepted": True, **result}


def decode_previous_mask(path: str | None) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)
