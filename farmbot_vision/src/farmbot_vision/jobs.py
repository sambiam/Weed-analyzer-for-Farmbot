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

from . import CONTRACT_VERSION, __version__
from .calibration import resolve_calibration
from .curves import fit_monotonic_curve
from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError, StaleRadiusError
from .models import (
    ApplyRadiusRequest,
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
        image_ids: list[int] | None = None,
        trigger: str = "manual",
        queue_if_busy: bool = False,
    ) -> dict:
        if self.lock.locked():
            self.current["queue_length"] = self.current.get("queue_length", 0) + 1
            if not queue_if_busy:
                LOGGER.info("Analysis request rejected: another analysis is already running")
                return {"accepted": False, "reason": "analysis already running"}
            LOGGER.info("Analysis request queued behind the running analysis")
        entry_id = entry_id or self.settings.selected_config_entry_id
        mode = mode or self.settings.mode
        if not entry_id:
            return {"accepted": False, "reason": "select a FarmBot before analysis"}
        async with self.lock:
            return await self._run_locked(entry_id, mode, plant_ids or [], image_ids or [], trigger)

    async def _run_locked(
        self,
        entry_id: str,
        mode: OperatingMode,
        plant_ids: list[int],
        image_ids: list[int],
        trigger: str,
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
            resolution = self.settings.resolution
            manual_calibration = self.db.active_calibration(entry_id)
            engine = ClassicalVisionEngine(
                self.settings.safety_margin_mm, self.settings.calibration_uncertainty_mm
            )
            wanted = [p for p in inventory.plants if not plant_ids or p.id in plant_ids]
            wanted_image_ids = set(image_ids)
            images = [
                image
                for image in sorted(inventory.images, key=lambda item: item.created_at)
                if image.processed and (not wanted_image_ids or image.id in wanted_image_ids)
            ]
            if wanted_image_ids and not images:
                return await self._finish(
                    entry_id,
                    job_id,
                    "warning",
                    start_wall,
                    start_cpu,
                    [],
                    "requested image is not yet available",
                )
            all_measurements = []
            self.current["resolution"] = resolution.as_dict()
            self.current["images_processed"] = 0
            self.current["uncalibrated_images"] = 0
            self.current["calibration_warnings"] = []
            self.current["calibration_source"] = None
            artifacts = self.settings.data_dir / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            for image_number, image_info in enumerate(images):
                available, resource_reason = self.resources_available()
                if not available:
                    LOGGER.warning("Analysis paused: %s", resource_reason)
                    break
                self.current["progress"] = f"Processing image {image_number + 1}/{len(images)}"
                # Request the configured resolution; images are fetched one at a time.
                response = await self.client.image(
                    VisionImageRequest(
                        config_entry_id=entry_id,
                        image_id=image_info.id,
                        max_width=self.settings.analysis_width,
                        max_height=self.settings.analysis_height,
                    ),
                    self.settings.max_image_payload_bytes,
                )
                image_bytes = base64.b64decode(response.image_base64, validate=True)
                resolved = resolve_calibration(
                    response,
                    inventory.camera_calibration,
                    manual_calibration,
                    resolution,
                    self.settings.calibration_uncertainty_mm,
                )
                self.current["calibration_source"] = resolved.source
                for warning in resolved.warnings:
                    if warning not in self.current["calibration_warnings"]:
                        self.current["calibration_warnings"].append(warning)
                self.current["source_dimensions"] = (
                    [response.source_width, response.source_height]
                    if response.source_width
                    else None
                )
                self.current["oriented_dimensions"] = (
                    [response.oriented_width, response.oriented_height]
                    if response.oriented_width
                    else None
                )
                self.current["processed_dimensions"] = [response.width, response.height]
                self.current["resize_scales"] = (
                    [response.resize_scale_x, response.resize_scale_y]
                    if response.resize_scale_x
                    else None
                )
                self.current["images_processed"] += 1

                overlay_path = artifacts / f"{job_id}-{response.image_id}-overlay.jpg"

                if resolved.calibration is None:
                    # No valid metric calibration: pixel-only diagnostics, no
                    # measurement, no write (Part 6).
                    self.current["uncalibrated_images"] += 1
                    result = await asyncio.wait_for(
                        asyncio.to_thread(engine.diagnostic_only, image_bytes),
                        timeout=60,
                    )
                    if result.overlay_jpeg:
                        overlay_path.write_bytes(result.overlay_jpeg)
                    if result.mask:
                        (artifacts / f"{job_id}-{response.image_id}-mask.png").write_bytes(
                            result.mask
                        )
                    del image_bytes, result
                    continue

                calibration = self.db.record_calibration(entry_id, resolved.calibration)
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
                del image_bytes, previous_masks
                decided = [decide(item, mode, self.settings) for item in result.measurements]
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
                skip_reasons = self.current.setdefault("skip_reasons", {})
                for plant_id, reason in result.skipped.items():
                    skip_reasons[str(plant_id)] = reason
                del result
                if mode == OperatingMode.AUTO_RADIUS:
                    for item in decided:
                        # Never write without a valid calibration (Part 6).
                        if item.decision != Decision.APPLIED or not item.calibrated:
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
            LOGGER.exception("Analysis failed for entry_id=%s: %s", entry_id, exc)
            return await self._finish(
                entry_id,
                job_id,
                "error",
                start_wall,
                start_cpu,
                [],
                f"analysis failed: {type(exc).__name__}",
            )

    async def _status(
        self,
        entry_id: str,
        job_id: UUID | None,
        status: str,
        message: str,
        measurements: list | None = None,
    ) -> None:
        measurements = measurements or []
        LOGGER.debug(
            "Reporting status to Home Assistant: entry_id=%s job_id=%s status=%s message=%s",
            entry_id,
            job_id,
            status,
            message,
        )
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
                    app_version=__version__,
                )
            )
        except HomeAssistantError as exc:
            # If this keeps failing, HA-side entities (Vision Available, Vision
            # Status, ...) will never leave their unavailable/disconnected state
            # even though jobs are running -- the reason is always logged here.
            LOGGER.warning(
                "Could not report job status to Home Assistant: entry_id=%s status=%s (%s): %s",
                entry_id,
                status,
                type(exc).__name__,
                exc,
            )

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
        skip_reasons = self.current.get("skip_reasons", {})
        crop_slugs = sorted({measurement.crop_slug for measurement in measurements})
        spread_curves = {
            slug: fit_monotonic_curve(
                self.db.measurements_for_crop(slug),
                safety_margin_mm=self.settings.safety_margin_mm,
            )
            for slug in crop_slugs
        }
        result = {
            "id": str(job_id),
            "status": status,
            "message": message,
            "completed_at": datetime.now(UTC).isoformat(),
            "plants_analysed": len(measurements),
            "plants_measured": sum(1 for m in measurements if m.calibrated),
            "recommendations": sum(m.decision == Decision.RECOMMENDED for m in measurements),
            "automatically_applied": sum(m.decision == Decision.APPLIED for m in measurements),
            "uncertain": sum(m.decision == Decision.UNCERTAIN for m in measurements),
            "skipped": len(skip_reasons),
            "skip_reasons": skip_reasons,
            "cpu_seconds": round(time.process_time() - start_cpu, 3),
            "peak_memory_mb": round(peak, 1),
            "duration_seconds": round((datetime.now(UTC) - start_wall).total_seconds(), 3),
            "analysis_resolution": self.settings.resolution.as_dict(),
            "images_processed": self.current.get("images_processed", 0),
            "uncalibrated_images": self.current.get("uncalibrated_images", 0),
            "calibration_source": self.current.get("calibration_source"),
            "calibration_warnings": self.current.get("calibration_warnings", []),
            "source_dimensions": self.current.get("source_dimensions"),
            "oriented_dimensions": self.current.get("oriented_dimensions"),
            "processed_dimensions": self.current.get("processed_dimensions"),
            "resize_scales": self.current.get("resize_scales"),
            "contract_version": CONTRACT_VERSION,
            "spread_curves": spread_curves,
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
