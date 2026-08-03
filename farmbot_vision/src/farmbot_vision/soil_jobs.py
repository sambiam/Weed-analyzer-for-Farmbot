"""Soil-height capture, calibration, review, and persistence orchestration."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
from pathlib import Path
from uuid import UUID, uuid4

from .database import Database
from .home_assistant import HomeAssistantClient
from .models import (
    InventoryRequest,
    SoilCaptureStartRequest,
    SoilMeasurement,
    SoilPoint,
    SoilPointInventory,
    SoilSite,
    VisionImageRequest,
)
from .photo_quality import inspect_photo_quality
from .soil_height import (
    ALGORITHM_VERSION as SOIL_ALGORITHM_VERSION,
)
from .soil_height import (
    SoilCalibrationQualityError,
    SoilFrame,
    SoilHeightError,
    analyse_soil_height,
    fit_calibration,
)
from .soil_sites import plan_safe_soil_sites
from .zones import ZoneStore

LOGGER = logging.getLogger(__name__)
SAFE_SITE_CACHE_SECONDS = 30.0


class SoilJobManager:
    """One resumability-safe, fail-closed soil workflow sharing the vision lock."""

    def __init__(
        self,
        database: Database,
        client: HomeAssistantClient,
        data_dir: Path,
        shared_lock: asyncio.Lock,
        zone_store: ZoneStore,
        soil_settings_store=None,
    ):
        self.db = database
        self.client = client
        self.data_dir = data_dir
        self.shared_lock = shared_lock
        self.zone_store = zone_store
        self.soil_settings_store = soil_settings_store
        self.task: asyncio.Task | None = None
        self.stop_requested = False
        self.current: dict = {"status": "idle", "message": "Not run"}
        self._pending_calibration_override: dict | None = None
        self._safe_site_cache: dict[
            tuple[str, float, float], tuple[float, tuple[SoilPointInventory, list[SoilSite]]]
        ] = {}
        self._safe_site_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def pending_override_job_id(self) -> str | None:
        """The failed calibration job, if any, waiting on an override decision."""
        return (
            self._pending_calibration_override["job_id"]
            if self._pending_calibration_override
            else None
        )

    def _start(self, coroutine, *, name: str) -> str:
        if self.running:
            raise ValueError("a soil-height job is already running")
        job_id = str(uuid4())
        self.stop_requested = False
        self.current = {
            "id": job_id,
            "status": "queued",
            "message": "Waiting for the vision job lock",
            "completed_count": 0,
            "failed_count": 0,
        }
        self.task = asyncio.create_task(coroutine(job_id), name=f"{name}-{job_id}")
        return job_id

    def start_calibration(
        self,
        *,
        config_entry_id: str,
        point_id: int | None,
        capture_x: float | None = None,
        capture_y: float | None = None,
        capture_z: float,
        baseline_mm: float,
        reference_distance_mm: float,
    ) -> str:
        # A fresh capture invalidates any earlier failure's frames.
        self._pending_calibration_override = None
        return self._start(
            lambda job_id: self._run_calibration(
                job_id=job_id,
                config_entry_id=config_entry_id,
                point_id=point_id,
                capture_x=capture_x,
                capture_y=capture_y,
                capture_z=capture_z,
                baseline_mm=baseline_mm,
                reference_distance_mm=reference_distance_mm,
            ),
            name="soil-calibration",
        )

    def start_calibration_override(self) -> str:
        """Recompute and accept the last failed calibration's images anyway.

        Uses the frames already captured for the calibration job that most
        recently failed a numeric quality gate (``SoilCalibrationQualityError``),
        so accepting the override never triggers another bot movement.
        """
        pending = self._pending_calibration_override
        if pending is None:
            raise ValueError("no calibration is waiting for an override decision")
        return self._start(
            lambda job_id: self._run_calibration_override(job_id=job_id, pending=pending),
            name="soil-calibration-override",
        )

    def start_measurements(
        self,
        *,
        config_entry_id: str,
        point_ids: list[int],
        custom_point_id: int | None = None,
        custom_x: float | None = None,
        custom_y: float | None = None,
        capture_z: float,
        baseline_mm: float,
    ) -> str:
        if custom_point_id is None and not point_ids:
            raise ValueError("select at least one soil point")
        if (custom_x is None) != (custom_y is None):
            raise ValueError("custom measurement X and Y must be supplied together")
        if custom_point_id is not None and custom_x is None:
            raise ValueError("custom measurement requires both coordinates")
        return self._start(
            lambda job_id: self._run_measurements(
                job_id=job_id,
                config_entry_id=config_entry_id,
                point_ids=list(dict.fromkeys(point_ids)),
                custom_point_id=custom_point_id,
                custom_x=custom_x,
                custom_y=custom_y,
                capture_z=capture_z,
                baseline_mm=baseline_mm,
            ),
            name="soil-measurement",
        )

    def request_stop(self) -> None:
        """Stop before the next point; never interrupt the bot's atomic RPC."""
        self.stop_requested = True
        self.current["message"] = "Stopping after the current soil point"
        job_id = self.current.get("id")
        if job_id:
            self.db.update_soil_job(job_id, stop_requested=True, message=self.current["message"])

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    @staticmethod
    def _nearest_order(points: list[SoilPoint], start_x: float, start_y: float) -> list[SoilPoint]:
        remaining, ordered = list(points), []
        x, y = start_x, start_y
        while remaining:
            point = min(remaining, key=lambda item: math.hypot(item.x - x, item.y - y))
            remaining.remove(point)
            ordered.append(point)
            x, y = point.x, point.y
        return ordered

    @staticmethod
    def _nearest_site_order(
        sites: list[SoilSite], start_x: float, start_y: float
    ) -> list[SoilSite]:
        remaining, ordered = list(sites), []
        x, y = start_x, start_y
        while remaining:
            site = min(
                remaining,
                key=lambda item: math.hypot(item.capture_x - x, item.capture_y - y),
            )
            remaining.remove(site)
            ordered.append(site)
            x, y = site.capture_x, site.capture_y
        return ordered

    async def safe_sites(
        self,
        config_entry_id: str,
        baseline_mm: float,
        *,
        clear_soil_margin_mm: float = 75,
        refresh: bool = False,
    ) -> tuple[SoilPointInventory, list[SoilSite]]:
        key = (config_entry_id, round(float(baseline_mm), 3), round(float(clear_soil_margin_mm), 3))
        cached = self._safe_site_cache.get(key)
        if not refresh and cached and time.monotonic() - cached[0] < SAFE_SITE_CACHE_SECONDS:
            return cached[1]
        async with self._safe_site_lock:
            cached = self._safe_site_cache.get(key)
            if not refresh and cached and time.monotonic() - cached[0] < SAFE_SITE_CACHE_SECONDS:
                return cached[1]
            soil, garden = await asyncio.gather(
                self.client.soil_points(config_entry_id),
                self.client.inventory(InventoryRequest(config_entry_id=config_entry_id)),
            )
            result = (
                soil,
                plan_safe_soil_sites(
                    soil,
                    garden,
                    self.db.current_vision_plants(config_entry_id),
                    self.db.current_vision_weeds(config_entry_id),
                    self.zone_store.zones(),
                    baseline_mm=baseline_mm,
                    clear_soil_margin_mm=clear_soil_margin_mm,
                ),
            )
            self._safe_site_cache[key] = (time.monotonic(), result)
            return result

    def cached_safe_sites(
        self,
        config_entry_id: str,
        baseline_mm: float,
        *,
        clear_soil_margin_mm: float = 75,
    ) -> tuple[SoilPointInventory, list[SoilSite]] | None:
        """Return the last successful plan even when it is due for refresh.

        This is for responsive, read-only UI rendering only. Job execution
        continues to call ``safe_sites`` after invalidating its cache so stale
        coordinates are never used for motion.
        """
        key = (
            config_entry_id,
            round(float(baseline_mm), 3),
            round(float(clear_soil_margin_mm), 3),
        )
        cached = self._safe_site_cache.get(key)
        return cached[1] if cached else None

    def invalidate_safe_sites(self, config_entry_id: str, baseline_mm: float) -> None:
        """Force the safety-critical job path to fetch a fresh live plan."""
        prefix = (config_entry_id, round(float(baseline_mm), 3))
        for key in [key for key in self._safe_site_cache if key[:2] == prefix]:
            self._safe_site_cache.pop(key, None)

    @staticmethod
    def _declared_camera_signature(images) -> str:
        signatures = {
            json.dumps(
                image.processed_calibration.model_dump(mode="json")
                if image.processed_calibration
                else {},
                sort_keys=True,
                separators=(",", ":"),
            )
            for image in images
        }
        if len(signatures) != 1:
            raise SoilHeightError("camera calibration metadata changed within the capture")
        return signatures.pop()

    @staticmethod
    def _validate_soil_geometry(
        actual: tuple[int, int, int, int],
        expected: tuple[int, int, int, int] | None,
    ) -> None:
        """Accept calibration geometry and bind measurements to that exact format."""
        if any(value <= 0 for value in actual):
            raise SoilHeightError("soil image reported invalid geometry")
        if expected is not None and actual != expected:
            raise SoilHeightError(
                "soil image geometry changed from the soil-height calibration: "
                f"expected {expected[0]}x{expected[1]} processed from "
                f"{expected[2]}x{expected[3]} source, received "
                f"{actual[0]}x{actual[1]} processed from {actual[2]}x{actual[3]} source"
            )

    async def _capture_frames(
        self,
        *,
        config_entry_id: str,
        point_id: int | None,
        capture_x: float,
        capture_y: float,
        capture_z: float,
        baseline_mm: float,
        z_offsets_mm: list[float],
        expected_geometry: tuple[int, int, int, int] | None = None,
        batch_id: UUID | None = None,
    ) -> tuple[UUID, list[SoilFrame], str]:
        busy_deadline = time.monotonic() + 600
        busy_logged = False
        while True:
            started = await self.client.start_soil_capture(
                SoilCaptureStartRequest(
                    config_entry_id=config_entry_id,
                    point_id=point_id,
                    capture_x=capture_x,
                    capture_y=capture_y,
                    capture_z=capture_z,
                    baseline_mm=baseline_mm,
                    z_offsets_mm=z_offsets_mm,
                    batch_id=batch_id,
                )
            )
            if started.status == "queued" and started.capture_id is not None:
                break
            if "busy" not in started.message.lower():
                raise SoilHeightError(started.message)
            if time.monotonic() >= busy_deadline:
                raise SoilHeightError("FarmBot remained busy for 10 minutes")
            self.current["message"] = "Waiting for FarmBot to finish its current operation"
            if not busy_logged:
                LOGGER.info("FarmBot reported busy; waiting to start the soil capture")
                busy_logged = True
            await asyncio.sleep(2)
        # Each frame is verified before the integration advances and can be
        # retried five times. Allow 75 seconds per attempt plus a ten-minute
        # floor for motion/restoration without timing out a valid calibration.
        status_deadline = time.monotonic() + max(600, 3 * len(z_offsets_mm) * 5 * 75)
        last_status_message = ""
        while time.monotonic() < status_deadline:
            status = await self.client.soil_capture_status(config_entry_id, str(started.capture_id))
            self.current["message"] = status.message
            if status.message != last_status_message:
                LOGGER.info("Soil capture %s: %s", started.capture_id, status.message)
                last_status_message = status.message
            if status.status == "failed":
                raise SoilHeightError(status.message)
            if status.status == "complete":
                break
            await asyncio.sleep(2)
        else:
            raise SoilHeightError("soil capture status timed out")
        expected_count = 3 * len(z_offsets_mm)
        if len(status.frames) != expected_count:
            raise SoilHeightError(
                f"FarmBot returned {len(status.frames)} of {expected_count} soil images"
            )
        images = []
        frames = []
        request_width = expected_geometry[0] if expected_geometry is not None else 1280
        request_height = expected_geometry[1] if expected_geometry is not None else 960
        for item in status.frames:
            response = await self.client.image(
                VisionImageRequest(
                    config_entry_id=config_entry_id,
                    image_id=item.image_id,
                    max_width=request_width,
                    max_height=request_height,
                ),
                5 * 1024 * 1024,
            )
            if not response.full_metadata:
                raise SoilHeightError("soil image lacks contract-v2 geometry")
            actual_geometry = (
                response.width,
                response.height,
                int(response.source_width),
                int(response.source_height),
            )
            self._validate_soil_geometry(actual_geometry, expected_geometry)
            jpeg = base64.b64decode(response.image_base64)
            coordinate_error = math.sqrt(
                (response.meta.x - item.x) ** 2
                + (response.meta.y - item.y) ** 2
                + (response.meta.z - item.z) ** 2
            )
            if coordinate_error > 5:
                raise SoilHeightError(
                    f"soil image {response.image_id} coordinates missed the requested frame "
                    f"by {coordinate_error:.1f} mm"
                )
            quality = await asyncio.to_thread(inspect_photo_quality, jpeg)
            if quality.issue != "usable":
                issue = quality.issue.replace("_", " ")
                raise SoilHeightError(
                    f"soil image {response.image_id} failed final quality validation: {issue}"
                )
            LOGGER.info(
                "Accepted soil image %s at X %.1f Y %.1f Z %.1f "
                "(coordinate error %.2f mm, integration attempt %s)",
                response.image_id,
                response.meta.x,
                response.meta.y,
                response.meta.z,
                coordinate_error,
                item.capture_attempt or "legacy",
            )
            images.append(response)
            frames.append(
                SoilFrame(
                    image_id=response.image_id,
                    jpeg=jpeg,
                    x=item.x,
                    y=item.y,
                    z=item.z,
                    lateral_offset_mm=item.lateral_offset_mm,
                    z_offset_mm=item.z_offset_mm,
                    processed_width=response.width,
                    processed_height=response.height,
                    source_width=int(response.source_width),
                    source_height=int(response.source_height),
                )
            )
        return started.capture_id, frames, self._declared_camera_signature(images)

    async def _finish_capture_batch(self, config_entry_id: str, batch_id: UUID) -> None:
        response = await self.client.finish_soil_capture_batch(config_entry_id, str(batch_id))
        if response.get("status") != "complete":
            raise SoilHeightError(
                str(response.get("message") or "FarmBot starting position was not restored")
            )

    async def _run_calibration(
        self,
        *,
        job_id: str,
        config_entry_id: str,
        point_id: int | None,
        capture_z: float,
        baseline_mm: float,
        reference_distance_mm: float,
        capture_x: float | None = None,
        capture_y: float | None = None,
    ) -> None:
        self.db.start_soil_job(
            job_id,
            config_entry_id,
            "calibration",
            [point_id] if point_id is not None else [],
        )
        try:
            async with self.shared_lock:
                self.invalidate_safe_sites(config_entry_id, baseline_mm)
                margin = (
                    self.soil_settings_store.load().clear_soil_margin_mm
                    if self.soil_settings_store is not None
                    else 75
                )
                inventory, sites = await self.safe_sites(
                    config_entry_id, baseline_mm, clear_soil_margin_mm=margin
                )
                if capture_x is None or capture_y is None:
                    site = next((item for item in sites if item.point_id == point_id), None)
                    if site is None:
                        raise SoilHeightError(
                            "selected calibration point no longer has a plant- and weed-free site"
                        )
                    target_x, target_y = site.capture_x, site.capture_y
                else:
                    target_x, target_y = capture_x, capture_y
                self.current.update(
                    status="running",
                    message=(f"Capturing calibration images at ({target_x:.0f}, {target_y:.0f})"),
                )
                capture_id, frames, signature = await self._capture_frames(
                    config_entry_id=config_entry_id,
                    point_id=point_id,
                    capture_x=target_x,
                    capture_y=target_y,
                    capture_z=capture_z,
                    baseline_mm=baseline_mm,
                    z_offsets_mm=[0, 25, 50],
                )
                calibration = await asyncio.to_thread(
                    fit_calibration,
                    config_entry_id=config_entry_id,
                    point_id=point_id if point_id is not None else 0,
                    capture_z=capture_z,
                    baseline_mm=baseline_mm,
                    reference_distance_mm=reference_distance_mm,
                    z_direction=inventory.motion.z_direction,
                    frames=frames,
                    declared_camera_signature=signature,
                )
                calibration = self.db.save_soil_calibration(calibration)
                message = (
                    f"Calibration {calibration.calibration_id} saved; "
                    f"maximum residual {calibration.residual_mm:.1f} mm"
                )
                self.current.update(status="complete", message=message, capture_id=str(capture_id))
                self.db.update_soil_job(
                    job_id,
                    status="complete",
                    completed_count=1,
                    message=message,
                    complete=True,
                )
        except asyncio.CancelledError:
            self.db.update_soil_job(
                job_id, status="interrupted", message="app stopped", complete=True
            )
            raise
        except SoilCalibrationQualityError as err:
            detail = str(err)
            message = detail if len(detail) <= 240 else f"{detail[:237]}..."
            # Keep the frames so the operator can override without another
            # (safety-critical) 50 mm capture movement toward the soil.
            self._pending_calibration_override = {
                "job_id": job_id,
                "config_entry_id": config_entry_id,
                "point_id": point_id,
                "capture_z": capture_z,
                "baseline_mm": baseline_mm,
                "reference_distance_mm": reference_distance_mm,
                "z_direction": inventory.motion.z_direction,
                "frames": frames,
                "signature": signature,
            }
            LOGGER.warning("Soil calibration %s failed quality gates: %s", job_id, detail)
            self.current.update(status="failed", message=message, detail=detail)
            self.db.update_soil_job(
                job_id,
                status="failed",
                failed_count=1,
                message=message,
                detail=detail,
                complete=True,
            )
        except Exception as err:  # pylint: disable=broad-except
            message = str(err)[:240] or "Soil calibration failed"
            LOGGER.warning("Soil calibration %s failed: %s", job_id, err)
            self.current.update(status="failed", message=message)
            self.db.update_soil_job(
                job_id, status="failed", failed_count=1, message=message, complete=True
            )

    async def _run_calibration_override(self, *, job_id: str, pending: dict) -> None:
        self.db.start_soil_job(
            job_id,
            pending["config_entry_id"],
            "calibration",
            [pending["point_id"]] if pending["point_id"] is not None else [],
        )
        try:
            async with self.shared_lock:
                self.current.update(
                    status="running",
                    message="Recomputing calibration with quality gates overridden",
                )
                calibration = await asyncio.to_thread(
                    fit_calibration,
                    config_entry_id=pending["config_entry_id"],
                    point_id=pending["point_id"] if pending["point_id"] is not None else 0,
                    capture_z=pending["capture_z"],
                    baseline_mm=pending["baseline_mm"],
                    reference_distance_mm=pending["reference_distance_mm"],
                    z_direction=pending["z_direction"],
                    frames=pending["frames"],
                    declared_camera_signature=pending["signature"],
                    force=True,
                )
                calibration = self.db.save_soil_calibration(calibration)
                detail = "; ".join(calibration.quality_warnings)
                message = (
                    f"Calibration {calibration.calibration_id} saved with quality gates "
                    f"overridden; maximum residual {calibration.residual_mm:.1f} mm"
                )
                LOGGER.warning("Soil calibration %s accepted by override: %s", job_id, detail)
                self.current.update(status="complete", message=message, detail=detail)
                self.db.update_soil_job(
                    job_id,
                    status="complete",
                    completed_count=1,
                    message=message,
                    detail=detail,
                    complete=True,
                )
                self._pending_calibration_override = None
        except asyncio.CancelledError:
            self.db.update_soil_job(
                job_id, status="interrupted", message="app stopped", complete=True
            )
            raise
        except Exception as err:  # pylint: disable=broad-except
            message = str(err)[:240] or "Calibration override failed"
            LOGGER.warning("Soil calibration override %s failed: %s", job_id, err)
            self.current.update(status="failed", message=message)
            self.db.update_soil_job(
                job_id, status="failed", failed_count=1, message=message, complete=True
            )

    async def _run_measurements(
        self,
        *,
        job_id: str,
        config_entry_id: str,
        point_ids: list[int],
        custom_point_id: int | None,
        custom_x: float | None,
        custom_y: float | None,
        capture_z: float,
        baseline_mm: float,
    ) -> None:
        self.db.start_soil_job(job_id, config_entry_id, "measurement", point_ids)
        completed = failed = 0
        try:
            async with self.shared_lock:
                self.invalidate_safe_sites(config_entry_id, baseline_mm)
                margin = (
                    self.soil_settings_store.load().clear_soil_margin_mm
                    if self.soil_settings_store is not None
                    else 75
                )
                inventory, sites = await self.safe_sites(
                    config_entry_id, baseline_mm, clear_soil_margin_mm=margin
                )
                if custom_point_id is not None:
                    point = next(
                        (item for item in inventory.points if item.id == custom_point_id), None
                    )
                    if point is None or custom_x is None or custom_y is None:
                        raise SoilHeightError(
                            "custom measurement soil point or coordinates are invalid"
                        )
                    relocation_distance = math.hypot(custom_x - point.x, custom_y - point.y)
                    if relocation_distance >= 200:
                        raise SoilHeightError(
                            "custom measurement coordinates must be less than 200 mm from the soil point"
                        )
                    capture_targets = [
                        {
                            "point_id": point.id,
                            "point_name": point.name,
                            "expected_x": point.x,
                            "expected_y": point.y,
                            "expected_z": point.z,
                            "point_updated_at": point.updated_at,
                            "capture_x": custom_x,
                            "capture_y": custom_y,
                            "relocation_distance_mm": relocation_distance,
                        }
                    ]
                else:
                    wanted = [site for site in sites if site.point_id in set(point_ids)]
                    if len(wanted) != len(set(point_ids)):
                        raise SoilHeightError(
                            "one or more selected points are no longer stale or clear"
                        )
                    capture_targets = [
                        {
                            "point_id": site.point_id,
                            "point_name": site.point_name,
                            "expected_x": site.expected_x,
                            "expected_y": site.expected_y,
                            "expected_z": site.expected_z,
                            "point_updated_at": site.point_updated_at,
                            "capture_x": site.capture_x,
                            "capture_y": site.capture_y,
                            "relocation_distance_mm": site.relocation_distance_mm,
                        }
                        for site in wanted
                    ]
                position = inventory.motion.position
                ordered = sorted(
                    capture_targets,
                    key=lambda item: math.hypot(
                        item["capture_x"] - float(position.get("x") or 0),
                        item["capture_y"] - float(position.get("y") or 0),
                    ),
                )
                calibration = self.db.active_soil_calibration(config_entry_id)
                if calibration is None:
                    raise SoilHeightError("complete guided soil calibration first")
                if abs(calibration.baseline_mm - baseline_mm) > 0.01:
                    raise SoilHeightError("capture baseline changed; recalibrate before measuring")
                if calibration.z_direction != inventory.motion.z_direction:
                    raise SoilHeightError(
                        "FarmBot Z direction changed; recalibrate before measuring"
                    )
                measurement_batch_id = uuid4()
                for target in ordered:
                    if self.stop_requested:
                        break
                    self.current.update(
                        status="running",
                        current_point_id=target["point_id"],
                        message=(
                            f"Measuring soil at ({target['capture_x']:.0f}, "
                            f"{target['capture_y']:.0f})"
                        ),
                    )
                    self.db.update_soil_job(
                        job_id,
                        current_point_id=target["point_id"],
                        completed_count=completed,
                        failed_count=failed,
                        message=self.current["message"],
                    )
                    try:
                        capture_id, frames, signature = await self._capture_frames(
                            config_entry_id=config_entry_id,
                            point_id=target["point_id"],
                            capture_x=target["capture_x"],
                            capture_y=target["capture_y"],
                            capture_z=capture_z,
                            baseline_mm=baseline_mm,
                            z_offsets_mm=[0],
                            batch_id=measurement_batch_id,
                            expected_geometry=(
                                calibration.processed_width,
                                calibration.processed_height,
                                calibration.source_width,
                                calibration.source_height,
                            ),
                        )
                        analysis = await asyncio.to_thread(
                            analyse_soil_height,
                            frames,
                            calibration,
                            declared_camera_signature=signature,
                        )
                        artifact_paths = self._save_artifacts(
                            analysis.measurement_id, analysis.artifacts
                        )
                        measurement = SoilMeasurement(
                            measurement_id=analysis.measurement_id,
                            config_entry_id=config_entry_id,
                            point_id=target["point_id"],
                            point_name=target["point_name"],
                            expected_x=target["expected_x"],
                            expected_y=target["expected_y"],
                            old_z_mm=target["expected_z"],
                            point_updated_at=target["point_updated_at"],
                            capture_x=target["capture_x"],
                            capture_y=target["capture_y"],
                            relocation_distance_mm=target["relocation_distance_mm"],
                            proposed_z_mm=analysis.proposed_z_mm,
                            confidence=analysis.confidence,
                            uncertainty_mm=analysis.uncertainty_mm,
                            status="valid" if analysis.valid else "failed",
                            reason=analysis.reason,
                            capture_id=capture_id,
                            calibration_id=calibration.calibration_id,
                            frame_ids=[frame.image_id for frame in frames],
                            metrics=analysis.metrics,
                            artifact_paths=artifact_paths,
                            algorithm_version=SOIL_ALGORITHM_VERSION,
                        )
                        self.db.save_soil_measurement(measurement)
                        if analysis.valid:
                            completed += 1
                        else:
                            failed += 1
                            LOGGER.warning(
                                "Soil measurement for point %s failed: %s",
                                target["point_id"],
                                analysis.reason,
                            )
                    except Exception as err:  # pylint: disable=broad-except
                        failed += 1
                        LOGGER.warning(
                            "Soil measurement for point %s failed: %s",
                            target["point_id"],
                            err,
                        )
                        measurement = SoilMeasurement(
                            measurement_id=uuid4(),
                            config_entry_id=config_entry_id,
                            point_id=target["point_id"],
                            point_name=target["point_name"],
                            expected_x=target["expected_x"],
                            expected_y=target["expected_y"],
                            old_z_mm=target["expected_z"],
                            point_updated_at=target["point_updated_at"],
                            capture_x=target["capture_x"],
                            capture_y=target["capture_y"],
                            relocation_distance_mm=target["relocation_distance_mm"],
                            status="failed",
                            reason=str(err)[:240] or "Soil measurement failed",
                            calibration_id=calibration.calibration_id,
                            algorithm_version=SOIL_ALGORITHM_VERSION,
                        )
                        self.db.save_soil_measurement(measurement)
                    self.current.update(completed_count=completed, failed_count=failed)
                self.current["message"] = "Restoring the FarmBot starting position"
                await self._finish_capture_batch(config_entry_id, measurement_batch_id)
                status = "stopped" if self.stop_requested else "complete"
                message = (
                    f"Stopped after current point; {completed} valid, {failed} failed"
                    if self.stop_requested
                    else f"Soil measurement complete: {completed} valid, {failed} failed"
                )
                self.current.update(status=status, message=message)
                self.db.update_soil_job(
                    job_id,
                    status=status,
                    completed_count=completed,
                    failed_count=failed,
                    message=message,
                    complete=True,
                )
        except asyncio.CancelledError:
            self.db.update_soil_job(
                job_id, status="interrupted", message="app stopped", complete=True
            )
            raise
        except Exception as err:  # pylint: disable=broad-except
            message = str(err)[:240] or "Soil job failed"
            LOGGER.warning("Soil job %s failed: %s", job_id, err)
            self.current.update(status="failed", message=message)
            self.db.update_soil_job(
                job_id,
                status="failed",
                completed_count=completed,
                failed_count=failed,
                message=message,
                complete=True,
            )

    def _save_artifacts(self, measurement_id: UUID, artifacts: dict[str, bytes]) -> list[str]:
        directory = self.data_dir / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for suffix, data in artifacts.items():
            if not data:
                continue
            path = directory / f"soil-{measurement_id}-{Path(suffix).name}"
            path.write_bytes(data)
            paths.append(str(path))
        return paths
