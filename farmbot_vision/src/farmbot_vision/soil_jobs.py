"""Soil-height capture, calibration, review, and persistence orchestration."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
from pathlib import Path
from uuid import UUID, uuid4

from .database import Database
from .home_assistant import HomeAssistantClient
from .models import (
    SoilCaptureStartRequest,
    SoilMeasurement,
    SoilPoint,
    VisionImageRequest,
)
from .soil_height import (
    SoilFrame,
    SoilHeightError,
    analyse_soil_height,
    fit_calibration,
)

LOGGER = logging.getLogger(__name__)


class SoilJobManager:
    """One resumability-safe, fail-closed soil workflow sharing the vision lock."""

    def __init__(
        self,
        database: Database,
        client: HomeAssistantClient,
        data_dir: Path,
        shared_lock: asyncio.Lock,
    ):
        self.db = database
        self.client = client
        self.data_dir = data_dir
        self.shared_lock = shared_lock
        self.task: asyncio.Task | None = None
        self.stop_requested = False
        self.current: dict = {"status": "idle", "message": "Not run"}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

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
        point_id: int,
        capture_z: float,
        baseline_mm: float,
        reference_distance_mm: float,
        z_direction: int,
    ) -> str:
        return self._start(
            lambda job_id: self._run_calibration(
                job_id=job_id,
                config_entry_id=config_entry_id,
                point_id=point_id,
                capture_z=capture_z,
                baseline_mm=baseline_mm,
                reference_distance_mm=reference_distance_mm,
                z_direction=z_direction,
            ),
            name="soil-calibration",
        )

    def start_measurements(
        self,
        *,
        config_entry_id: str,
        point_ids: list[int],
        capture_z: float,
        baseline_mm: float,
    ) -> str:
        if not point_ids:
            raise ValueError("select at least one soil point")
        return self._start(
            lambda job_id: self._run_measurements(
                job_id=job_id,
                config_entry_id=config_entry_id,
                point_ids=list(dict.fromkeys(point_ids)),
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

    async def _capture_frames(
        self,
        *,
        config_entry_id: str,
        point_id: int,
        capture_z: float,
        baseline_mm: float,
        z_offsets_mm: list[float],
    ) -> tuple[UUID, list[SoilFrame], str]:
        started = await self.client.start_soil_capture(
            SoilCaptureStartRequest(
                config_entry_id=config_entry_id,
                point_id=point_id,
                capture_z=capture_z,
                baseline_mm=baseline_mm,
                z_offsets_mm=z_offsets_mm,
            )
        )
        if started.status != "queued" or started.capture_id is None:
            raise SoilHeightError(started.message)
        # The integration may legitimately use the full 120 s RPC, 180 s image
        # processing and 60 s best-effort restoration windows.
        for _ in range(190):
            status = await self.client.soil_capture_status(config_entry_id, str(started.capture_id))
            self.current["message"] = status.message
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
        for item in status.frames:
            response = await self.client.image(
                VisionImageRequest(
                    config_entry_id=config_entry_id,
                    image_id=item.image_id,
                    max_width=1280,
                    max_height=960,
                ),
                5 * 1024 * 1024,
            )
            if not response.full_metadata:
                raise SoilHeightError("soil image lacks contract-v2 geometry")
            if (response.width, response.height) != (1280, 960):
                raise SoilHeightError(
                    "soil image does not meet the required 1280×960 processed contract"
                )
            jpeg = base64.b64decode(response.image_base64)
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

    async def _run_calibration(
        self,
        *,
        job_id: str,
        config_entry_id: str,
        point_id: int,
        capture_z: float,
        baseline_mm: float,
        reference_distance_mm: float,
        z_direction: int,
    ) -> None:
        self.db.start_soil_job(job_id, config_entry_id, "calibration", [point_id])
        try:
            async with self.shared_lock:
                self.current.update(status="running", message="Capturing calibration images")
                capture_id, frames, signature = await self._capture_frames(
                    config_entry_id=config_entry_id,
                    point_id=point_id,
                    capture_z=capture_z,
                    baseline_mm=baseline_mm,
                    z_offsets_mm=[0, 25, 50],
                )
                calibration = await asyncio.to_thread(
                    fit_calibration,
                    config_entry_id=config_entry_id,
                    point_id=point_id,
                    capture_z=capture_z,
                    baseline_mm=baseline_mm,
                    reference_distance_mm=reference_distance_mm,
                    z_direction=z_direction,
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
        except Exception as err:  # pylint: disable=broad-except
            message = str(err)[:240] or "Soil calibration failed"
            LOGGER.warning("Soil calibration %s failed: %s", job_id, err)
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
        capture_z: float,
        baseline_mm: float,
    ) -> None:
        self.db.start_soil_job(job_id, config_entry_id, "measurement", point_ids)
        completed = failed = 0
        try:
            async with self.shared_lock:
                inventory = await self.client.soil_points(config_entry_id)
                wanted = [point for point in inventory.points if point.id in set(point_ids)]
                if len(wanted) != len(set(point_ids)):
                    raise SoilHeightError("one or more selected soil points no longer exist")
                position = inventory.motion.position
                ordered = self._nearest_order(
                    wanted, float(position.get("x") or 0), float(position.get("y") or 0)
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
                for point in ordered:
                    if self.stop_requested:
                        break
                    self.current.update(
                        status="running",
                        current_point_id=point.id,
                        message=f"Measuring {point.name} at ({point.x:.0f}, {point.y:.0f})",
                    )
                    self.db.update_soil_job(
                        job_id,
                        current_point_id=point.id,
                        completed_count=completed,
                        failed_count=failed,
                        message=self.current["message"],
                    )
                    try:
                        capture_id, frames, signature = await self._capture_frames(
                            config_entry_id=config_entry_id,
                            point_id=point.id,
                            capture_z=capture_z,
                            baseline_mm=baseline_mm,
                            z_offsets_mm=[0],
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
                            point_id=point.id,
                            point_name=point.name,
                            expected_x=point.x,
                            expected_y=point.y,
                            old_z_mm=point.z,
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
                        )
                        self.db.save_soil_measurement(measurement)
                        if analysis.valid:
                            completed += 1
                        else:
                            failed += 1
                    except Exception as err:  # pylint: disable=broad-except
                        failed += 1
                        measurement = SoilMeasurement(
                            measurement_id=uuid4(),
                            config_entry_id=config_entry_id,
                            point_id=point.id,
                            point_name=point.name,
                            expected_x=point.x,
                            expected_y=point.y,
                            old_z_mm=point.z,
                            status="failed",
                            reason=str(err)[:240] or "Soil measurement failed",
                            calibration_id=calibration.calibration_id,
                        )
                        self.db.save_soil_measurement(measurement)
                    self.current.update(completed_count=completed, failed_count=failed)
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
