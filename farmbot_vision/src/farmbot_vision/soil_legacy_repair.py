"""One-shot repair of soil-stereo-v2 measurements.

This is intentionally not a general reprocessing API. Its database selector is
hard-coded to the retired v2 algorithm and every eligible row receives one
durable classification, after which the workflow has nothing left to run on.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
from uuid import UUID

from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError
from .models import ApplySoilHeightRequest, SoilStereoCalibration, VisionImageRequest
from .soil_height import SoilFrame, SoilHeightError, analyse_soil_height, fit_calibration

LOGGER = logging.getLogger(__name__)
LEGACY_SOIL_ALGORITHM = "soil-stereo-v2"
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class LegacySoilRepairManager:
    """Stage and apply corrections for the finite set of persisted v2 heights."""

    def __init__(self, database: Database, client: HomeAssistantClient) -> None:
        self.db = database
        self.client = client
        self._task: asyncio.Task | None = None
        self.current: dict[str, object] = {
            "status": "idle",
            "message": "Legacy soil-height repair has not run",
        }

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, config_entry_id: str) -> None:
        if self.running:
            raise ValueError("Legacy soil-height repair is already running")
        summary = self.db.legacy_soil_repair_summary(config_entry_id)
        if not summary["unprocessed"]:
            raise ValueError("No unprocessed soil-stereo-v2 measurements remain")
        self.current = {
            "status": "running",
            "message": "Checking retained legacy soil images without changing FarmBot points",
        }
        self._task = asyncio.create_task(
            self._run(config_entry_id), name="legacy-soil-height-repair"
        )

    async def _run(self, config_entry_id: str) -> None:
        try:
            await self.scan_now(config_entry_id)
            pending = len(self.db.pending_legacy_soil_repairs(config_entry_id))
            self.current = {
                "status": "complete",
                "message": (
                    f"Legacy repair check complete; {pending} change"
                    f"{'s' if pending != 1 else ''} require review"
                ),
            }
        except Exception as err:  # noqa: BLE001 - retained for visible job diagnostics
            LOGGER.exception("Legacy soil-height repair failed")
            self.current = {"status": "failed", "message": str(err)[:240]}

    async def _fetch_frame(
        self,
        *,
        config_entry_id: str,
        image_id: int,
        calibration: SoilStereoCalibration,
        lateral_offset_mm: float,
        z_offset_mm: float,
    ) -> SoilFrame:
        image = await self.client.image(
            VisionImageRequest(
                config_entry_id=config_entry_id,
                image_id=image_id,
                max_width=calibration.processed_width,
                max_height=calibration.processed_height,
            ),
            MAX_IMAGE_BYTES,
        )
        if not image.full_metadata:
            raise SoilHeightError(f"legacy image {image_id} lacks geometry metadata")
        geometry = (
            image.width,
            image.height,
            int(image.source_width),
            int(image.source_height),
        )
        expected = (
            calibration.processed_width,
            calibration.processed_height,
            calibration.source_width,
            calibration.source_height,
        )
        if geometry != expected:
            raise SoilHeightError(
                f"legacy image {image_id} geometry changed from {expected} to {geometry}"
            )
        return SoilFrame(
            image_id=image.image_id,
            jpeg=base64.b64decode(image.image_base64),
            x=image.meta.x,
            y=image.meta.y,
            z=image.meta.z,
            lateral_offset_mm=lateral_offset_mm,
            z_offset_mm=z_offset_mm,
            processed_width=image.width,
            processed_height=image.height,
            source_width=int(image.source_width),
            source_height=int(image.source_height),
        )

    async def _refit_calibration(self, calibration: SoilStereoCalibration) -> SoilStereoCalibration:
        if len(calibration.source_image_ids) != 9:
            raise SoilHeightError("legacy calibration does not reference nine source images")
        frames = []
        for index, image_id in enumerate(calibration.source_image_ids):
            level = index // 3
            lateral = index % 3
            frames.append(
                await self._fetch_frame(
                    config_entry_id=calibration.config_entry_id,
                    image_id=image_id,
                    calibration=calibration,
                    lateral_offset_mm=lateral * calibration.baseline_mm,
                    z_offset_mm=level * 25.0,
                )
            )
        return await asyncio.to_thread(
            fit_calibration,
            config_entry_id=calibration.config_entry_id,
            point_id=calibration.point_id,
            capture_z=calibration.capture_z,
            baseline_mm=calibration.baseline_mm,
            reference_distance_mm=calibration.reference_distance_mm,
            z_direction=calibration.z_direction,
            frames=frames,
            force=calibration.quality_override,
        )

    async def _measurement_frames(
        self, row: dict, calibration: SoilStereoCalibration
    ) -> list[SoilFrame]:
        image_ids = [int(value) for value in row.get("frame_ids") or []]
        if len(image_ids) != 3:
            raise SoilHeightError("legacy measurement does not reference three source images")
        return [
            await self._fetch_frame(
                config_entry_id=row["config_entry_id"],
                image_id=image_id,
                calibration=calibration,
                lateral_offset_mm=index * calibration.baseline_mm,
                z_offset_mm=0,
            )
            for index, image_id in enumerate(image_ids)
        ]

    async def scan_now(self, config_entry_id: str) -> None:
        """Classify every unprocessed v2 height without writing a FarmBot point."""

        # Fail before classifying individual rows when the integration itself
        # is unavailable. A missing historical image can then be recorded as
        # unavailable without mistaking an offline FarmBot for image deletion.
        await self.client.soil_points(config_entry_id)
        rows = self.db.unprocessed_legacy_soil_measurements(config_entry_id)
        calibrations: dict[int, SoilStereoCalibration] = {}
        calibration_errors: dict[int, str] = {}
        for row in rows:
            old_z = float(row["proposed_z_mm"])
            measurement_id = str(row["measurement_id"])
            calibration_id = row.get("calibration_id")
            try:
                if calibration_id is None:
                    raise SoilHeightError("legacy measurement has no calibration reference")
                calibration_id = int(calibration_id)
                if calibration_id in calibration_errors:
                    raise SoilHeightError(calibration_errors[calibration_id])
                repaired_calibration = calibrations.get(calibration_id)
                if repaired_calibration is None:
                    stored = self.db.soil_calibration(calibration_id)
                    if stored is None:
                        raise SoilHeightError("legacy calibration record is missing")
                    try:
                        repaired_calibration = await self._refit_calibration(stored)
                    except (HomeAssistantError, SoilHeightError) as err:
                        calibration_errors[calibration_id] = str(err)
                        raise
                    calibrations[calibration_id] = repaired_calibration
                frames = await self._measurement_frames(row, repaired_calibration)
                analysis = await asyncio.to_thread(
                    analyse_soil_height, frames, repaired_calibration
                )
                if not analysis.valid or analysis.proposed_z_mm is None:
                    raise SoilHeightError(analysis.reason or "legacy recalculation failed")
                repaired_z = float(analysis.proposed_z_mm)
                delta = repaired_z - old_z
                state = "unchanged" if abs(delta) < 0.5 else "pending"
                reason = (
                    "Recalculated with recorded camera positions; no rounded height change"
                    if state == "unchanged"
                    else "Recalculated from retained source images with soil-stereo-v3"
                )
                self.db.save_legacy_soil_repair(
                    legacy_measurement_id=measurement_id,
                    config_entry_id=config_entry_id,
                    source_status=str(row["status"]),
                    state=state,
                    old_proposed_z_mm=old_z,
                    repaired_z_mm=repaired_z,
                    confidence=analysis.confidence,
                    uncertainty_mm=analysis.uncertainty_mm,
                    reason=reason,
                )
            except (HomeAssistantError, SoilHeightError, ValueError) as err:
                self.db.save_legacy_soil_repair(
                    legacy_measurement_id=measurement_id,
                    config_entry_id=config_entry_id,
                    source_status=str(row["status"]),
                    state="unavailable",
                    old_proposed_z_mm=old_z,
                    repaired_z_mm=None,
                    confidence=None,
                    uncertainty_mm=None,
                    reason=str(err) or type(err).__name__,
                )

    @staticmethod
    def _matches_legacy_state(point, measurement: dict) -> bool:
        if measurement["status"] == "valid":
            expected_x = float(measurement["expected_x"])
            expected_y = float(measurement["expected_y"])
            expected_z = float(measurement["old_z_mm"])
        else:
            expected_x = float(measurement.get("capture_x") or measurement["expected_x"])
            expected_y = float(measurement.get("capture_y") or measurement["expected_y"])
            expected_z = float(measurement["proposed_z_mm"])
        return (
            math.isclose(point.x, expected_x, abs_tol=0.1)
            and math.isclose(point.y, expected_y, abs_tol=0.1)
            and math.isclose(point.z, expected_z, abs_tol=0.1)
        )

    async def apply_selected(self, config_entry_id: str, measurement_ids: list[str]) -> None:
        """Apply explicitly selected staged repairs after rechecking current point state."""

        inventory = await self.client.soil_points(config_entry_id)
        points = {point.id: point for point in inventory.points}
        for measurement_id in measurement_ids:
            repair = self.db.legacy_soil_repair(measurement_id)
            measurement = self.db.soil_measurement(measurement_id)
            if (
                repair is None
                or repair["config_entry_id"] != config_entry_id
                or repair["state"] != "pending"
                or repair["repaired_z_mm"] is None
                or measurement is None
                or measurement["algorithm_version"] != LEGACY_SOIL_ALGORITHM
            ):
                continue
            point = points.get(int(measurement["point_id"]))
            if point is None or point.updated_at is None:
                self.db.resolve_legacy_soil_repair(
                    measurement_id, "conflict", "FarmBot soil point is missing or undated"
                )
                continue
            if not self._matches_legacy_state(point, measurement):
                self.db.resolve_legacy_soil_repair(
                    measurement_id,
                    "conflict",
                    "FarmBot soil point changed after the legacy measurement",
                )
                continue
            recommended_x = (
                float(measurement.get("capture_x") or point.x)
                if measurement["status"] == "valid"
                else point.x
            )
            recommended_y = (
                float(measurement.get("capture_y") or point.y)
                if measurement["status"] == "valid"
                else point.y
            )
            request = ApplySoilHeightRequest(
                config_entry_id=config_entry_id,
                point_id=point.id,
                measurement_id=UUID(measurement_id),
                expected_x=point.x,
                expected_y=point.y,
                expected_z=point.z,
                expected_updated_at=point.updated_at,
                recommended_x=recommended_x,
                recommended_y=recommended_y,
                recommended_z_mm=float(repair["repaired_z_mm"]),
                confidence=float(repair["confidence"] or 0),
                apply=True,
                human_approved=True,
            )
            try:
                response = await self.client.apply_soil_height(request)
            except HomeAssistantError as err:
                response = {"status": "conflict", "message": str(err)}
            status = str(response.get("status") or "rejected")
            state = "applied" if status == "applied" else "conflict"
            reason = str(response.get("message") or status)
            self.db.resolve_legacy_soil_repair(measurement_id, state, reason)
            self.db.record_soil_decision(
                measurement_id,
                "legacy_v2_repair_apply" if state == "applied" else "legacy_v2_repair_conflict",
                response,
            )

    def reject_selected(self, config_entry_id: str, measurement_ids: list[str]) -> None:
        for measurement_id in measurement_ids:
            repair = self.db.legacy_soil_repair(measurement_id)
            if (
                repair is not None
                and repair["config_entry_id"] == config_entry_id
                and repair["state"] == "pending"
            ):
                self.db.resolve_legacy_soil_repair(
                    measurement_id, "rejected", "User kept the existing legacy height"
                )
                self.db.record_soil_decision(
                    measurement_id,
                    "legacy_v2_repair_reject",
                    {"status": "rejected", "reason": "kept existing height"},
                )
