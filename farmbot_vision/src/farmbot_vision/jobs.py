from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
from datetime import UTC, datetime, timedelta
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
from .canopy_fusion import fuse_canopy_masks
from .canopy_settings import CanopyFusionSettings, CanopyFusionSettingsStore
from .curve_edit import propose_curve_point, radius_mm_to_diameter_mm
from .curves import fit_monotonic_curve
from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError, StaleRadiusError
from .models import (
    ApplyRadiusRequest,
    ApplyRemovalRequest,
    CreateWeedRequest,
    Decision,
    InventoryRequest,
    KnownWeedSeed,
    OperatingMode,
    PlantSeed,
    RemoveWeedRequest,
    UpdateWeedRadiusRequest,
    UpsertCurveRequest,
    VisionImageRequest,
    VisionStatus,
)
from .photo_grid import PhotoGridStore
from .plant_measurement import select_measurement_evidence, selection_diagnostics
from .review_composite import build_plant_review
from .safety import decide
from .settings import Settings
from .vision import ClassicalVisionEngine, garden_to_pixel, pixel_to_garden
from .weed_settings import WeedSettingsStore
from .weed_verifier import WeedVisualVerifier
from .zones import Zone, ZoneAspect, ZoneStore, evaluate

LOGGER = logging.getLogger(__name__)
ANALYSIS_COOPERATIVE_PAUSE_SECONDS = 0.1


def _measurement_image_artifacts(measurements: list) -> list[str]:
    """Return the saved images that explain an automatic decision."""

    paths: list[str] = []
    for measurement in measurements:
        for name in (
            "source_image_path",
            "overlay_path",
            "composite_path",
            "composite_overlay_path",
            "fusion_diagnostic_path",
        ):
            path = getattr(measurement, name, None)
            if path and path not in paths:
                paths.append(str(path))
        for path in getattr(measurement, "artifact_paths", []) or []:
            if path and path not in paths:
                paths.append(str(path))
    return paths


def _automatic_change_details(result: dict, measurements: list) -> dict:
    """Keep the API result plus enough provenance for the dashboard View button."""

    details = dict(result)
    artifacts = _measurement_image_artifacts(measurements)
    if artifacts:
        details["image_artifacts"] = artifacts
    if measurements:
        details["measurement_id"] = str(measurements[0].measurement_id)
    return details


def curve_radius_after_measurement(measurement, curve_data: dict[str, float]) -> float:
    """Keep a plant's curve from shrinking when a radius observation drops."""

    proposed = float(measurement.recommended_protection_radius_mm)
    if proposed >= float(measurement.current_radius_mm):
        return proposed
    age = measurement.plant_age_days
    prior_diameters = (
        [float(value) for day, value in curve_data.items() if int(day) < int(age)]
        if age is not None
        else []
    )
    previous_maximum = max(prior_diameters, default=0.0) / 2
    return max(proposed, float(measurement.current_radius_mm), previous_maximum)


def limit_weed_radius_growth(
    current_radius_mm: float,
    measured_radius_mm: float,
    rolling_baseline_radius_mm: float,
    maximum_growth_mm: float,
    maximum_growth_percent: float,
) -> tuple[float, float]:
    """Return a safe automatic radius and the rolling-window ceiling."""

    baseline = max(0.0, float(rolling_baseline_radius_mm))
    ceiling = min(
        baseline + max(0.0, float(maximum_growth_mm)),
        baseline * (1.0 + max(0.0, float(maximum_growth_percent)) / 100.0),
    )
    target = max(
        float(current_radius_mm),
        min(float(measured_radius_mm), ceiling),
    )
    return target, ceiling


def build_plant_composite(
    measurements: list,
    output_path: Path,
    overlay_output_path: Path | None = None,
    *,
    plants: list | None = None,
    proposed_radii: dict[int, float] | None = None,
    grid_record=None,
) -> bool:
    """Stitch all views of one plant using their calibrated garden transforms.

    ``output_path`` is the clean photo composite. When ``overlay_output_path``
    is supplied, a second copy is written with the plant ownership masks
    tinted over the same stitched pixels. Radius and centre annotations are
    drawn on both copies.
    """
    return build_plant_review(
        measurements,
        output_path,
        overlay_output_path,
        plants=plants,
        proposed_radii=proposed_radii,
        grid_record=grid_record,
    )


class JobManager:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        client: HomeAssistantClient,
        weed_settings_store: WeedSettingsStore | None = None,
        zone_store: ZoneStore | None = None,
    ):
        self.settings = settings
        self.db = database
        self.client = client
        self.lock = asyncio.Lock()
        self.current: dict = {"status": "idle", "queue_length": 0, "progress": "Not run"}
        self.last: dict = {}
        self.queued_image_ids: list[int] = []
        self.weed_settings_store = weed_settings_store
        self.zone_store = zone_store

    def add_to_queue(self, image_ids: list[int]) -> int:
        for image_id in image_ids:
            if image_id > 0 and image_id not in self.queued_image_ids:
                self.queued_image_ids.append(image_id)
        self.current["queue_length"] = len(self.queued_image_ids)
        return len(self.queued_image_ids)

    def resources_available(self) -> tuple[bool, str]:
        memory_mb = psutil.virtual_memory().available / 1024 / 1024
        # interval=None is non-blocking. The old 100 ms sampling interval ran
        # synchronously on the web event loop before every image.
        cpu = psutil.cpu_percent(interval=None)
        if memory_mb < self.settings.minimum_free_memory_mb:
            return False, f"free memory below {self.settings.minimum_free_memory_mb} MB"
        if cpu > self.settings.maximum_system_load_percent:
            return False, f"system CPU load above {self.settings.maximum_system_load_percent}%"
        return True, "resources available"

    async def _yield_for_server(self) -> tuple[bool, str]:
        """Leave scheduling space for Home Assistant and interactive requests."""
        await asyncio.sleep(ANALYSIS_COOPERATIVE_PAUSE_SECONDS)
        while True:
            available, reason = self.resources_available()
            if available:
                return True, reason
            if reason.startswith("free memory"):
                return False, reason
            self.current["progress"] = f"Yielding to Home Assistant: {reason}"
            await asyncio.sleep(0.5)

    async def _update_curve_after_radius(
        self, entry_id: str, inventory, measurement, *, human_approved: bool
    ) -> dict:
        plant = next((item for item in inventory.plants if item.id == measurement.plant_id), None)
        if plant is None or measurement.plant_age_days is None:
            return {"status": "skipped", "message": "plant or plant age is unavailable"}
        curve = next((item for item in inventory.curves if item.id == plant.spread_curve_id), None)
        # Plants without a curve still need one after an approved measurement.
        # Seed a Vision-owned curve with the current FarmBot diameter and add
        # the measured-age point below.
        base_curve_data = (
            curve.data
            if curve is not None
            else {"0": radius_mm_to_diameter_mm(measurement.current_radius_mm)}
        )
        curve_radius = curve_radius_after_measurement(measurement, base_curve_data)
        edit = propose_curve_point(
            base_curve_data,
            measurement.plant_age_days,
            radius_mm_to_diameter_mm(curve_radius),
            max_daily_growth_mm=self.settings.maximum_daily_radius_growth_mm,
            maximum_plant_radius_mm=self.settings.maximum_plant_radius_mm,
        )
        users = (
            [item for item in inventory.plants if item.spread_curve_id == curve.id]
            if curve is not None
            else []
        )
        may_patch = (
            curve is not None and curve.name.startswith("[FarmBot Vision]") and len(users) == 1
        )
        target_curve_id = curve.id if may_patch and curve is not None else None
        target_name = (
            curve.name
            if may_patch and curve is not None
            else f"[FarmBot Vision] {plant.name} spread"
        )
        if edit.verdict == "flagged" and not human_approved:
            proposal_id = self.db.create_curve_proposal(
                config_entry_id=entry_id,
                plant_id=plant.id,
                measurement_id=str(measurement.measurement_id),
                crop_slug=measurement.crop_slug,
                plant_age_days=measurement.plant_age_days,
                curve_id=target_curve_id,
                curve_name=target_name,
                previous_data=base_curve_data,
                data=edit.data,
                reason=edit.reason or "curve edit needs review",
                conflict_day=edit.conflict_day,
                conflict_old_diameter=edit.conflict_old_diameter,
                overlay_path=measurement.overlay_path,
                warning="; ".join(edit.warnings) or None,
            )
            result = {
                "status": "flagged",
                "message": edit.reason or "curve edit needs review",
                "proposal_id": proposal_id,
            }
            self.db.record_decision(str(measurement.measurement_id), "curve_flagged", result)
            return result
        # The companion contract uses integer millimetres. Round deliberately
        # instead of relying on its schema coercion to truncate float values.
        curve_data = {day: float(round(value)) for day, value in edit.data.items()}
        try:
            result = await self.client.upsert_curve(
                UpsertCurveRequest(
                    config_entry_id=entry_id,
                    crop_slug=measurement.crop_slug,
                    curve_id=target_curve_id,
                    name=target_name,
                    data=curve_data,
                    assign_to_plant_ids=[plant.id],
                    apply=True,
                    human_approved=human_approved,
                )
            )
        except HomeAssistantError as exc:
            result = {"status": "rejected", "message": str(exc)}
        self.db.record_decision(
            str(measurement.measurement_id),
            "curve_applied" if result.get("status") == "applied" else "curve_rejected",
            result,
        )
        return result

    async def _apply_consolidated_automatic(
        self,
        *,
        entry_id: str,
        inventory,
        measurements: list,
        zones: list[Zone],
        fusion_settings: CanopyFusionSettings,
    ) -> None:
        """Apply at most one robust automatic judgement for repeated views."""
        if len(measurements) < 2:
            return
        evidence = select_measurement_evidence(measurements)
        if not evidence.used:
            return
        aggregate = Database._consolidate_measurement_rows(
            [item.model_dump(mode="json") for item in measurements]
        )
        partial_views_require_fusion = (
            fusion_settings.enabled
            and fusion_settings.automatic_requires_reliable_fusion
            and evidence.mode != "single_complete"
        )
        if partial_views_require_fusion and not bool(aggregate.get("fusion_reliable")):
            return
        if aggregate.get("fused_canopy") and not bool(aggregate.get("fusion_reliable")):
            return
        representative = next(
            (
                item
                for item in evidence.used
                if str(item.measurement_id) == str(aggregate["measurement_id"])
            ),
            max(evidence.used, key=lambda item: item.image_timestamp),
        )
        candidate = representative.model_copy(
            update={
                "typical_canopy_radius_mm": aggregate["typical_canopy_radius_mm"],
                "maximum_accepted_canopy_radius_mm": aggregate["maximum_accepted_canopy_radius_mm"],
                "recommended_protection_radius_mm": aggregate["recommended_protection_radius_mm"],
                "confidence": aggregate["confidence"],
                "visible_fraction": aggregate["visible_fraction"],
                "vegetation_absent": bool(aggregate["vegetation_absent"]),
                "absent_observations": aggregate.get("absent_observations", 0),
                "reason": aggregate["reason"],
            }
        )
        candidate = decide(
            candidate,
            OperatingMode.AUTO_RADIUS,
            self.settings,
            previously_observed_canopy=self.db.has_present_measurement(
                entry_id, candidate.plant_id
            ),
        )
        if candidate.decision == Decision.REMOVED:
            try:
                result = await self.client.apply_removal(
                    ApplyRemovalRequest(
                        config_entry_id=entry_id,
                        plant_id=candidate.plant_id,
                        measurement_id=candidate.measurement_id,
                        expected_current_radius_mm=candidate.current_radius_mm,
                        confidence=candidate.confidence,
                        apply=True,
                    )
                )
            except (HomeAssistantError, StaleRadiusError) as exc:
                LOGGER.warning("Consolidated automatic plant removal failed: %s", exc)
                return
            if result.get("status") == "applied":
                self.db.update_measurement_outcome(
                    str(candidate.measurement_id), decision="removed", applied=True
                )
                self.db.record_group_decision(str(candidate.measurement_id), "removed", result)
                self.db.record_change(
                    entity_type="plant",
                    entity_id=candidate.plant_id,
                    crop_type=candidate.crop_slug,
                    x=candidate.recorded_center_x,
                    y=candidate.recorded_center_y,
                    z=None,
                    change_type="plant removed",
                    original_radius_mm=float(
                        result.get("old_radius_mm") or candidate.current_radius_mm
                    ),
                    current_radius_mm=0,
                    decision_method="auto",
                    confidence=candidate.confidence,
                    details=_automatic_change_details(result, measurements),
                )
            return
        if candidate.decision != Decision.APPLIED or not candidate.calibrated:
            return
        if candidate.recorded_center_x is not None and candidate.recorded_center_y is not None:
            verdict = evaluate(
                zones,
                ZoneAspect.RADIUS,
                candidate.recorded_center_x,
                candidate.recorded_center_y,
                candidate.recommended_protection_radius_mm,
            )
            if not verdict.allowed:
                self.current["zone_blocked_radius"] += 1
                return
        try:
            result = await self.client.apply_radius(
                ApplyRadiusRequest(
                    config_entry_id=entry_id,
                    plant_id=candidate.plant_id,
                    measurement_id=candidate.measurement_id,
                    expected_current_radius_mm=candidate.current_radius_mm,
                    recommended_radius_mm=candidate.recommended_protection_radius_mm,
                    confidence=candidate.confidence,
                    apply=True,
                )
            )
        except (HomeAssistantError, StaleRadiusError) as exc:
            LOGGER.warning("Consolidated automatic radius update failed: %s", exc)
            return
        if result.get("status") == "applied":
            self.db.update_measurement_outcome(
                str(candidate.measurement_id), decision="applied", applied=True
            )
            self.db.record_group_decision(str(candidate.measurement_id), "applied", result)
            new_radius = float(
                result.get(
                    "new_radius_mm",
                    result.get("radius_mm", candidate.recommended_protection_radius_mm),
                )
            )
            self.db.record_change(
                entity_type="plant",
                entity_id=candidate.plant_id,
                crop_type=candidate.crop_slug,
                x=candidate.recorded_center_x,
                y=candidate.recorded_center_y,
                z=None,
                change_type=(
                    "radius increased"
                    if new_radius > candidate.current_radius_mm
                    else "radius decreased"
                ),
                original_radius_mm=candidate.current_radius_mm,
                current_radius_mm=new_radius,
                decision_method="auto",
                confidence=candidate.confidence,
                details=_automatic_change_details(result, measurements),
            )
            await self._update_curve_after_radius(
                entry_id, inventory, candidate, human_approved=False
            )

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
        if trigger == "manual" and not image_ids and self.queued_image_ids:
            image_ids = list(self.queued_image_ids)
            self.queued_image_ids.clear()
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
            grid_record = PhotoGridStore(self.settings.data_dir / "photo_grid_latest.json").load()
            if grid_record is not None and grid_record.config_entry_id != entry_id:
                grid_record = None
            grid_frames_by_image = (
                {
                    int(frame.image_id): frame
                    for frame in [
                        *grid_record.frames,
                        *grid_record.quality_overlay_frames,
                    ]
                }
                if grid_record is not None
                else {}
            )
            grid_targets_by_index = (
                {int(target.index): target for target in grid_record.targets}
                if grid_record is not None
                else {}
            )
            excluded_grid_image_ids = (
                {int(image_id) for image_id in grid_record.excluded_image_ids}
                if grid_record is not None
                else set()
            )
            self.current["progress"] = "Inventory loaded"
            resolution = self.settings.resolution
            manual_calibration = self.db.active_calibration(entry_id)
            engine = ClassicalVisionEngine(
                self.settings.safety_margin_mm,
                self.settings.calibration_uncertainty_mm,
                WeedVisualVerifier(self.settings.data_dir / "weed_visual_model.json"),
            )
            weed_settings = (
                self.weed_settings_store.load() if self.weed_settings_store is not None else None
            )
            fusion_settings = CanopyFusionSettingsStore(
                self.settings.data_dir / "canopy_fusion_settings.json"
            ).load()
            # Boundaries and exclusion zones are read once per job so a mid-job
            # edit cannot make one image obey different rules than the next.
            zones: list[Zone] = self.zone_store.zones() if self.zone_store is not None else []
            wanted = [p for p in inventory.plants if not plant_ids or p.id in plant_ids]
            wanted_image_ids = set(image_ids)
            images = [
                image
                for image in sorted(inventory.images, key=lambda item: item.created_at)
                if image.processed
                and image.id not in excluded_grid_image_ids
                and (not wanted_image_ids or image.id in wanted_image_ids)
            ]
            LOGGER.info(
                "Analysis job %s selected %d of %d inventory image(s): trigger=%s "
                "requested=%d excluded_grid=%d plants=%d known_weeds=%d calibration=%s",
                job_id,
                len(images),
                len(inventory.images),
                trigger,
                len(wanted_image_ids),
                len(excluded_grid_image_ids),
                len(inventory.plants),
                len(inventory.weeds),
                "configured" if manual_calibration is not None else "not configured",
            )
            if wanted_image_ids and not images:
                LOGGER.warning(
                    "Analysis job %s received requested image IDs but none were available "
                    "as processed, non-excluded inventory images",
                    job_id,
                )
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
            self.current["images_total"] = len(images)
            self.current["images_failed"] = 0
            self.current["image_errors"] = []
            self.current["uncalibrated_images"] = 0
            self.current["weed_candidates"] = 0
            self.current["calibration_warnings"] = []
            self.current["calibration_source"] = None
            self.current["zone_blocked_weeds"] = 0
            self.current["zone_blocked_radius"] = 0
            artifacts = self.settings.data_dir / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            for image_number, image_info in enumerate(images):
                available, resource_reason = await self._yield_for_server()
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
                overlay_path = artifacts / f"{job_id}-{response.image_id}-overlay.jpg"
                weed_review_path = artifacts / f"{job_id}-{response.image_id}-weed-review.jpg"
                source_image_path = artifacts / f"{job_id}-{response.image_id}-photo.jpg"
                source_image_path.write_bytes(image_bytes)

                if resolved.calibration is None:
                    # No valid metric calibration: pixel-only diagnostics, no
                    # measurement, no write (Part 6).
                    LOGGER.warning(
                        "Analysis job %s image %d produced diagnostics only: no valid "
                        "metric calibration (warnings=%s)",
                        job_id,
                        response.image_id,
                        "; ".join(resolved.warnings) or "none",
                    )
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
                    self.current["images_processed"] += 1
                    LOGGER.info(
                        "Image %s persisted diagnostics only: source_photo=%s overlay=%s",
                        response.image_id,
                        source_image_path.is_file(),
                        overlay_path.is_file(),
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
                    # Every inventory plant participates in ownership, even
                    # when the caller requested measurements for a subset.
                    for plant in inventory.plants
                ]
                known_weed_seeds = [
                    KnownWeedSeed(
                        weed_id=weed.id,
                        center_px=garden_to_pixel(
                            weed.x,
                            weed.y,
                            response.meta.x,
                            response.meta.y,
                            response.width,
                            response.height,
                            calibration,
                        ),
                        radius_mm=weed.radius,
                    )
                    for weed in inventory.weeds
                ]
                previous_masks = {}
                for seed in seeds:
                    prior = decode_previous_mask(self.db.latest_mask_path(seed.plant_id))
                    if prior is not None:
                        previous_masks[seed.plant_id] = prior
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            engine.analyse,
                            image_bytes,
                            response.image_id,
                            response.meta.created_at,
                            seeds,
                            calibration,
                            previous_masks,
                            weed_settings,
                            known_weed_seeds,
                        ),
                        timeout=60,
                    )
                except Exception as exc:
                    self.current["images_failed"] += 1
                    error = {
                        "image_id": response.image_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self.current["image_errors"].append(error)
                    LOGGER.exception(
                        "Analysis skipped image %s after the vision engine failed: %s",
                        response.image_id,
                        exc,
                    )
                    continue
                del image_bytes, previous_masks
                decided = []
                plants_by_id = {plant.id: plant for plant in wanted}
                for item in result.measurements:
                    if item.plant_id not in plants_by_id:
                        continue
                    plant = plants_by_id[item.plant_id]
                    try:
                        transform_details = json.loads(item.transform_json or "{}")
                    except (TypeError, json.JSONDecodeError):
                        transform_details = {}
                    transform_details.update(
                        {
                            "image_x": response.meta.x,
                            "image_y": response.meta.y,
                            "image_z": response.meta.z,
                            "plant_x": plant.x,
                            "plant_y": plant.y,
                        }
                    )
                    grid_frame = grid_frames_by_image.get(int(item.image_id))
                    grid_target = (
                        grid_targets_by_index.get(int(grid_frame.target_index))
                        if grid_frame is not None
                        else None
                    )
                    if grid_target is not None:
                        transform_details.update(
                            {
                                "grid_session_id": grid_record.session_id,
                                "grid_target_index": grid_target.index,
                                "grid_row": grid_target.row,
                                "grid_column": grid_target.column,
                                "tile_footprint_width_mm": grid_record.footprint_width_mm,
                                "tile_footprint_height_mm": grid_record.footprint_height_mm,
                            }
                        )
                    item = item.model_copy(
                        update={
                            "recorded_center_x": plant.x,
                            "recorded_center_y": plant.y,
                            "transform_json": json.dumps(transform_details, separators=(",", ":")),
                        }
                    )
                    suggested_center = item.recommended_center_px
                    if suggested_center is not None:
                        suggested_center = pixel_to_garden(
                            suggested_center[0],
                            suggested_center[1],
                            response.meta.x,
                            response.meta.y,
                            response.width,
                            response.height,
                            calibration,
                        )
                        item = item.model_copy(update={"recommended_center_px": suggested_center})
                    if item.vegetation_absent:
                        item = item.model_copy(
                            update={
                                "config_entry_id": entry_id,
                                "absent_observations": self.db.absent_streak(
                                    entry_id,
                                    item.plant_id,
                                    current_image_id=item.image_id,
                                    current_image_timestamp=item.image_timestamp,
                                ),
                            }
                        )
                    else:
                        item = item.model_copy(update={"config_entry_id": entry_id})
                    decided.append(
                        decide(
                            item,
                            (
                                OperatingMode.RECOMMEND
                                if mode == OperatingMode.AUTO_RADIUS and len(images) > 1
                                else mode
                            ),
                            self.settings,
                            previously_observed_canopy=self.db.has_present_measurement(
                                entry_id, item.plant_id
                            ),
                        )
                    )
                if result.overlay_jpeg:
                    overlay_path.write_bytes(result.overlay_jpeg)
                if result.weed_review_jpeg:
                    weed_review_path.write_bytes(result.weed_review_jpeg)
                vegetation_path = artifacts / f"{job_id}-{response.image_id}-mask.png"
                if result.mask:
                    vegetation_path.write_bytes(result.mask)
                ownership = None
                if result.ownership_mask:
                    ownership = cv2.imdecode(
                        np.frombuffer(result.ownership_mask, dtype=np.uint8), cv2.IMREAD_UNCHANGED
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
                    artifact_paths = [str(overlay_path)] if result.overlay_jpeg else []
                    if result.mask:
                        artifact_paths.append(str(vegetation_path))
                    if ownership is not None:
                        artifact_paths.append(str(mask_path))
                    persisted.append(
                        item.model_copy(
                            update={
                                "overlay_path": str(overlay_path),
                                "mask_path": str(mask_path) if ownership is not None else None,
                                "artifact_paths": artifact_paths,
                                "source_image_path": str(source_image_path),
                            }
                        )
                    )
                # Persist plant evidence before per-image weed writes. This
                # keeps plant recommendations reviewable if a later storage
                # operation fails after weed candidates have been detected.
                decided = persisted
                self.db.save_measurements(decided)
                self.current["images_processed"] += 1
                LOGGER.info(
                    "Image %s persisted: plant_measurements=%d artifacts=%d source_photo=%s "
                    "overlay=%s",
                    response.image_id,
                    len(decided),
                    len(decided[0].artifact_paths) if decided else 0,
                    source_image_path.is_file(),
                    overlay_path.is_file(),
                )
                self.current["weed_candidates"] += len(result.weeds)
                if result.weed_candidate_stats:
                    stats = result.weed_candidate_stats
                    LOGGER.info(
                        "Image %s weed candidates: %s found, %s detected "
                        "(%s crop-protected, %s oversized, and %s shadow-rescued scored; "
                        "dropped: %s size, %s colour/shape, %s low score)",
                        response.image_id,
                        stats.get("blobs", 0),
                        len(result.weeds),
                        stats.get("protected_scored", 0),
                        stats.get("oversized_scored", 0),
                        stats.get("verifier_rescued", 0),
                        stats.get("size", 0),
                        stats.get("shape", 0),
                        stats.get("score", 0),
                    )
                if result.boundary_verifier_stats:
                    LOGGER.info(
                        "Image %s plant-boundary checks: %s",
                        response.image_id,
                        json.dumps(result.boundary_verifier_stats, separators=(",", ":")),
                    )
                matched_known_weed_ids: set[int] = set()
                known_weeds_by_id = {weed.id: weed for weed in inventory.weeds}
                known_observations = {
                    observation.weed_id: observation
                    for observation in result.known_weed_observations
                }
                for weed in result.weeds:
                    crop_path = artifacts / (
                        f"{job_id}-{response.image_id}-weed-{weed.detection_id}-crop.jpg"
                    )
                    if weed.crop_jpeg:
                        crop_path.write_bytes(weed.crop_jpeg)
                    stored_crop_path = str(crop_path) if weed.crop_jpeg else None
                    weed_x, weed_y = pixel_to_garden(
                        weed.center_px[0],
                        weed.center_px[1],
                        response.meta.x,
                        response.meta.y,
                        response.width,
                        response.height,
                        calibration,
                    )
                    duplicate_distance = max(20.0, weed.radius_mm * 1.5)
                    associated_known_id = int(weed.features.get("known_weed_id", 0))
                    known_weed = known_weeds_by_id.get(associated_known_id)
                    if known_weed is None:
                        known_weed = min(
                            (
                                existing
                                for existing in inventory.weeds
                                if math.hypot(existing.x - weed_x, existing.y - weed_y)
                                <= max(duplicate_distance, existing.radius)
                            ),
                            key=lambda existing: math.hypot(
                                existing.x - weed_x, existing.y - weed_y
                            ),
                            default=None,
                        )
                    if known_weed is not None:
                        matched_known_weed_ids.add(known_weed.id)
                        measured_radius = max(float(known_weed.radius), float(weed.radius_mm))
                        baseline_radius = self.db.radius_growth_baseline(
                            entity_type="weed",
                            entity_id=known_weed.id,
                            since=datetime.now(UTC) - timedelta(hours=24),
                            current_radius_mm=known_weed.radius,
                        )
                        target_radius, maximum_radius = limit_weed_radius_growth(
                            known_weed.radius,
                            measured_radius,
                            baseline_radius,
                            weed_settings.maximum_radius_growth_mm_per_day,
                            weed_settings.maximum_radius_growth_percent_per_day,
                        )
                        status = "matched"
                        observation = known_observations.get(known_weed.id)
                        verifier_allows_radius = bool(
                            weed_settings.visual_verifier_enabled
                            and not weed_settings.visual_verifier_shadow_mode
                            and weed.verifier_confidence is not None
                            and weed.verifier_confidence
                            >= weed_settings.visual_verifier_acceptance_confidence
                            and observation is not None
                            and observation.status == "present"
                        )
                        prior_track = self.db.weed_track(entry_id, known_weed.id)
                        already_observed = self.db.has_known_weed_observation(
                            entry_id, known_weed.id, weed.image_id
                        )
                        present_observations = (
                            int((prior_track or {}).get("present_observations") or 0)
                            + (0 if already_observed else 1)
                            if verifier_allows_radius
                            else 0
                        )
                        if (
                            weed_settings
                            and weed_settings.automatic_radius_adjustment
                            and trigger != "weeding_verification"
                            and target_radius > float(known_weed.radius) + 0.5
                            and verifier_allows_radius
                            and not already_observed
                            and present_observations >= weed_settings.radius_min_consecutive_present
                            and not bool(weed.features.get("extent_permissive_fallback", 0.0))
                        ):
                            try:
                                update_result = await self.client.update_weed_radius(
                                    UpdateWeedRadiusRequest(
                                        config_entry_id=entry_id,
                                        weed_id=known_weed.id,
                                        expected_current_radius_mm=known_weed.radius,
                                        recommended_radius_mm=target_radius,
                                        confidence=weed.confidence,
                                        apply=True,
                                    )
                                )
                                if update_result.get("status") == "applied":
                                    status = "radius_adjusted"
                                    self.db.record_change(
                                        entity_type="weed",
                                        entity_id=known_weed.id,
                                        crop_type="weed",
                                        x=known_weed.x,
                                        y=known_weed.y,
                                        z=known_weed.z,
                                        change_type="radius increased",
                                        original_radius_mm=known_weed.radius,
                                        current_radius_mm=float(
                                            update_result.get("radius_mm", target_radius)
                                        ),
                                        decision_method="auto",
                                        confidence=weed.confidence,
                                        details={
                                            **update_result,
                                            "measured_radius_mm": measured_radius,
                                            "rolling_baseline_radius_mm": baseline_radius,
                                            "maximum_allowed_radius_mm": maximum_radius,
                                            "image_artifacts": [
                                                path
                                                for path in (
                                                    str(overlay_path)
                                                    if result.overlay_jpeg
                                                    else None,
                                                    str(weed_review_path)
                                                    if result.weed_review_jpeg
                                                    else None,
                                                    stored_crop_path,
                                                )
                                                if path
                                            ],
                                            "image_id": weed.image_id,
                                        },
                                    )
                            except HomeAssistantError as exc:
                                LOGGER.warning("Automatic weed radius adjustment failed: %s", exc)
                        self.db.upsert_weed_track(
                            config_entry_id=entry_id,
                            weed_id=known_weed.id,
                            x=known_weed.x,
                            y=known_weed.y,
                            radius_mm=target_radius
                            if status == "radius_adjusted"
                            else known_weed.radius,
                            confidence=weed.confidence,
                            seen_at=weed.image_timestamp,
                            status=status,
                            absent_observations=0,
                            present_observations=present_observations,
                            observation_image_id=weed.image_id,
                        )
                        self.db.record_known_weed_observation(
                            entry_id,
                            known_weed.id,
                            weed.image_id,
                            observation.status if observation is not None else "inconclusive",
                            observation.confidence if observation is not None else 0,
                        )
                        self.db.save_weed_detection(
                            detection_id=str(weed.detection_id),
                            config_entry_id=entry_id,
                            image_id=weed.image_id,
                            image_timestamp=weed.image_timestamp,
                            x=weed_x,
                            y=weed_y,
                            z=response.meta.z,
                            area_mm2=weed.area_mm2,
                            radius_mm=weed.radius_mm,
                            confidence=weed.confidence,
                            overlay_path=str(overlay_path) if result.overlay_jpeg else None,
                            review_path=(
                                str(weed_review_path) if result.weed_review_jpeg else None
                            ),
                            status=status,
                            center_px_x=weed.center_px[0],
                            center_px_y=weed.center_px[1],
                            processed_width=response.width,
                            processed_height=response.height,
                            heuristic_confidence=weed.heuristic_confidence,
                            verifier_confidence=weed.verifier_confidence,
                            features=weed.features,
                            crop_path=stored_crop_path,
                        )
                        continue
                    # A post-weeding scan has one question only: is each weed
                    # we deliberately mowed still present? Unknown vegetation
                    # in those frames must not create, recommend, or train a
                    # new weed as a side effect of verification.
                    if trigger == "weeding_verification":
                        continue
                    if self.db.has_terminal_weed_detection_near(
                        entry_id,
                        weed_x,
                        weed_y,
                        duplicate_distance,
                        source_image_id=weed.image_id,
                        source_image_timestamp=weed.image_timestamp,
                    ):
                        continue
                    # Zones decide where a weed may exist at all: a position the
                    # user has ruled out is not stored, so it is never created
                    # automatically and never offered for review either.
                    zone_check = evaluate(zones, ZoneAspect.WEEDS, weed_x, weed_y)
                    if not zone_check.allowed:
                        self.current["zone_blocked_weeds"] += 1
                        LOGGER.info(
                            "Weed at (%.1f, %.1f) discarded: %s",
                            weed_x,
                            weed_y,
                            zone_check.reason,
                        )
                        continue
                    track = self.db.observe_weed_candidate(
                        config_entry_id=entry_id,
                        image_id=weed.image_id,
                        seen_at=weed.image_timestamp,
                        x=weed_x,
                        y=weed_y,
                        confidence=weed.confidence,
                        match_distance_mm=weed_settings.temporal_match_distance_mm,
                        max_gap_hours=weed_settings.temporal_max_gap_hours,
                    )
                    observations = int(track["observations"])
                    recommendation_observations = (
                        weed_settings.recommendation_min_observations
                        if weed_settings.temporal_confirmation_enabled
                        else 1
                    )
                    weed_status = (
                        "recommended"
                        if observations >= recommendation_observations
                        else "observing"
                    )
                    automatic_observations = (
                        weed_settings.automatic_min_observations
                        if weed_settings.temporal_confirmation_enabled
                        else 1
                    )
                    verifier_allows_automatic = (
                        weed_settings.visual_verifier_enabled
                        and not weed_settings.visual_verifier_shadow_mode
                        and (
                            weed.verifier_confidence is not None
                            and weed.verifier_confidence
                            >= weed_settings.visual_verifier_acceptance_confidence
                        )
                    )
                    if (
                        weed_settings
                        and weed_settings.automatic_creation
                        and verifier_allows_automatic
                        and weed.verifier_confidence is not None
                        and weed.verifier_confidence >= weed_settings.automatic_creation_confidence
                        and observations >= automatic_observations
                        # Verifier-enabled candidates are intentionally visible
                        # inside crop protection so misses can be reviewed and
                        # labelled.  Visibility does not grant authority to
                        # create a FarmBot weed on or beside a known crop.
                        and not bool(weed.features.get("crop_protection_overlap", 0.0))
                        and not bool(weed.features.get("configured_maximum_area_exceeded", 0.0))
                        and not bool(weed.features.get("extent_permissive_fallback", 0.0))
                    ):
                        try:
                            create_result = await self.client.create_weed(
                                CreateWeedRequest(
                                    config_entry_id=entry_id,
                                    detection_id=weed.detection_id,
                                    x=weed_x,
                                    y=weed_y,
                                    z=response.meta.z,
                                    radius=weed.radius_mm,
                                    confidence=weed.confidence,
                                    apply=True,
                                )
                            )
                            if create_result.get("status") == "applied":
                                weed_status = "created"
                                self.db.record_change(
                                    entity_type="weed",
                                    entity_id=create_result.get("weed_id") or weed.detection_id,
                                    crop_type="weed",
                                    x=weed_x,
                                    y=weed_y,
                                    z=response.meta.z,
                                    change_type="weed added",
                                    original_radius_mm=0,
                                    current_radius_mm=weed.radius_mm,
                                    decision_method="auto",
                                    confidence=weed.confidence,
                                    details={
                                        **create_result,
                                        "image_artifacts": [
                                            path
                                            for path in (
                                                str(overlay_path) if result.overlay_jpeg else None,
                                                str(weed_review_path)
                                                if result.weed_review_jpeg
                                                else None,
                                                stored_crop_path,
                                            )
                                            if path
                                        ],
                                        "image_id": weed.image_id,
                                    },
                                )
                        except HomeAssistantError as exc:
                            LOGGER.warning(
                                "Automatic weed creation failed; keeping recommendation: %s", exc
                            )
                    self.db.supersede_pending_weed_detections(
                        entry_id, weed_x, weed_y, duplicate_distance
                    )
                    self.db.save_weed_detection(
                        detection_id=str(weed.detection_id),
                        config_entry_id=entry_id,
                        image_id=weed.image_id,
                        image_timestamp=weed.image_timestamp,
                        x=weed_x,
                        y=weed_y,
                        z=response.meta.z,
                        area_mm2=weed.area_mm2,
                        radius_mm=weed.radius_mm,
                        confidence=weed.confidence,
                        overlay_path=str(overlay_path) if result.overlay_jpeg else None,
                        review_path=(str(weed_review_path) if result.weed_review_jpeg else None),
                        status=weed_status,
                        center_px_x=weed.center_px[0],
                        center_px_y=weed.center_px[1],
                        processed_width=response.width,
                        processed_height=response.height,
                        heuristic_confidence=weed.heuristic_confidence,
                        verifier_confidence=weed.verifier_confidence,
                        features=weed.features,
                        crop_path=stored_crop_path,
                        observation_count=observations,
                        candidate_track_id=int(track["id"]),
                    )
                if weed_settings and weed_settings.enabled:
                    for known_weed in inventory.weeds:
                        px, py = garden_to_pixel(
                            known_weed.x,
                            known_weed.y,
                            response.meta.x,
                            response.meta.y,
                            response.width,
                            response.height,
                            calibration,
                        )
                        margin_px = max(
                            8.0,
                            (known_weed.radius + weed_settings.plant_exclusion_margin_mm)
                            * math.sqrt(calibration.pixels_per_mm_x * calibration.pixels_per_mm_y),
                        )
                        fully_visible = (
                            margin_px <= px < response.width - margin_px
                            and margin_px <= py < response.height - margin_px
                        )
                        if not fully_visible or known_weed.id in matched_known_weed_ids:
                            continue
                        observation = known_observations.get(known_weed.id)
                        if observation is None:
                            # Older/custom engines that do not return explicit
                            # known-weed evidence must never imply absence.
                            continue
                        prior_track = self.db.weed_track(entry_id, known_weed.id)
                        already_observed = self.db.has_known_weed_observation(
                            entry_id, known_weed.id, response.image_id
                        )
                        if already_observed:
                            continue
                        if observation.status != "absent":
                            self.db.upsert_weed_track(
                                config_entry_id=entry_id,
                                weed_id=known_weed.id,
                                x=known_weed.x,
                                y=known_weed.y,
                                radius_mm=known_weed.radius,
                                confidence=observation.confidence,
                                seen_at=None,
                                status=observation.status,
                                absent_observations=0,
                                present_observations=0,
                                observation_image_id=response.image_id,
                            )
                            self.db.record_known_weed_observation(
                                entry_id,
                                known_weed.id,
                                response.image_id,
                                observation.status,
                                observation.confidence,
                            )
                            continue
                        absent_observations = int(
                            (prior_track or {}).get("absent_observations") or 0
                        ) + (0 if already_observed else 1)
                        prior_confidence = (
                            float((prior_track or {}).get("confidence") or 1.0)
                            if int((prior_track or {}).get("absent_observations") or 0)
                            else 1.0
                        )
                        # Repeated observations add safety through the streak
                        # gate; they must not manufacture a larger probability.
                        absence_confidence = min(prior_confidence, observation.confidence)
                        track_status = "removal_recommended"
                        if (
                            weed_settings.automatic_removal
                            and trigger != "weeding_verification"
                            and weed_settings.visual_verifier_enabled
                            and not weed_settings.visual_verifier_shadow_mode
                            and not already_observed
                            and absent_observations >= weed_settings.removal_min_consecutive_absent
                        ):
                            try:
                                removal_result = await self.client.remove_weed(
                                    RemoveWeedRequest(
                                        config_entry_id=entry_id,
                                        weed_id=known_weed.id,
                                        confidence=absence_confidence,
                                        apply=True,
                                    )
                                )
                                if removal_result.get("status") == "applied":
                                    track_status = "removed"
                                    self.db.record_change(
                                        entity_type="weed",
                                        entity_id=known_weed.id,
                                        crop_type="weed",
                                        x=known_weed.x,
                                        y=known_weed.y,
                                        z=known_weed.z,
                                        change_type="weed removed",
                                        original_radius_mm=known_weed.radius,
                                        current_radius_mm=0,
                                        decision_method="auto",
                                        confidence=absence_confidence,
                                        details={
                                            **removal_result,
                                            "image_artifacts": [
                                                path
                                                for path in (
                                                    str(overlay_path)
                                                    if result.overlay_jpeg
                                                    else None,
                                                    str(weed_review_path)
                                                    if result.weed_review_jpeg
                                                    else None,
                                                )
                                                if path
                                            ],
                                            "image_id": response.image_id,
                                        },
                                    )
                            except HomeAssistantError as exc:
                                LOGGER.warning("Automatic weed removal failed: %s", exc)
                        self.db.upsert_weed_track(
                            config_entry_id=entry_id,
                            weed_id=known_weed.id,
                            x=known_weed.x,
                            y=known_weed.y,
                            radius_mm=known_weed.radius,
                            confidence=absence_confidence,
                            seen_at=None,
                            status=track_status,
                            absent_observations=absent_observations,
                            present_observations=0,
                            observation_image_id=response.image_id,
                        )
                        self.db.record_known_weed_observation(
                            entry_id,
                            known_weed.id,
                            response.image_id,
                            observation.status,
                            observation.confidence,
                        )
                skip_reasons = self.current.setdefault("skip_reasons", {})
                for plant_id, reason in result.skipped.items():
                    skip_reasons[str(plant_id)] = reason
                del result
                if mode == OperatingMode.AUTO_RADIUS and len(images) == 1:
                    for item_index, item in enumerate(decided):
                        if item.decision == Decision.REMOVED:
                            try:
                                removal_result = await self.client.apply_removal(
                                    ApplyRemovalRequest(
                                        config_entry_id=entry_id,
                                        plant_id=item.plant_id,
                                        measurement_id=item.measurement_id,
                                        expected_current_radius_mm=item.current_radius_mm,
                                        confidence=item.confidence,
                                        apply=True,
                                    )
                                )
                                removal_status = str(removal_result.get("status", "error"))
                                removal_applied = removal_status == "applied"
                                updated = item.model_copy(
                                    update={
                                        "decision": (
                                            Decision.REMOVED
                                            if removal_applied
                                            else Decision.REMOVAL_RECOMMENDED
                                        ),
                                        "applied": removal_applied,
                                    }
                                )
                                decided[item_index] = updated
                                self.db.update_measurement_outcome(
                                    str(item.measurement_id),
                                    decision=updated.decision.value,
                                    applied=updated.applied,
                                )
                                self.db.record_decision(
                                    str(item.measurement_id),
                                    "removed"
                                    if removal_status == "applied"
                                    else "removal_rejected",
                                    removal_result,
                                )
                                if removal_applied:
                                    self.db.record_change(
                                        entity_type="plant",
                                        entity_id=item.plant_id,
                                        crop_type=item.crop_slug,
                                        x=item.recorded_center_x,
                                        y=item.recorded_center_y,
                                        z=None,
                                        change_type="plant removed",
                                        original_radius_mm=float(
                                            removal_result.get(
                                                "old_radius_mm", item.current_radius_mm
                                            )
                                        ),
                                        current_radius_mm=0,
                                        decision_method="auto",
                                        confidence=item.confidence,
                                        details=_automatic_change_details(removal_result, [item]),
                                    )
                            except StaleRadiusError:
                                decided[item_index] = item.model_copy(
                                    update={"decision": Decision.REMOVAL_RECOMMENDED}
                                )
                                self.db.update_measurement_outcome(
                                    str(item.measurement_id),
                                    decision=Decision.REMOVAL_RECOMMENDED.value,
                                    applied=False,
                                )
                                self.db.record_decision(
                                    str(item.measurement_id), "removal_conflict", {}
                                )
                                await self.client.inventory(
                                    InventoryRequest(
                                        config_entry_id=entry_id,
                                        image_lookback_hours=self.settings.image_lookback_hours,
                                    )
                                )
                            continue
                        # Never write without a valid calibration (Part 6).
                        if item.decision != Decision.APPLIED or not item.calibrated:
                            continue
                        # A protection radius may only grow into areas the zones
                        # permit. Blocked growth stays a recommendation so the
                        # zone can be adjusted or the result rejected by hand.
                        radius_zone_check = (
                            evaluate(
                                zones,
                                ZoneAspect.RADIUS,
                                item.recorded_center_x,
                                item.recorded_center_y,
                                item.recommended_protection_radius_mm,
                            )
                            if item.recorded_center_x is not None
                            and item.recorded_center_y is not None
                            else None
                        )
                        if radius_zone_check is not None and not radius_zone_check.allowed:
                            self.current["zone_blocked_radius"] += 1
                            decided[item_index] = item.model_copy(
                                update={
                                    "decision": Decision.RECOMMENDED,
                                    "applied": False,
                                    "reason": (
                                        f"{item.reason}; radius growth blocked: "
                                        f"{radius_zone_check.reason}"
                                    ),
                                }
                            )
                            self.db.update_measurement_outcome(
                                str(item.measurement_id),
                                decision=Decision.RECOMMENDED.value,
                                applied=False,
                                reason=decided[item_index].reason,
                            )
                            self.db.record_decision(
                                str(item.measurement_id),
                                "zone_blocked",
                                {"aspect": "radius", "reason": radius_zone_check.reason},
                            )
                            continue
                        try:
                            apply_result = await self.client.apply_radius(
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
                            apply_status = str(apply_result.get("status", "error"))
                            radius_applied = apply_status == "applied"
                            fallback_decision = (
                                Decision.UNCERTAIN
                                if apply_status == "conflict"
                                else Decision.RECOMMENDED
                            )
                            updated = item.model_copy(
                                update={
                                    "decision": Decision.APPLIED
                                    if radius_applied
                                    else fallback_decision,
                                    "applied": radius_applied,
                                }
                            )
                            decided[item_index] = updated
                            self.db.update_measurement_outcome(
                                str(item.measurement_id),
                                decision=updated.decision.value,
                                applied=updated.applied,
                            )
                            self.db.record_decision(
                                str(item.measurement_id),
                                "applied" if apply_status == "applied" else "apply_rejected",
                                apply_result,
                            )
                            if radius_applied:
                                new_radius = float(
                                    apply_result.get(
                                        "new_radius_mm",
                                        apply_result.get(
                                            "radius_mm", item.recommended_protection_radius_mm
                                        ),
                                    )
                                )
                                self.db.record_change(
                                    entity_type="plant",
                                    entity_id=item.plant_id,
                                    crop_type=item.crop_slug,
                                    x=item.recorded_center_x,
                                    y=item.recorded_center_y,
                                    z=None,
                                    change_type=(
                                        "radius increased"
                                        if new_radius > item.current_radius_mm
                                        else "radius decreased"
                                    ),
                                    original_radius_mm=item.current_radius_mm,
                                    current_radius_mm=new_radius,
                                    decision_method="auto",
                                    confidence=item.confidence,
                                    details=_automatic_change_details(apply_result, [item]),
                                )
                            if radius_applied:
                                await self._update_curve_after_radius(
                                    entry_id, inventory, updated, human_approved=False
                                )
                            elif apply_status == "conflict":
                                await self.client.inventory(
                                    InventoryRequest(
                                        config_entry_id=entry_id,
                                        image_lookback_hours=self.settings.image_lookback_hours,
                                    )
                                )
                        except StaleRadiusError:
                            decided[item_index] = item.model_copy(
                                update={"decision": Decision.UNCERTAIN}
                            )
                            self.db.update_measurement_outcome(
                                str(item.measurement_id),
                                decision=Decision.UNCERTAIN.value,
                                applied=False,
                            )
                            self.db.record_decision(str(item.measurement_id), "stale_radius", {})
                            await self.client.inventory(
                                InventoryRequest(
                                    config_entry_id=entry_id,
                                    image_lookback_hours=self.settings.image_lookback_hours,
                                )
                            )
                all_measurements.extend(decided)
            measurements_by_plant: dict[int, list] = {}
            for measurement in all_measurements:
                measurements_by_plant.setdefault(measurement.plant_id, []).append(measurement)
            current_image_ids = {int(item.image_id) for item in all_measurements}
            current_grid_image_ids = set(grid_frames_by_image)
            # A photo grid is uploaded as separate new-image events. Include
            # measurements already saved for this verified grid so evidence
            # selection, fusion, and the review image describe one garden
            # neighbourhood rather than whichever upload happened to run last.
            if (
                current_image_ids
                and current_grid_image_ids
                and current_image_ids <= current_grid_image_ids
            ):
                for plant_id in list(measurements_by_plant):
                    grid_measurements = self.db.pending_plant_measurements(
                        entry_id,
                        plant_id,
                        current_grid_image_ids,
                    )
                    if grid_measurements:
                        measurements_by_plant[plant_id] = grid_measurements
            proposed_radii = {
                candidate_plant_id: float(
                    Database._consolidate_measurement_rows(
                        [item.model_dump(mode="json") for item in candidate_measurements]
                    )["recommended_protection_radius_mm"]
                )
                for candidate_plant_id, candidate_measurements in measurements_by_plant.items()
            }
            for plant_id, plant_measurements in measurements_by_plant.items():
                evidence = select_measurement_evidence(plant_measurements)
                evidence_diagnostics = selection_diagnostics(evidence)
                self.db.set_evidence_selection(plant_measurements)
                LOGGER.info(
                    "Plant evidence: plant_id=%s crop=%s candidates=%d useful=%d "
                    "selected=%d mode=%s coverage=%.0f%%",
                    plant_id,
                    plant_measurements[0].crop_slug,
                    len(evidence.candidates),
                    len(evidence.useful),
                    len(evidence.used),
                    evidence.mode,
                    evidence.boundary_coverage * 100,
                )
                LOGGER.debug(
                    "Plant measurement evidence details: %s",
                    json.dumps(
                        {
                            "plant_id": plant_id,
                            "crop_name": plant_measurements[0].crop_slug,
                            "plant_center_bed_mm": [
                                plant_measurements[0].recorded_center_x,
                                plant_measurements[0].recorded_center_y,
                            ],
                            **evidence_diagnostics,
                            "candidate_footprints": [
                                {
                                    "image_id": item.image_id,
                                    "transform": json.loads(item.transform_json or "{}"),
                                    "boundary_coverage": item.boundary_coverage,
                                    "image_quality": item.image_quality,
                                    "segmentation_quality": item.segmentation_quality,
                                }
                                for item in plant_measurements
                            ],
                        },
                        separators=(",", ":"),
                    ),
                )
                fused = await asyncio.to_thread(
                    fuse_canopy_masks, plant_measurements, fusion_settings
                )
                if fused is not None:
                    individual = Database._consolidate_measurement_rows(
                        [item.model_dump(mode="json") for item in plant_measurements]
                    )
                    disagreement = abs(
                        fused.maximum_radius_mm
                        - float(individual["maximum_accepted_canopy_radius_mm"])
                    )
                    reliable = (
                        fused.angular_coverage >= fusion_settings.minimum_angular_coverage
                        and (
                            fused.view_count == 1
                            or fused.corroborated_fraction
                            >= fusion_settings.minimum_corroborated_fraction
                        )
                        and disagreement <= fusion_settings.maximum_automatic_disagreement_mm
                    )
                    safety_margin = max(item.safety_margin_mm for item in plant_measurements)
                    calibration_uncertainty = max(
                        item.calibration_uncertainty_mm for item in plant_measurements
                    )
                    diagnostic_path = artifacts / f"{job_id}-plant-{plant_id}-fusion.jpg"
                    stored_diagnostic = None
                    if fusion_settings.save_diagnostics and fused.diagnostic_jpeg is not None:
                        diagnostic_path.write_bytes(fused.diagnostic_jpeg)
                        stored_diagnostic = str(diagnostic_path)
                    fused_values = {
                        "fused_canopy": True,
                        "fused_typical_radius_mm": fused.typical_radius_mm,
                        "fused_maximum_radius_mm": fused.maximum_radius_mm,
                        "fused_recommended_radius_mm": (
                            fused.maximum_radius_mm + safety_margin + calibration_uncertainty
                        ),
                        "fused_confidence": fused.confidence,
                        "fusion_view_count": fused.view_count,
                        "fusion_angular_coverage": fused.angular_coverage,
                        "fusion_corroborated_fraction": fused.corroborated_fraction,
                        "fusion_disagreement_mm": disagreement,
                        "fusion_reliable": reliable,
                        "fusion_diagnostic_path": stored_diagnostic,
                    }
                    proposed_radii[plant_id] = float(fused_values["fused_recommended_radius_mm"])
                    self.db.set_fused_canopy(
                        [str(item.measurement_id) for item in plant_measurements],
                        fused_values,
                    )
                    plant_measurements = [
                        item.model_copy(update=fused_values) for item in plant_measurements
                    ]
                if mode == OperatingMode.AUTO_RADIUS:
                    await self._apply_consolidated_automatic(
                        entry_id=entry_id,
                        inventory=inventory,
                        measurements=plant_measurements,
                        zones=zones,
                        fusion_settings=fusion_settings,
                    )
                composite_path = artifacts / f"{job_id}-plant-{plant_id}-composite.jpg"
                composite_overlay_path = (
                    artifacts / f"{job_id}-plant-{plant_id}-composite-overlay.jpg"
                )
                composite_built = await asyncio.to_thread(
                    build_plant_composite,
                    plant_measurements,
                    composite_path,
                    composite_overlay_path,
                    plants=inventory.plants,
                    proposed_radii=proposed_radii,
                    grid_record=grid_record,
                )
                if composite_built:
                    self.db.set_composite_path(
                        [str(item.measurement_id) for item in plant_measurements],
                        str(composite_path),
                        str(composite_overlay_path),
                    )
            image_errors = self.current.get("image_errors", [])
            completion_status = "warning" if image_errors else "idle"
            completion_message = (
                f"completed with {len(image_errors)} image error(s)"
                if image_errors
                else "completed"
            )
            return await self._finish(
                entry_id,
                job_id,
                completion_status,
                start_wall,
                start_cpu,
                all_measurements,
                completion_message,
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
            "images_failed": self.current.get("images_failed", 0),
            "image_errors": self.current.get("image_errors", []),
            "uncalibrated_images": self.current.get("uncalibrated_images", 0),
            "weed_candidates": self.current.get("weed_candidates", 0),
            "zone_blocked_weeds": self.current.get("zone_blocked_weeds", 0),
            "zone_blocked_radius": self.current.get("zone_blocked_radius", 0),
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
        self.current = {
            "status": "idle",
            "queue_length": len(self.queued_image_ids),
            "progress": message,
        }
        self.db.finish_job(str(job_id), result)
        await self._status(entry_id, job_id, status, message, measurements)
        LOGGER.info(
            "Analysis job %s finished: status=%s images=%d failed=%d uncalibrated=%d "
            "plants=%d measured=%d recommendations=%d automatic=%d uncertain=%d message=%s",
            job_id,
            status,
            result["images_processed"],
            result["images_failed"],
            result["uncalibrated_images"],
            result["plants_analysed"],
            result["plants_measured"],
            result["recommendations"],
            result["automatically_applied"],
            result["uncertain"],
            message,
        )
        return {"accepted": True, **result}


def decode_previous_mask(path: str | None) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)
