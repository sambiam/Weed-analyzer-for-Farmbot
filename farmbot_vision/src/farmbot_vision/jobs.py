from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
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
    OperatingMode,
    PlantSeed,
    RemoveWeedRequest,
    UpdateWeedRadiusRequest,
    UpsertCurveRequest,
    VisionImageRequest,
    VisionStatus,
)
from .safety import decide
from .settings import Settings
from .vision import ClassicalVisionEngine, garden_to_pixel, pixel_to_garden
from .weed_settings import WeedSettingsStore
from .weed_verifier import WeedVisualVerifier
from .zones import Zone, ZoneAspect, ZoneStore, evaluate

LOGGER = logging.getLogger(__name__)


def build_plant_composite(
    measurements: list,
    output_path: Path,
    overlay_output_path: Path | None = None,
) -> bool:
    """Stitch all views of one plant using their calibrated garden transforms.

    ``output_path`` is the clean photo composite. When ``overlay_output_path``
    is supplied, a second copy is written with the plant ownership masks
    tinted over the same stitched pixels. Radius and centre annotations are
    drawn on both copies.
    """
    frames = []
    for item in measurements:
        if not item.source_image_path or not item.plant_center_px:
            continue
        image = cv2.imread(item.source_image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        try:
            transform = json.loads(item.transform_json or "{}")
        except (TypeError, json.JSONDecodeError):
            transform = {}
        ppm_x = float(transform.get("pixels_per_mm_x") or 0)
        ppm_y = float(transform.get("pixels_per_mm_y") or 0)
        if ppm_x <= 0 or ppm_y <= 0:
            continue
        rotation = math.radians(float(transform.get("rotation_degrees") or 0))
        cos_t, sin_t = math.cos(rotation), math.sin(rotation)
        origin = str(transform.get("origin_location") or "top_left")
        sign_x = -1 if origin in {"top_right", "bottom_right"} else 1
        sign_y = -1 if origin in {"bottom_left", "bottom_right"} else 1
        cx, cy = item.plant_center_px
        height, width = image.shape[:2]
        # Source pixel -> millimetres relative to this plant. This is the
        # relative form of vision.pixel_to_garden: anchoring each photo at the
        # same plant cancels camera position and calibration offsets while
        # retaining rotation, scale and origin reflection.
        relative_transform = np.float64(
            [
                [
                    sign_x * cos_t / ppm_x,
                    -sign_x * sin_t / ppm_x,
                    sign_x * (-cos_t * cx + sin_t * cy) / ppm_x,
                ],
                [
                    sign_y * sin_t / ppm_y,
                    sign_y * cos_t / ppm_y,
                    sign_y * (-sin_t * cx - cos_t * cy) / ppm_y,
                ],
            ]
        )
        corners = cv2.transform(
            np.float64([[[0, 0], [width, 0], [0, height], [width, height]]]),
            relative_transform,
        )[0]
        frames.append(
            {
                "item": item,
                "image": image,
                "transform": relative_transform,
                "bounds": (
                    float(corners[:, 0].min()),
                    float(corners[:, 0].max()),
                    float(corners[:, 1].min()),
                    float(corners[:, 1].max()),
                ),
                "scale": math.sqrt(ppm_x * ppm_y),
            }
        )
    if not frames:
        return False
    min_x = min(frame["bounds"][0] for frame in frames)
    max_x = max(frame["bounds"][1] for frame in frames)
    min_y = min(frame["bounds"][2] for frame in frames)
    max_y = max(frame["bounds"][3] for frame in frames)
    ppm = float(np.median([frame["scale"] for frame in frames]))
    ppm = min(ppm, 2400 / max(1.0, max(max_x - min_x, max_y - min_y)))
    canvas_width = max(1, round((max_x - min_x) * ppm))
    canvas_height = max(1, round((max_y - min_y) * ppm))
    accumulated = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    ownership = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    for frame in sorted(frames, key=lambda frame: frame["item"].image_timestamp):
        affine = frame["transform"].copy()
        affine[0] = affine[0] * ppm
        affine[1] = affine[1] * ppm
        affine[0, 2] -= min_x * ppm
        affine[1, 2] -= min_y * ppm
        size = (canvas_width, canvas_height)
        warped = cv2.warpAffine(
            frame["image"], affine, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        valid = cv2.warpAffine(
            np.full(frame["image"].shape[:2], 255, dtype=np.uint8),
            affine,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        selected = valid > 0
        accumulated[selected] += warped[selected]
        weights[selected] += 1
        mask_path = frame["item"].mask_path
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                warped_mask = cv2.warpAffine(
                    mask,
                    affine,
                    size,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                ownership = cv2.max(ownership, warped_mask)
    canvas = np.zeros_like(accumulated, dtype=np.uint8)
    present = weights > 0
    canvas[present] = np.clip(accumulated[present] / weights[present, np.newaxis], 0, 255).astype(
        np.uint8
    )
    overlay_canvas = canvas.copy()
    selected = ownership > 0
    overlay_canvas[selected] = (
        overlay_canvas[selected].astype(np.float32) * 0.58
        + np.asarray((255, 190, 20), dtype=np.float32) * 0.42
    ).astype(np.uint8)
    representative = Database._consolidate_measurement_rows(
        [item.model_dump(mode="json") for item in measurements]
    )
    center = (round(-min_x * ppm), round(-min_y * ppm))
    current = float(representative["current_radius_mm"])
    planned = float(representative["recommended_protection_radius_mm"])
    thickness = max(4, round(min(canvas_width, canvas_height) / 300))
    label = f"original {current:.1f} mm | new {planned:.1f} mm"

    def annotate(target: np.ndarray) -> None:
        cv2.circle(target, center, max(1, round(current * ppm)), (255, 255, 0), thickness)
        cv2.circle(target, center, max(1, round(planned * ppm)), (0, 0, 255), thickness + 1)
        dot_radius = max(5, thickness + 2)
        cv2.circle(target, center, dot_radius + 2, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(target, center, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)
        text_origin = (12, max(28, round(canvas_height * 0.04)))
        font_scale = max(0.6, min(1.1, canvas_width / 1400))
        cv2.putText(
            target,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            max(4, thickness),
            cv2.LINE_AA,
        )
        cv2.putText(
            target,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    annotate(canvas)
    clean_written = bool(cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88]))
    if overlay_output_path is None:
        return clean_written
    annotate(overlay_canvas)
    overlay_written = bool(
        cv2.imwrite(str(overlay_output_path), overlay_canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    )
    return clean_written and overlay_written


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
        cpu = psutil.cpu_percent(interval=0.1)
        if memory_mb < self.settings.minimum_free_memory_mb:
            return False, f"free memory below {self.settings.minimum_free_memory_mb} MB"
        if cpu > self.settings.maximum_system_load_percent:
            return False, f"system CPU load above {self.settings.maximum_system_load_percent}%"
        return True, "resources available"

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
        edit = propose_curve_point(
            base_curve_data,
            measurement.plant_age_days,
            radius_mm_to_diameter_mm(measurement.recommended_protection_radius_mm),
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
        aggregate = Database._consolidate_measurement_rows(
            [item.model_dump(mode="json") for item in measurements]
        )
        partial_views_require_fusion = (
            fusion_settings.enabled
            and fusion_settings.automatic_requires_reliable_fusion
            and any(
                item.visible_fraction < fusion_settings.activation_visible_fraction
                for item in measurements
            )
        )
        if partial_views_require_fusion and not bool(aggregate.get("fusion_reliable")):
            return
        if aggregate.get("fused_canopy") and not bool(aggregate.get("fusion_reliable")):
            return
        latest = max(measurements, key=lambda item: item.image_timestamp)
        candidate = latest.model_copy(
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
            self.current["zone_blocked_weeds"] = 0
            self.current["zone_blocked_radius"] = 0
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
                weed_review_path = artifacts / f"{job_id}-{response.image_id}-weed-review.jpg"
                source_image_path = artifacts / f"{job_id}-{response.image_id}-photo.jpg"
                source_image_path.write_bytes(image_bytes)

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
                    # Every inventory plant participates in ownership, even
                    # when the caller requested measurements for a subset.
                    for plant in inventory.plants
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
                        weed_settings,
                    ),
                    timeout=60,
                )
                del image_bytes, previous_masks
                decided = []
                plants_by_id = {plant.id: plant for plant in wanted}
                for item in result.measurements:
                    if item.plant_id not in plants_by_id:
                        continue
                    plant = plants_by_id[item.plant_id]
                    item = item.model_copy(
                        update={
                            "recorded_center_x": plant.x,
                            "recorded_center_y": plant.y,
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
                matched_known_weed_ids: set[int] = set()
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
                    known_weed = min(
                        (
                            existing
                            for existing in inventory.weeds
                            if math.hypot(existing.x - weed_x, existing.y - weed_y)
                            <= max(duplicate_distance, existing.radius)
                        ),
                        key=lambda existing: math.hypot(existing.x - weed_x, existing.y - weed_y),
                        default=None,
                    )
                    if known_weed is not None:
                        matched_known_weed_ids.add(known_weed.id)
                        target_radius = max(float(known_weed.radius), float(weed.radius_mm))
                        status = "matched"
                        if (
                            weed_settings
                            and weed_settings.automatic_radius_adjustment
                            and target_radius > float(known_weed.radius) + 0.5
                            and weed.confidence >= weed_settings.radius_adjustment_confidence
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
                            radius_mm=target_radius,
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
                    if self.db.has_terminal_weed_detection_near(
                        entry_id, weed_x, weed_y, duplicate_distance
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
                        not weed_settings.visual_verifier_required_for_automatic
                        or (
                            weed.verifier_confidence is not None
                            and weed.verifier_confidence
                            >= weed_settings.visual_verifier_minimum_confidence
                        )
                    )
                    if (
                        weed_settings
                        and weed_settings.automatic_creation
                        and weed.confidence >= weed_settings.automatic_creation_confidence
                        and observations >= automatic_observations
                        and verifier_allows_automatic
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
                        prior_track = self.db.weed_track(entry_id, known_weed.id)
                        absent_observations = (
                            int((prior_track or {}).get("absent_observations") or 0) + 1
                        )
                        absence_confidence = min(0.95, 0.58 + 0.12 * absent_observations)
                        track_status = "removal_recommended"
                        if (
                            weed_settings.automatic_removal
                            and absent_observations >= weed_settings.removal_min_consecutive_absent
                            and absence_confidence >= weed_settings.removal_confidence
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
                        )
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
                decided = persisted
                self.db.save_measurements(decided)
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
            for plant_id, plant_measurements in measurements_by_plant.items():
                fused = fuse_canopy_masks(plant_measurements, fusion_settings)
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
                        and fused.corroborated_fraction
                        >= fusion_settings.minimum_corroborated_fraction
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
                if build_plant_composite(
                    plant_measurements,
                    composite_path,
                    composite_overlay_path,
                ):
                    self.db.set_composite_path(
                        [str(item.measurement_id) for item in plant_measurements],
                        str(composite_path),
                        str(composite_overlay_path),
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
        return {"accepted": True, **result}


def decode_previous_mask(path: str | None) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)
