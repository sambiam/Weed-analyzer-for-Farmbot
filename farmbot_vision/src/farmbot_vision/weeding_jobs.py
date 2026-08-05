"""End-to-end adaptive weeding orchestration for the Vision web app."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from .database import Database
from .home_assistant import HomeAssistantClient
from .jobs import JobManager
from .models import (
    InventoryRequest,
    RemoveWeedRequest,
    SoilPoint,
    WeedingRunRequest,
    WeedingTarget,
)
from .soil_jobs import SoilJobManager
from .weeding import (
    PLANT_MARGIN_MM,
    CutPath,
    confirmed_weeds,
    estimate_soil_height,
    exclusion_zone_obstacles,
    nearest_neighbour_order,
    plan_cut_path,
    protected_tall_plants,
    recent_soil_samples,
    route_cut_path,
)
from .zones import ZoneKind, ZoneStore

LOGGER = logging.getLogger(__name__)
CLEAR_SOIL_RADIUS_MM = 75.0
MAX_NEW_SOIL_DELTA_MM = 40.0


class WeedingJobManager:
    """Runs one user-confirmed batch, including soil and photo verification."""

    def __init__(
        self,
        database: Database,
        client: HomeAssistantClient,
        analysis_jobs: JobManager,
        soil_jobs: SoilJobManager,
        zone_store: ZoneStore,
    ):
        self.db = database
        self.client = client
        self.analysis_jobs = analysis_jobs
        self.soil_jobs = soil_jobs
        self.zone_store = zone_store
        self.task: asyncio.Task | None = None
        self.stop_requested = False
        self.current: dict = {"status": "idle", "message": "Not run"}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, *, entry_id: str, weed_ids: list[int], options: dict) -> str:
        if self.running:
            raise ValueError("a weeding job is already running")
        if self.analysis_jobs.lock.locked() or self.soil_jobs.running:
            raise ValueError("analysis or soil measurement is already running")
        job_id = str(uuid4())
        self.stop_requested = False
        self.current = {
            "id": job_id,
            "status": "queued",
            "message": "Loading current plants, weeds, and soil heights",
            "weeds_total": len(weed_ids),
            "weeds_attempted": 0,
            "weeds_removed": 0,
            "weeds_remaining": 0,
            "weeds_skipped": 0,
            "results": [],
        }
        self.task = asyncio.create_task(
            self._run(entry_id=entry_id, weed_ids=weed_ids, options=options),
            name=f"vision-weeding-{job_id}",
        )
        return job_id

    def request_stop(self) -> None:
        self.stop_requested = True
        self.current["message"] = "Stopping before the next stage"

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    @staticmethod
    def _clear_soil_site(weed, soil_inventory, garden) -> tuple[SoilPoint, float, float] | None:
        """Find clear soil near a weed and less than 200 mm from an anchor point."""
        x_bounds = soil_inventory.motion.axis_bounds.get("x")
        y_bounds = soil_inventory.motion.axis_bounds.get("y")
        if x_bounds is None or y_bounds is None:
            return None
        obstacles = [
            (plant.x, plant.y, plant.radius + CLEAR_SOIL_RADIUS_MM) for plant in garden.plants
        ]
        obstacles.extend(
            (item.x, item.y, max(item.radius, 10) + CLEAR_SOIL_RADIUS_MM) for item in garden.weeds
        )
        candidates = []
        for radius in range(75, 326, 25):
            for degrees in range(0, 360, 15):
                angle = math.radians(degrees)
                x = weed.x + radius * math.cos(angle)
                y = weed.y + radius * math.sin(angle)
                if not (x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]):
                    continue
                if any(math.hypot(x - ox, y - oy) < exclusion for ox, oy, exclusion in obstacles):
                    continue
                anchor = min(
                    soil_inventory.points,
                    key=lambda point: math.hypot(point.x - x, point.y - y),
                    default=None,
                )
                if anchor is None or math.hypot(anchor.x - x, anchor.y - y) >= 200:
                    continue
                clearance = min(
                    (math.hypot(x - ox, y - oy) - exclusion for ox, oy, exclusion in obstacles),
                    default=10_000,
                )
                candidates.append((radius, -clearance, anchor, x, y))
        if not candidates:
            return None
        _, _, anchor, x, y = min(candidates, key=lambda item: (item[0], item[1]))
        return anchor, x, y

    async def _measure_missing_soil(self, entry_id: str, weed, soil_inventory, garden) -> bool:
        calibration = self.db.active_soil_calibration(entry_id)
        if calibration is None:
            self.current["message"] = (
                "A weed needs a new soil height, but soil calibration is missing"
            )
            return False
        site = self._clear_soil_site(weed, soil_inventory, garden)
        if site is None:
            self.current["message"] = (
                f"No clear soil measurement site was found near weed {weed.id}"
            )
            return False
        anchor, x, y = site
        started_at = datetime.now(UTC)
        self.current["message"] = f"Measuring clear soil near weed {weed.id}"
        self.soil_jobs.start_measurements(
            config_entry_id=entry_id,
            point_ids=[],
            custom_point_id=anchor.id,
            custom_x=x,
            custom_y=y,
            capture_z=calibration.capture_z,
            baseline_mm=calibration.baseline_mm,
        )
        if self.soil_jobs.task is not None:
            await self.soil_jobs.task
        candidates = [
            row
            for row in self.db.recent_soil_measurements(entry_id, 20)
            if datetime.fromisoformat(str(row["created_at"])).astimezone(UTC) >= started_at
            and row.get("status") == "valid"
            and row.get("proposed_z_mm") is not None
        ]
        if not candidates:
            return False
        result = candidates[0]
        delta = abs(float(result["proposed_z_mm"]) - float(result["old_z_mm"]))
        if delta > MAX_NEW_SOIL_DELTA_MM:
            reason = (
                f"new soil height differed from nearby history by {delta:.1f} mm "
                f"(limit {MAX_NEW_SOIL_DELTA_MM:.0f} mm)"
            )
            self.db.update_soil_measurement_status(result["measurement_id"], "rejected", reason)
            self.current["message"] = reason
            return False
        return True

    async def _wait_weeding(self, entry_id: str, run_id: str):
        for _ in range(900):
            status = await self.client.weeding_status(entry_id, run_id)
            self.current["message"] = status.message
            self.current["weeds_attempted"] = status.weeds_completed or 0
            if status.status in {"complete", "failed", "rejected"}:
                return status
            await asyncio.sleep(2)
        raise RuntimeError("adaptive weeding status timed out")

    async def _verify(self, entry_id: str, paths: list[CutPath], capture_z: float) -> None:
        if not paths or self.stop_requested:
            return
        self.current.update(
            status="verifying", message="Photographing mowed weeds with the light on"
        )
        result = await self.client.start_grid_repair(
            entry_id,
            [
                {
                    "x": (path.start_x + path.end_x) / 2,
                    "y": (path.start_y + path.end_y) / 2,
                    "z": capture_z,
                    "index": path.weed_id,
                }
                for path in paths
            ],
        )
        repair_id = result.get("repair_id")
        if not repair_id:
            raise RuntimeError(str(result.get("message") or "verification scan was rejected"))
        for _ in range(900):
            capture = await self.client.grid_repair_status(entry_id, str(repair_id))
            self.current["message"] = str(capture.get("message") or "Capturing verification images")
            if capture.get("status") in {"complete", "failed"}:
                break
            await asyncio.sleep(2)
        else:
            raise RuntimeError("verification photo scan timed out")
        frames = capture.get("frames") or []
        image_by_weed = {
            int(frame["target_index"]): int(frame["image_id"])
            for frame in frames
            if frame.get("target_index") is not None and frame.get("image_id") is not None
        }
        if image_by_weed:
            self.current["message"] = "Checking only the mowed weeds in verification photos"
            await self.analysis_jobs.run(
                entry_id=entry_id,
                image_ids=list(image_by_weed.values()),
                trigger="weeding_verification",
                queue_if_busy=True,
            )
        for path in paths:
            image_id = image_by_weed.get(path.weed_id)
            observation = (
                self.db.known_weed_observation(entry_id, path.weed_id, image_id)
                if image_id is not None
                else None
            )
            if observation and observation["status"] == "absent":
                response = await self.client.remove_weed(
                    RemoveWeedRequest(
                        config_entry_id=entry_id,
                        weed_id=path.weed_id,
                        confidence=float(observation["confidence"]),
                        apply=True,
                        human_approved=True,
                    )
                )
                removed = response.get("status") == "applied"
                self.current["weeds_removed"] += int(removed)
                self.current["results"].append(
                    {"weed_id": path.weed_id, "verification": "absent", "removed": removed}
                )
            else:
                self.current["weeds_remaining"] += 1
                self.current["results"].append(
                    {
                        "weed_id": path.weed_id,
                        "verification": observation["status"] if observation else "not captured",
                        "removed": False,
                    }
                )

    def _complete_all_skipped(self) -> None:
        """Finish normally when planning found no safely mowable weed."""
        skipped = int(self.current.get("weeds_skipped", 0))
        self.current.update(
            status="complete",
            message=f"No weeds were mown; {skipped} weed(s) were safely skipped",
            completed_at=datetime.now(UTC).isoformat(),
        )

    async def _run(self, *, entry_id: str, weed_ids: list[int], options: dict) -> None:
        try:
            self.current.update(
                status="planning", message="Planning soil heights and plant-safe paths"
            )
            garden, soil_inventory = await asyncio.gather(
                self.client.inventory(
                    InventoryRequest(config_entry_id=entry_id, image_lookback_hours=720)
                ),
                self.client.soil_points(entry_id),
            )
            selected_ids = set(weed_ids)
            selected = [weed for weed in confirmed_weeds(garden.weeds) if weed.id in selected_ids]
            if not selected:
                raise RuntimeError("none of the selected weeds still exists on FarmBot")
            x_bounds = soil_inventory.motion.axis_bounds.get("x")
            y_bounds = soil_inventory.motion.axis_bounds.get("y")
            z_bounds = soil_inventory.motion.axis_bounds.get("z")
            if x_bounds is None or y_bounds is None or z_bounds is None:
                raise RuntimeError("FarmBot axis bounds are unavailable")
            paths: list[CutPath] = []
            exclusion_zones = [
                zone
                for zone in self.zone_store.zones()
                if zone.enabled and zone.kind is ZoneKind.EXCLUSION
            ]
            check_soil = bool(options.get("check_soil_heights", False))
            max_soil_age_days = float(options.get("soil_height_max_age_days", 30))
            cut_after_measurement_failure = bool(
                options.get("attempt_cut_if_soil_measurement_fails", False)
            )
            for weed in selected:
                if self.stop_requested:
                    break
                containing_zone = next(
                    (zone for zone in exclusion_zones if zone.contains_point(weed.x, weed.y)),
                    None,
                )
                if containing_zone is not None:
                    self.current["weeds_skipped"] += 1
                    self.current["results"].append(
                        {
                            "weed_id": weed.id,
                            "status": "skipped",
                            "reason": (f'weed is inside exclusion zone "{containing_zone.name}"'),
                        }
                    )
                    continue
                measurements = self.db.recent_soil_measurements(entry_id, 500)
                all_samples = recent_soil_samples(
                    soil_inventory.points, measurements, max_age_days=None
                )
                samples = recent_soil_samples(
                    soil_inventory.points,
                    measurements,
                    max_age_days=max_soil_age_days if check_soil else None,
                )
                estimate = estimate_soil_height(weed.x, weed.y, samples)
                fallback_estimate = estimate_soil_height(weed.x, weed.y, all_samples)
                if estimate is None and check_soil:
                    try:
                        measured = await self._measure_missing_soil(
                            entry_id, weed, soil_inventory, garden
                        )
                    except Exception as err:  # continue when the user explicitly permits it
                        LOGGER.warning("Soil measurement near weed %s failed: %s", weed.id, err)
                        measured = False
                    if measured:
                        samples = recent_soil_samples(
                            soil_inventory.points,
                            self.db.recent_soil_measurements(entry_id, 500),
                            max_age_days=max_soil_age_days,
                        )
                        estimate = estimate_soil_height(weed.x, weed.y, samples)
                    if estimate is None and cut_after_measurement_failure:
                        estimate = fallback_estimate
                if estimate is None:
                    self.current["weeds_skipped"] += 1
                    self.current["results"].append(
                        {
                            "weed_id": weed.id,
                            "status": "skipped",
                            "reason": (
                                "soil height check failed and no older nearby height is available"
                                if check_soil
                                else "no nearby soil height is available"
                            ),
                        }
                    )
                    continue
                try:
                    paths.append(
                        plan_cut_path(
                            weed,
                            garden.plants,
                            estimate,
                            x_bounds=x_bounds,
                            y_bounds=y_bounds,
                            plant_margin_mm=25.0,
                            exclusion_zones=exclusion_zones,
                        )
                    )
                except ValueError as err:
                    self.current["weeds_skipped"] += 1
                    self.current["results"].append(
                        {"weed_id": weed.id, "status": "skipped", "reason": str(err)}
                    )
            if not paths and not self.stop_requested:
                self._complete_all_skipped()
                return
            if self.stop_requested:
                raise RuntimeError("no weed has both a safe path and a trustworthy soil height")
            position = soil_inventory.motion.position
            ordered = nearest_neighbour_order(
                paths, float(position.get("x") or 0), float(position.get("y") or 0)
            )
            safe_z = z_bounds[1] if soil_inventory.motion.z_direction == -1 else z_bounds[0]
            transit_z = max(float(safe_z), -100.0)
            if not z_bounds[0] <= transit_z <= z_bounds[1]:
                raise RuntimeError(
                    "FarmBot cannot reach the required exclusion-zone transit height of Z -100"
                )
            protected = protected_tall_plants(
                garden.plants,
                enabled=bool(options.get("avoid_tall_plants", True)),
                minimum_height_mm=float(options.get("tall_plant_height_mm", 300)),
            )
            zone_obstacles = exclusion_zone_obstacles(exclusion_zones)
            current_xy = (float(position.get("x") or 0), float(position.get("y") or 0))
            if options.get("manage_tool"):
                slot_x = float(options.get("tool_slot_x", 4.2))
                slot_y = float(options.get("tool_slot_y", 576.8))
                current_xy = {
                    1: (slot_x + 100, slot_y),
                    2: (slot_x - 100, slot_y),
                    3: (slot_x, slot_y + 100),
                    4: (slot_x, slot_y - 100),
                }[int(options.get("tool_pullout_direction", 1))]
            targets: list[WeedingTarget] = []
            routed: list[CutPath] = []
            for path in ordered:
                try:
                    path, waypoints = route_cut_path(
                        current_xy,
                        path,
                        [*protected, *zone_obstacles],
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        endpoint_margin_mm=PLANT_MARGIN_MM,
                    )
                except ValueError:
                    # Exclusion zones are preferably avoided in X/Y. If their
                    # conservative circular envelopes leave no route, crossing
                    # is permitted only at the separately enforced transit Z.
                    try:
                        path, waypoints = route_cut_path(
                            current_xy,
                            path,
                            protected,
                            x_bounds=x_bounds,
                            y_bounds=y_bounds,
                            endpoint_margin_mm=PLANT_MARGIN_MM,
                        )
                    except ValueError as err:
                        self.current["weeds_skipped"] += 1
                        self.current["results"].append(
                            {"weed_id": path.weed_id, "status": "skipped", "reason": str(err)}
                        )
                        continue
                targets.append(
                    WeedingTarget(
                        weed_id=path.weed_id,
                        transit_start={"x": current_xy[0], "y": current_xy[1]},
                        start={"x": path.start_x, "y": path.start_y},
                        end={"x": path.end_x, "y": path.end_y},
                        soil_z=path.soil_z,
                        travel_z=transit_z,
                        approach_waypoints=waypoints,
                    )
                )
                routed.append(path)
                current_xy = (path.end_x, path.end_y)
            if not targets:
                self._complete_all_skipped()
                return
            ordered = routed
            self.current.update(status="weeding", paths=[asdict(path) for path in ordered])
            integration_options = {
                key: value
                for key, value in options.items()
                if key
                not in {
                    "check_soil_heights",
                    "soil_height_max_age_days",
                    "attempt_cut_if_soil_measurement_fails",
                }
            }
            started = await self.client.start_weeding(
                WeedingRunRequest(config_entry_id=entry_id, weeds=targets, **integration_options)
            )
            if started.status != "queued" or started.run_id is None:
                raise RuntimeError(started.message or "adaptive weeding was rejected")
            run = await self._wait_weeding(entry_id, str(started.run_id))
            if run.status == "failed" and not (run.weeds_completed or 0):
                raise RuntimeError(run.message)
            attempted_ids = {
                int(item["weed_id"]) for item in run.results if item.get("status") == "attempted"
            }
            attempted_paths = [path for path in ordered if path.weed_id in attempted_ids]
            image_z_values = [float(image.meta.z) for image in garden.images]
            capture_z = (
                sorted(image_z_values)[len(image_z_values) // 2] if image_z_values else safe_z
            )
            await self._verify(entry_id, attempted_paths, capture_z)
            self.current.update(
                status="complete",
                message=(
                    f"Weeding complete: {self.current['weeds_removed']} removed after photo "
                    f"verification, {self.current['weeds_remaining']} retained, "
                    f"{self.current['weeds_skipped']} safely skipped"
                ),
                completed_at=datetime.now(UTC).isoformat(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.warning("Weeding job failed: %s", err)
            self.current.update(
                status="failed",
                message=str(err)[:240] or "weeding job failed",
                completed_at=datetime.now(UTC).isoformat(),
            )
