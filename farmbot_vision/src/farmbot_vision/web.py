from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Annotated
from urllib.parse import quote
from uuid import UUID, uuid4

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.types import ASGIApp, Receive, Scope, Send

from . import (
    ALGORITHM_VERSION,
    CONTRACT_VERSION,
    MINIMUM_INTEGRATION_VERSION,
    __version__,
)
from .calibration import from_farmbot_calibration
from .calibration_store import CalibrationStore, FarmbotCalibrationInput
from .canopy_settings import CanopyFusionSettings, CanopyFusionSettingsStore
from .curve_edit import propose_curve_point
from .curves import fit_monotonic_curve
from .database import Database
from .grid_repair import (
    COORDINATE_TOLERANCE_MM,
    GridRepairSettings,
    GridRepairSettingsStore,
    GridRun,
    RepairCaptures,
    RepairCaptureStore,
    RepairTarget,
    detect_latest_grid_run,
    looks_like_gantry_photo,
    same_grid_run,
    target_payload,
)
from .home_assistant import HomeAssistantClient, HomeAssistantError, StaleRadiusError
from .jobs import JobManager
from .models import (
    ApplyPlantCenterRequest,
    ApplyRadiusRequest,
    ApplyRemovalRequest,
    ApplySoilHeightRequest,
    Bot,
    Calibration,
    CreateWeedRequest,
    InventoryRequest,
    Measurement,
    OperatingMode,
    OriginLocation,
    QueueImagesRequest,
    UpsertCurveRequest,
    VisionImageRequest,
    WeedBulkAcceptRequest,
)
from .photo_grid import (
    PHOTO_GRID_CONTINUOUS_CAPABILITY,
    PhotoGridFrame,
    PhotoGridQualityRepair,
    PhotoGridRecord,
    PhotoGridStore,
    PhotoGridTarget,
    match_verified_frames,
    photo_grid_chunk_size,
    plan_photo_grid,
    plan_targeted_plant_captures,
)
from .photo_quality import PhotoQuality, best_unobscured_photo, inspect_photo_quality
from .settings import Settings
from .soil_jobs import SoilJobManager
from .vision import garden_to_pixel
from .weed_settings import WeedSettings, WeedSettingsStore
from .weed_verifier import (
    ALL_LABELS,
    FEATURE_NAMES,
    LABEL_DESCRIPTIONS,
    WeedVisualVerifier,
)
from .zones import (
    Zone,
    ZoneAspect,
    ZoneKind,
    ZoneShape,
    ZoneStore,
    ZoneVerdict,
    evaluate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
settings = Settings.load()
database = Database(settings.data_dir / "farmbot_vision.db")
calibration_store = CalibrationStore(settings.data_dir / "farmbot_calibration.json")
client = HomeAssistantClient()
weed_settings_store = WeedSettingsStore(settings.data_dir / "weed_settings.json")
canopy_fusion_settings_store = CanopyFusionSettingsStore(
    settings.data_dir / "canopy_fusion_settings.json"
)
grid_repair_settings_store = GridRepairSettingsStore(
    settings.data_dir / "grid_repair_settings.json"
)
repair_capture_store = RepairCaptureStore(settings.data_dir / "grid_repair_captures.json")
photo_grid_store = PhotoGridStore(settings.data_dir / "photo_grid_latest.json")
# A model shipped inside the image gives a fresh install useful filtering
# before its owner has labelled anything. It is only consulted when no locally
# trained model exists (see WeedVisualVerifier.reload).
weed_verifier = WeedVisualVerifier(
    settings.data_dir / "weed_visual_model.json",
    bundled_path=Path(__file__).with_name("bundled_weed_model.json"),
)
zone_store = ZoneStore(settings.data_dir / "zones.json")
jobs = JobManager(settings, database, client, weed_settings_store, zone_store)
soil_jobs = SoilJobManager(database, client, settings.data_dir, jobs.lock, zone_store)
grid_repair_state: dict[str, object] = {
    "run": None,
    "checked_at": None,
    "error": "",
    "message": "",
    "gantry_image_ids": (),
    "status": "idle",
    "repair_id": "",
}
grid_repair_task: asyncio.Task | None = None
photo_grid_task: asyncio.Task | None = None
GRID_REPAIR_STATUS_POLL_SECONDS = 5
GRID_REPAIR_BUSY_RETRY_SECONDS = 30
# A photo-grid batch start can be rejected as "busy" while the previous
# batch's task is still unwinding (returning the bot to its original
# position) or while FarmBot's own busy flag hasn't cleared yet. That is a
# transient race, not a real failure, so retry with backoff instead of
# discarding the whole batch's targets.
PHOTO_GRID_BATCH_START_MAX_ATTEMPTS = 6
PHOTO_GRID_BATCH_START_RETRY_BASE_SECONDS = 5
PHOTO_GRID_BATCH_START_RETRY_MAX_SECONDS = 30
# Bounded multi-pass retry for coordinates that came back unverified. A pass
# that verifies zero new frames means another identical pass would just move
# the bot pointlessly, so the worker stops early rather than always running
# every pass.
PHOTO_GRID_WORKER_MAX_PASSES = 3
# Cap how many missing coordinates are ever written to the log or the
# terminal status message so a large failure can't flood either.
PHOTO_GRID_MISSING_LOG_LIMIT = 20
PHOTO_GRID_MESSAGE_EXAMPLE_LIMIT = 5
# Move far enough for the camera to look around a close leaf while retaining
# generous overlap with the original cell.
PHOTO_GRID_LEAF_OFFSET_FRACTION = 0.20
# A cell whose fresh photo still shows the gantry will photograph the gantry
# again; give it one retry, then move on rather than loop on it forever.
MAX_REPAIR_ATTEMPTS_PER_CELL = 2
TRAINING_LABELS = (
    "weed",
    "crop",
    "fallen_leaf",
    "mushroom",
    "moss",
    "soil",
    "hardware",
    # Keep old grouped labels selectable for samples saved by older versions.
    "mulch_soil",
    "fungus_moss",
    "hardware_other",
)


def _verifier_label_guess(features: dict) -> list[dict[str, object]]:
    """What the verifier thinks the object actually is, best guess first.

    Purely descriptive: the accept/reject decision stays with the binary score.
    It exists because a borderline detection is far easier to review when the
    model can say "this looks like moss".
    """
    if not isinstance(features, dict):
        return []
    return [
        {"label": LABEL_DESCRIPTIONS.get(label, label.replace("_", " ")), "probability": value}
        for label, value in weed_verifier.explain(features)[:3]
    ]


async def inspect_photo_grid(*, force: bool = False) -> GridRun | None:
    """Inspect and cache the latest likely grid, including gantry content."""
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        grid_repair_state.update(run=None, error="Select a FarmBot first")
        return None
    now = datetime.now(UTC)
    checked_at = grid_repair_state.get("checked_at")
    if not force and isinstance(checked_at, datetime) and now - checked_at < timedelta(minutes=5):
        cached = grid_repair_state.get("run")
        return cached if isinstance(cached, GridRun) else None
    try:
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=entry_id,
                image_lookback_hours=min(720, max(72, settings.image_lookback_hours)),
            )
        )
        captures = repair_capture_store.load()
        candidate = detect_latest_grid_run(inventory.images, None, captures)
        # Once FarmBot runs a fresh photo grid, the previous run's repair
        # photos say nothing about it; drop them so they can never fill a new
        # run's genuinely missing cells with stale images.
        if (
            candidate is not None
            and captures.run_started_at is not None
            and not same_grid_run(candidate.started_at, captures.run_started_at)
        ):
            captures = RepairCaptures()
            repair_capture_store.save(captures)
            candidate = detect_latest_grid_run(inventory.images)
        gantry_ids: set[int] = set()
        if candidate is not None:
            semaphore = asyncio.Semaphore(4)

            async def classify(image) -> None:
                if image.meta.name and "gantry" in image.meta.name.casefold():
                    gantry_ids.add(image.id)
                    return
                async with semaphore:
                    processed = await client.image(
                        VisionImageRequest(
                            config_entry_id=entry_id,
                            image_id=image.id,
                            max_width=320,
                            max_height=240,
                        ),
                        settings.max_image_payload_bytes,
                    )
                jpeg = base64.b64decode(processed.image_base64, validate=True)
                if looks_like_gantry_photo(jpeg, image.meta.name):
                    gantry_ids.add(image.id)

            results = await asyncio.gather(
                *(classify(image) for image in candidate.images),
                return_exceptions=True,
            )
            failed = sum(isinstance(result, Exception) for result in results)
            if failed:
                LOGGER.warning(
                    "Could not inspect %d photo-grid image(s) for gantry content", failed
                )
            candidate = detect_latest_grid_run(inventory.images, gantry_ids, captures)
        grid_repair_state.update(
            run=candidate,
            checked_at=now,
            error="",
            gantry_image_ids=tuple(sorted(gantry_ids)),
        )
        return candidate
    except HomeAssistantError as exc:
        grid_repair_state.update(
            run=None,
            checked_at=now,
            error=str(exc),
            gantry_image_ids=(),
        )
        return None


async def _require_grid_repair_capability(*, require_lighting: bool = False) -> Bot:
    bots = (await client.list_bots()).bots
    bot = next(
        (item for item in bots if item.config_entry_id == settings.selected_config_entry_id),
        None,
    )
    required = (
        "illuminated_photo_grid_capture"
        if require_lighting
        else "position_verified_photo_grid_repair"
    )
    if bot is None or not bot.supports(required):
        version = bot.integration_version if bot is not None else None
        loaded = f" (loaded version {version})" if version else ""
        raise HomeAssistantError(
            f"Photo-grid capture requires FarmBot integration V{MINIMUM_INTEGRATION_VERSION}"
            f"{loaded}. Install/update it and restart Home Assistant."
        )
    return bot


def _ordered_repair_targets(run: GridRun):
    """Prioritize empty cells, then keep travel deterministic."""
    return sorted(
        run.targets,
        key=lambda target: (target.reason != "missing", target.x, target.y),
    )


def _cell_attempts(cells: list[tuple[float, float]], target: RepairTarget) -> int:
    """Count how many recorded attempts refer to the same grid cell.

    Grid-axis positions are averages of the photos taken at them, so they
    shift slightly whenever the images behind a cell change; cells are matched
    by proximity rather than equality for that reason.
    """
    return sum(
        1 for x, y in cells if math.hypot(target.x - x, target.y - y) <= COORDINATE_TOLERANCE_MM
    )


def _next_repair_target(
    run: GridRun,
    skipped: list[tuple[float, float]],
) -> RepairTarget | None:
    """Return the next cell not already exhausted during this repair session."""
    return next(
        (target for target in _ordered_repair_targets(run) if not _cell_attempts(skipped, target)),
        None,
    )


def _unrepaired_message(
    skipped: list[tuple[float, float]],
    exhausted: list[tuple[float, float]],
) -> str:
    """Explain which cells the session gave up on, and why."""
    reasons = []
    if skipped:
        reasons.append(
            f"{len(skipped)} cell{'s' if len(skipped) != 1 else ''} could not be captured "
            "after six camera attempts"
        )
    if exhausted:
        reasons.append(
            f"{len(exhausted)} cell{'s' if len(exhausted) != 1 else ''} still had no usable "
            f"photo after {MAX_REPAIR_ATTEMPTS_PER_CELL} fresh captures"
        )
    return "Photo-grid repair finished the remaining cells, but " + " and ".join(reasons)


def _record_repair_captures(run: GridRun, image_ids: list[int]) -> None:
    """Credit newly captured photos to the grid run they were taken for."""
    if not image_ids:
        return
    captures = repair_capture_store.load()
    if not same_grid_run(run.started_at, captures.run_started_at):
        captures = RepairCaptures()
    captures.run_started_at = run.started_at
    captures.image_ids = sorted(set(captures.image_ids) | set(image_ids))
    repair_capture_store.save(captures)


def _repaired_image_ids(result: dict[str, object]) -> list[int]:
    """Image IDs the integration verified at the repaired coordinates."""
    frames = result.get("frames")
    if not isinstance(frames, list):
        return []
    image_ids = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            image_ids.append(int(frame["image_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return image_ids


async def _delete_replaced_gantry_image(image_id: int) -> None:
    """Retire a gantry-obscured photo now that a usable one has replaced it.

    Deletion is best effort: an older companion integration does not offer it,
    and a failure here must not abort a repair that has already succeeded.
    """
    try:
        bots = (await client.list_bots()).bots
        bot = next(
            (item for item in bots if item.config_entry_id == settings.selected_config_entry_id),
            None,
        )
        if bot is None or not bot.supports("vision_image_deletion"):
            LOGGER.info(
                "Replaced gantry photo %d was left in place: the loaded FarmBot "
                "integration does not support image deletion",
                image_id,
            )
            return
        result = await client.delete_image(settings.selected_config_entry_id, image_id)
        if str(result.get("status") or "") != "deleted":
            LOGGER.warning(
                "Could not delete replaced gantry photo %d: %s",
                image_id,
                result.get("message") or result.get("status"),
            )
            return
        LOGGER.info("Deleted gantry photo %d after a usable replacement was captured", image_id)
    except HomeAssistantError as exc:
        LOGGER.warning("Could not delete replaced gantry photo %d: %s", image_id, exc)


async def _queue_one_repair_target(
    run: GridRun,
    *,
    skipped: list[tuple[float, float]] | None = None,
) -> tuple[dict[str, object], RepairTarget]:
    target = _next_repair_target(run, skipped or [])
    if target is None:
        raise HomeAssistantError("No untried photo-grid cell remains")
    payload = target_payload((target,))
    result = await client.start_grid_repair(settings.selected_config_entry_id, payload)
    return result, target


async def _photo_grid_repair_worker(
    *,
    session_id: str,
    run: GridRun,
    active_repair_id: str | None,
) -> None:
    """Repair one verified image at a time until the detected grid is whole."""
    # Cells FarmBot could not photograph at all.
    skipped: list[tuple[float, float]] = []
    # Cells that were photographed successfully but still read as unusable
    # (a genuinely gantry-obscured cell photographs the gantry every time).
    exhausted: list[tuple[float, float]] = []
    captured: list[tuple[float, float]] = []
    active_target = _next_repair_target(run, skipped + exhausted)
    try:
        while True:
            if active_repair_id:
                try:
                    result = await client.grid_repair_status(
                        settings.selected_config_entry_id, active_repair_id
                    )
                except HomeAssistantError as exc:
                    grid_repair_state.update(
                        status="queued",
                        message=f"Repair status check is waiting for Home Assistant: {exc}",
                    )
                    await asyncio.sleep(GRID_REPAIR_BUSY_RETRY_SECONDS)
                    continue
                status = str(result.get("status") or "")
                message = str(result.get("message") or status)
                grid_repair_state.update(status=status, message=message)
                if status in {"queued", "running", "waiting_images"}:
                    await asyncio.sleep(GRID_REPAIR_STATUS_POLL_SECONDS)
                    continue
                if status != "complete" and active_target is not None:
                    skipped.append((active_target.x, active_target.y))
                    grid_repair_state.update(
                        status="retrying",
                        message=(
                            f"{message}. This cell exhausted six camera attempts; "
                            "repair is moving to the next cell."
                        ),
                    )
                elif status == "complete" and active_target is not None:
                    # Credit the new photo to this run so the cell counts as
                    # filled from here on, and retire the frame it replaced.
                    _record_repair_captures(run, _repaired_image_ids(result))
                    captured.append((active_target.x, active_target.y))
                    if (
                        active_target.reason == "gantry"
                        and active_target.image_id is not None
                        and any(
                            image_id != active_target.image_id
                            for image_id in _repaired_image_ids(result)
                        )
                    ):
                        await _delete_replaced_gantry_image(active_target.image_id)
                    if _cell_attempts(captured, active_target) >= MAX_REPAIR_ATTEMPTS_PER_CELL:
                        exhausted.append((active_target.x, active_target.y))
                active_repair_id = None
                active_target = None
                run = await inspect_photo_grid(force=True)
                while run is None:
                    grid_repair_state.update(
                        status="queued",
                        message=(
                            "The completed repair is waiting for a fresh grid "
                            "inventory before the next cell is sent"
                        ),
                    )
                    await asyncio.sleep(GRID_REPAIR_BUSY_RETRY_SECONDS)
                    run = await inspect_photo_grid(force=True)
                if not run.targets:
                    grid_repair_state.update(
                        status="complete",
                        message="Photo-grid repair complete; every cell has a usable image",
                    )
                    return
                active_target = _next_repair_target(run, skipped + exhausted)
                if active_target is None:
                    grid_repair_state.update(
                        status="failed",
                        message=_unrepaired_message(skipped, exhausted),
                    )
                    return

            try:
                result, active_target = await _queue_one_repair_target(
                    run, skipped=skipped + exhausted
                )
            except HomeAssistantError as exc:
                grid_repair_state.update(
                    status="queued",
                    message=f"Repair is waiting for Home Assistant: {exc}",
                )
                await asyncio.sleep(GRID_REPAIR_BUSY_RETRY_SECONDS)
                continue
            status = str(result.get("status") or "")
            message = str(result.get("message") or status)
            repair_id = str(result.get("repair_id") or "")
            if status == "rejected":
                if "busy" not in message.casefold():
                    raise HomeAssistantError(message or "FarmBot rejected the repair cell")
                grid_repair_state.update(
                    status="queued",
                    message="FarmBot is busy; the next repair cell remains queued",
                )
                await asyncio.sleep(GRID_REPAIR_BUSY_RETRY_SECONDS)
                continue
            if not repair_id:
                raise HomeAssistantError(message or "FarmBot did not return a repair session ID")
            remaining = len(run.targets)
            active_repair_id = repair_id
            grid_repair_state.update(
                status="running",
                message=(
                    f"Repairing one verified cell at a time "
                    f"({remaining} cell{'s' if remaining != 1 else ''} remaining)"
                ),
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Photo-grid repair session %s stopped", session_id)
        grid_repair_state.update(
            status="failed",
            error=str(exc),
            message=f"Photo-grid repair stopped: {exc}",
        )


async def start_photo_grid_repair() -> dict[str, object]:
    global grid_repair_task
    if grid_repair_task is not None and not grid_repair_task.done():
        return {
            "status": str(grid_repair_state.get("status") or "queued"),
            "repair_id": str(grid_repair_state.get("repair_id") or ""),
            "message": str(
                grid_repair_state.get("message") or "Photo-grid repair is already queued"
            ),
        }
    await _require_grid_repair_capability()
    run = await inspect_photo_grid(force=True)
    if run is None:
        return {"status": "rejected", "message": "No recent photo-grid run was found"}
    if not run.targets:
        return {"status": "complete", "message": "The latest photo grid is complete"}

    # Queue only one target. The integration does not mark it complete until a
    # newly processed image exists at the requested coordinates. This both
    # avoids FarmBot's asynchronous take_photo error and naturally stays below
    # the integration's legacy twelve-target service limit.
    result, _ = await _queue_one_repair_target(run)
    status = str(result.get("status") or "")
    message = str(result.get("message") or status)
    active_repair_id = str(result.get("repair_id") or "") or None
    if status == "rejected" and "busy" not in message.casefold():
        grid_repair_state.update(status=status, message=message)
        return result

    session_id = str(uuid4())
    queued_message = (
        f"Photo-grid repair queued one verified image at a time "
        f"({len(run.targets)} cell{'s' if len(run.targets) != 1 else ''} remaining)"
    )
    grid_repair_state.update(
        status="queued",
        repair_id=session_id,
        message=queued_message,
        error="",
    )
    grid_repair_task = asyncio.create_task(
        _photo_grid_repair_worker(
            session_id=session_id,
            run=run,
            active_repair_id=active_repair_id,
        ),
        name=f"photo-grid-repair-{session_id}",
    )
    return {
        "status": "queued",
        "repair_id": session_id,
        "message": queued_message,
    }


def _merge_photo_grid_frames(
    record: PhotoGridRecord,
    targets: list[PhotoGridTarget],
    raw_frames: object,
) -> list[PhotoGridFrame]:
    """Validate and persist newly uploaded frames without duplicating targets."""
    verified = match_verified_frames(targets, raw_frames)
    by_target = {frame.target_index: frame for frame in record.frames}
    for frame in verified:
        by_target[frame.target_index] = frame
    record.frames = [by_target[index] for index in sorted(by_target)]
    photo_grid_store.save(record)
    return verified


def _coordinate_label(target: PhotoGridTarget) -> str:
    """Render a target's garden coordinates for logs and status messages."""
    return f"{target.x:g},{target.y:g}"


async def _start_photo_grid_batch(
    record: PhotoGridRecord,
    targets: list[PhotoGridTarget],
) -> dict[str, object]:
    """Start one batch, retrying only a transient busy rejection.

    The integration can reject a new batch with a "busy" message while the
    previous batch's task is still unwinding (returning the bot to its
    original position) or while FarmBot's own busy flag hasn't cleared. That
    is a transient race, not a real failure -- retrying with backoff avoids
    losing an entire chunk's targets to it. Any other rejection reason (or a
    response missing a repair ID) stays a hard, non-retried failure.
    """
    # `index` is only sent to an integration that advertises it: older service
    # schemas reject any unknown target key outright, which would turn every
    # batch into an HTTP 400 that captures nothing.
    payload = [
        {"x": target.x, "y": target.y, "z": target.z}
        | ({"index": target.index} if record.indexed_targets else {})
        for target in targets
    ]
    attempt = 0
    while True:
        attempt += 1
        started = await client.start_grid_repair(record.config_entry_id, payload)
        status = str(started.get("status") or "")
        message = str(started.get("message") or status)
        repair_id = str(started.get("repair_id") or "")
        if status != "rejected" and repair_id:
            return started
        is_busy = status == "rejected" and "busy" in message.casefold()
        if is_busy and attempt < PHOTO_GRID_BATCH_START_MAX_ATTEMPTS:
            delay = min(
                PHOTO_GRID_BATCH_START_RETRY_BASE_SECONDS * attempt,
                PHOTO_GRID_BATCH_START_RETRY_MAX_SECONDS,
            )
            LOGGER.info(
                "Photo grid %s batch start busy (attempt %d/%d): %s; retrying in %ds",
                record.session_id,
                attempt,
                PHOTO_GRID_BATCH_START_MAX_ATTEMPTS,
                message,
                delay,
            )
            await asyncio.sleep(delay)
            continue
        raise HomeAssistantError(message or "FarmBot rejected the photo-grid batch")


async def _capture_photo_grid_targets(
    record: PhotoGridRecord,
    targets: list[PhotoGridTarget],
) -> list[PhotoGridTarget]:
    """Capture one integration-sized batch and return any unverified targets."""
    try:
        started = await _start_photo_grid_batch(record, targets)
    except HomeAssistantError as exc:
        if len(targets) <= 1:
            raise
        # Home Assistant rejects an over-long `targets` list during service
        # schema validation, which surfaces here as a bare HTTP 400 with no
        # per-target detail and no capture at all. Halving the batch lets a
        # cap mismatch between add-on and integration versions degrade into
        # smaller working calls instead of stranding every coordinate in the
        # chunk -- the failure mode that cost a whole grid when the app's
        # chunk size was raised past the integration's limit.
        middle = len(targets) // 2
        LOGGER.warning(
            "Photo grid %s batch of %d targets was rejected (%s); retrying as %d + %d",
            record.session_id,
            len(targets),
            exc,
            middle,
            len(targets) - middle,
        )
        unverified: list[PhotoGridTarget] = []
        for half in (targets[:middle], targets[middle:]):
            try:
                unverified.extend(await _capture_photo_grid_targets(record, half))
            except HomeAssistantError as half_exc:
                LOGGER.warning(
                    "Photo grid %s split batch of %d targets failed: %s",
                    record.session_id,
                    len(half),
                    half_exc,
                )
                verified = {frame.target_index for frame in record.frames}
                unverified.extend(item for item in half if item.index not in verified)
        return unverified
    repair_id = str(started.get("repair_id") or "")

    while True:
        result = await client.grid_repair_status(record.config_entry_id, repair_id)
        status = str(result.get("status") or "")
        message = str(result.get("message") or status)
        _merge_photo_grid_frames(record, targets, result.get("frames"))
        complete_indexes = {frame.target_index for frame in record.frames}
        record.status = "running"
        record.message = (
            f"{len(complete_indexes)} of {len(record.targets)} photos verified · {message}"
        )
        photo_grid_store.save(record)
        if status in {"queued", "running", "waiting_images"}:
            await asyncio.sleep(GRID_REPAIR_STATUS_POLL_SECONDS)
            continue
        return [target for target in targets if target.index not in complete_indexes]


async def _capture_targeted_plant_photos(record: PhotoGridRecord) -> None:
    """Queue and complete deduplicated centred follow-ups through grid repair."""

    inventory = await client.inventory(
        InventoryRequest(
            config_entry_id=record.config_entry_id,
            image_lookback_hours=min(720, max(72, settings.image_lookback_hours)),
        )
    )
    planned, diagnostics = plan_targeted_plant_captures(
        record,
        inventory.plants,
        safety_margin_mm=settings.safety_margin_mm,
    )
    record.targeted_capture_diagnostics = diagnostics
    if not planned:
        photo_grid_store.save(record)
        return
    # Persist queued state before calling Home Assistant. A worker restart or a
    # second completion callback therefore cannot enqueue an equivalent move.
    record.targeted_captures.extend(planned)
    photo_grid_store.save(record)
    for capture in planned:
        target = PhotoGridTarget(
            index=len(record.targets) + record.targeted_captures.index(capture),
            row=0,
            column=0,
            x=capture.x,
            y=capture.y,
            z=capture.z,
        )
        try:
            started = await _start_photo_grid_batch(record, [target])
            capture.repair_id = str(started.get("repair_id") or "")
            capture.status = "running"
            photo_grid_store.save(record)
            while True:
                result = await client.grid_repair_status(record.config_entry_id, capture.repair_id)
                status = str(result.get("status") or "")
                if status in {"queued", "running", "waiting_images"}:
                    await asyncio.sleep(GRID_REPAIR_STATUS_POLL_SECONDS)
                    continue
                frames = result.get("frames")
                image_id = None
                if isinstance(frames, list):
                    for frame in frames:
                        if isinstance(frame, dict) and frame.get("image_id") is not None:
                            image_id = int(frame["image_id"])
                            break
                capture.image_id = image_id
                capture.completed_at = datetime.now(UTC)
                capture.status = "complete" if image_id is not None else "failed"
                capture.reason = str(result.get("message") or capture.reason)
                photo_grid_store.save(record)
                LOGGER.info(
                    "Targeted plant capture: %s",
                    json.dumps(
                        {
                            "grid_session_id": record.session_id,
                            "plant_id": capture.plant_id,
                            "crop_name": capture.crop_name,
                            "status": capture.status,
                            "image_id": capture.image_id,
                            "reason": capture.reason,
                        },
                        separators=(",", ":"),
                    ),
                )
                break
        except HomeAssistantError as exc:
            capture.completed_at = datetime.now(UTC)
            capture.status = "failed"
            capture.reason = str(exc)
            photo_grid_store.save(record)
            LOGGER.warning(
                "Targeted capture for plant %s failed safely: %s",
                capture.plant_id,
                exc,
            )


async def _quality_jpeg(record: PhotoGridRecord, image_id: int) -> bytes:
    response = await client.image(
        VisionImageRequest(
            config_entry_id=record.config_entry_id,
            image_id=image_id,
            max_width=640,
            max_height=640,
        ),
        settings.max_image_payload_bytes,
    )
    return base64.b64decode(response.image_base64, validate=True)


async def _delete_discarded_grid_image(
    record: PhotoGridRecord,
    image_id: int,
    *,
    reason: str,
) -> None:
    """Best-effort remote deletion after app-side analysis exclusion."""

    try:
        result = await client.delete_image(record.config_entry_id, image_id)
        if str(result.get("status") or "") != "deleted":
            LOGGER.warning(
                "Photo-grid image %d was excluded but remote deletion did not complete: %s",
                image_id,
                result.get("message") or result.get("status"),
            )
            return
        LOGGER.info("Deleted photo-grid image %d (%s)", image_id, reason)
    except HomeAssistantError as exc:
        LOGGER.warning(
            "Photo-grid image %d remains remotely stored but is excluded from analysis: %s",
            image_id,
            exc,
        )


def _exclude_grid_image(record: PhotoGridRecord, image_id: int) -> None:
    record.excluded_image_ids = sorted(set(record.excluded_image_ids) | {int(image_id)})
    jobs.queued_image_ids = [
        queued for queued in jobs.queued_image_ids if int(queued) != int(image_id)
    ]
    jobs.current["queue_length"] = len(jobs.queued_image_ids)


def _leaf_offset_targets(
    record: PhotoGridRecord,
    target: PhotoGridTarget,
) -> list[PhotoGridTarget]:
    """Plan four bounded views around one obstructed cell."""

    dx = max(20.0, record.footprint_width_mm * PHOTO_GRID_LEAF_OFFSET_FRACTION)
    dy = max(20.0, record.footprint_height_mm * PHOTO_GRID_LEAF_OFFSET_FRACTION)
    lower_x, upper_x = record.bed_bounds["x"]
    lower_y, upper_y = record.bed_bounds["y"]
    vectors = (
        (-dx, 0.0),
        (dx, 0.0),
        (0.0, -dy),
        (0.0, dy),
        (-dx, -dy),
        (dx, -dy),
        (-dx, dy),
        (dx, dy),
        (-dx / 2, 0.0),
        (dx / 2, 0.0),
        (0.0, -dy / 2),
        (0.0, dy / 2),
    )
    coordinates: list[tuple[float, float]] = []
    for offset_x, offset_y in vectors:
        x = round(min(upper_x, max(lower_x, target.x + offset_x)), 3)
        y = round(min(upper_y, max(lower_y, target.y + offset_y)), 3)
        if (x, y) == (target.x, target.y) or (x, y) in coordinates:
            continue
        coordinates.append((x, y))
        if len(coordinates) == 4:
            break
    base_index = len(record.targets) + len(record.quality_repairs) * 16
    return [
        PhotoGridTarget(
            index=base_index + index,
            row=target.row,
            column=target.column,
            x=x,
            y=y,
            z=target.z,
        )
        for index, (x, y) in enumerate(coordinates)
    ]


async def _capture_quality_targets(
    record: PhotoGridRecord,
    targets: list[PhotoGridTarget],
) -> list[PhotoGridFrame]:
    """Make one capture call and return its verified frames without retrying."""

    if not targets:
        return []
    started = await _start_photo_grid_batch(record, targets)
    repair_id = str(started.get("repair_id") or "")
    while True:
        result = await client.grid_repair_status(record.config_entry_id, repair_id)
        status = str(result.get("status") or "")
        if status in {"queued", "running", "waiting_images"}:
            await asyncio.sleep(GRID_REPAIR_STATUS_POLL_SECONDS)
            continue
        return match_verified_frames(targets, result.get("frames"))


async def _repair_washed_out_frame(
    record: PhotoGridRecord,
    frame: PhotoGridFrame,
    repair: PhotoGridQualityRepair,
) -> None:
    target = next(item for item in record.targets if item.index == frame.target_index)
    _exclude_grid_image(record, frame.image_id)
    record.frames = [item for item in record.frames if item.image_id != frame.image_id]
    photo_grid_store.save(record)
    await _delete_discarded_grid_image(record, frame.image_id, reason="washed out")

    replacements = await _capture_quality_targets(record, [target])
    repair.candidate_image_ids = [item.image_id for item in replacements]
    photo_grid_store.save(record)
    if not replacements:
        repair.status = "failed"
        repair.message = "The single same-coordinate retake produced no verified image"
        return
    replacement = replacements[0]
    quality = inspect_photo_quality(await _quality_jpeg(record, replacement.image_id))
    if quality.issue == "washed_out":
        _exclude_grid_image(record, replacement.image_id)
        await _delete_discarded_grid_image(
            record,
            replacement.image_id,
            reason="washed-out repair was still unusable",
        )
        repair.status = "failed"
        repair.message = "The single retake was also washed out and was discarded"
        return
    record.frames.append(replacement.model_copy(update={"target_index": frame.target_index}))
    record.frames.sort(key=lambda item: item.target_index)
    repair.selected_image_id = replacement.image_id
    repair.status = "complete"
    repair.message = "Washed-out original deleted and replaced by one verified retake"


async def _repair_leaf_obstruction(
    record: PhotoGridRecord,
    frame: PhotoGridFrame,
    repair: PhotoGridQualityRepair,
) -> None:
    target = next(item for item in record.targets if item.index == frame.target_index)
    offsets = _leaf_offset_targets(record, target)
    candidates = await _capture_quality_targets(record, offsets)
    repair.candidate_image_ids = [item.image_id for item in candidates]
    # Until ranking finishes, none of the four offset photos is authorised for
    # analysis. Persist that conservative state before downloading them.
    for candidate in candidates:
        _exclude_grid_image(record, candidate.image_id)
    photo_grid_store.save(record)
    inspected: list[tuple[int, PhotoQuality]] = []
    for candidate in candidates:
        try:
            inspected.append(
                (
                    candidate.image_id,
                    inspect_photo_quality(await _quality_jpeg(record, candidate.image_id)),
                )
            )
        except (HomeAssistantError, ValueError) as exc:
            LOGGER.warning(
                "Could not inspect leaf-repair candidate %d: %s",
                candidate.image_id,
                exc,
            )
    selected_id = best_unobscured_photo(inspected)
    for candidate in candidates:
        if candidate.image_id == selected_id:
            continue
        await _delete_discarded_grid_image(
            record,
            candidate.image_id,
            reason="unselected leaf-obstruction offset",
        )
    if selected_id is None:
        repair.status = "failed"
        repair.message = "The four-position repair produced no inspectable verified image"
        return
    selected = next(item for item in candidates if item.image_id == selected_id)
    record.excluded_image_ids = [
        image_id for image_id in record.excluded_image_ids if image_id != selected_id
    ]
    record.quality_overlay_frames = [
        item for item in record.quality_overlay_frames if item.target_index != frame.target_index
    ]
    record.quality_overlay_frames.append(
        selected.model_copy(update={"target_index": frame.target_index})
    )
    repair.selected_image_id = selected_id
    repair.status = "complete"
    repair.message = (
        "Selected the offset with the most unobscured plant; original retained as background"
    )


async def _photo_grid_quality_pass(record: PhotoGridRecord) -> None:
    """Inspect every base tile once and persist at most one repair per issue."""

    attempted = {item.original_image_id for item in record.quality_repairs}
    originals = [item for item in record.frames if item.image_id not in attempted]
    targets = {item.index for item in record.targets}
    for frame in originals:
        if frame.target_index not in targets:
            continue
        try:
            quality = inspect_photo_quality(await _quality_jpeg(record, frame.image_id))
        except (HomeAssistantError, ValueError) as exc:
            LOGGER.warning("Could not quality-check photo-grid image %d: %s", frame.image_id, exc)
            continue
        if quality.issue == "usable":
            continue
        repair = PhotoGridQualityRepair(
            target_index=frame.target_index,
            issue=quality.issue,
            original_image_id=frame.image_id,
            attempted_at=datetime.now(UTC),
        )
        # Save before moving the bot. A restart can report this attempt as
        # interrupted, but can never issue an accidental second repair.
        record.quality_repairs.append(repair)
        photo_grid_store.save(record)
        try:
            if quality.issue == "washed_out":
                await _repair_washed_out_frame(record, frame, repair)
            else:
                await _repair_leaf_obstruction(record, frame, repair)
        except (HomeAssistantError, ValueError, StopIteration) as exc:
            repair.status = "failed"
            repair.message = str(exc)
            LOGGER.warning(
                "Quality repair for grid image %d failed safely: %s",
                frame.image_id,
                exc,
            )
        repair.completed_at = datetime.now(UTC)
        photo_grid_store.save(record)


async def _photo_grid_worker(record: PhotoGridRecord) -> None:
    """Capture the calibrated bed grid, retrying only coordinates not verified."""
    try:
        pending = list(record.targets)
        passes_run = 0
        # The integration verifies movement, upload processing, and returned
        # coordinates before advancing within each batch. Each pass is
        # limited to the exact targets whose verified frames were absent. A
        # pass that verifies no new frames stops the worker early, since
        # another identical pass would only move the bot pointlessly.
        for attempt in range(PHOTO_GRID_WORKER_MAX_PASSES):
            if not pending:
                break
            passes_run += 1
            verified_before_pass = len(record.frames)
            retry: list[PhotoGridTarget] = []
            # Batches are consecutive slices of the canonical route, in route
            # order: the first cell of batch N+1 is the cell that follows the
            # last cell of batch N. Nothing is renumbered, repeated or
            # regenerated at a boundary.
            chunk_size = max(1, record.chunk_size)
            for start in range(0, len(pending), chunk_size):
                chunk = pending[start : start + chunk_size]
                batch_number = start // chunk_size + 1
                try:
                    retry.extend(await _capture_photo_grid_targets(record, chunk))
                except HomeAssistantError as exc:
                    LOGGER.warning(
                        "Photo grid %s batch %d failed: %s",
                        record.session_id,
                        batch_number,
                        exc,
                    )
                    # Only requeue targets whose frames were never verified --
                    # a batch error must not discard photos already confirmed
                    # and merged into record.frames during polling.
                    verified_now = {frame.target_index for frame in record.frames}
                    retry.extend(target for target in chunk if target.index not in verified_now)
                verified_now = {frame.target_index for frame in record.frames}
                batch_verified = sum(1 for target in chunk if target.index in verified_now)
                LOGGER.info(
                    "Photo grid %s pass %d batch %d: %d of %d targets verified this "
                    "batch (%d of %d verified overall)",
                    record.session_id,
                    attempt + 1,
                    batch_number,
                    batch_verified,
                    len(chunk),
                    len(verified_now),
                    len(record.targets),
                )
            pending = retry
            if not pending:
                break
            if len(record.frames) == verified_before_pass:
                LOGGER.warning(
                    "Photo grid %s pass %d verified no new frames; stopping early "
                    "instead of running all %d passes",
                    record.session_id,
                    attempt + 1,
                    PHOTO_GRID_WORKER_MAX_PASSES,
                )
                break
            if attempt < PHOTO_GRID_WORKER_MAX_PASSES - 1:
                record.status = "retrying"
                record.message = (
                    f"Retrying {len(pending)} coordinate"
                    f"{'s' if len(pending) != 1 else ''} without a verified photo "
                    f"(pass {attempt + 2} of {PHOTO_GRID_WORKER_MAX_PASSES})"
                )
                photo_grid_store.save(record)

        record.completed_at = datetime.now(UTC)
        if pending:
            missing_all = ", ".join(
                _coordinate_label(target) for target in pending[:PHOTO_GRID_MISSING_LOG_LIMIT]
            )
            if len(pending) > PHOTO_GRID_MISSING_LOG_LIMIT:
                missing_all += f", … (+{len(pending) - PHOTO_GRID_MISSING_LOG_LIMIT} more)"
            LOGGER.warning(
                "Photo grid %s finished with %d of %d coordinates unverified: %s",
                record.session_id,
                len(pending),
                len(record.targets),
                missing_all,
            )
            examples = " · ".join(
                _coordinate_label(target) for target in pending[:PHOTO_GRID_MESSAGE_EXAMPLE_LIMIT]
            )
            record.status = "failed"
            record.message = (
                f"Photo grid stopped safely: {len(pending)} of {len(record.targets)} "
                f"coordinates had no correctly positioned uploaded photo after "
                f"{passes_run} pass{'es' if passes_run != 1 else ''} (e.g. {examples})"
            )
        else:
            record.status = "complete"
            record.message = (
                f"Photo grid complete: all {len(record.targets)} photos were uploaded "
                "and coordinate-verified"
            )
        photo_grid_store.save(record)
        if record.status == "complete":
            try:
                if record.quality_repair_enabled:
                    record.status = "quality_check"
                    record.message = (
                        "Photo grid captured; checking exposure and close-leaf obstruction"
                    )
                    photo_grid_store.save(record)
                    await _photo_grid_quality_pass(record)
                    record.status = "complete"
                    repaired = sum(item.status == "complete" for item in record.quality_repairs)
                    record.message = (
                        f"Photo grid complete: {len(record.frames)} base photos verified"
                        + (
                            f" · {repaired} quality repair{'s' if repaired != 1 else ''} complete"
                            if record.quality_repairs
                            else ""
                        )
                    )
                    photo_grid_store.save(record)
                await _capture_targeted_plant_photos(record)
            except HomeAssistantError as exc:
                LOGGER.warning(
                    "Post-grid capture work failed after grid %s: %s",
                    record.session_id,
                    exc,
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Photo grid %s stopped", record.session_id)
        record.completed_at = datetime.now(UTC)
        record.status = "failed"
        record.message = f"Photo grid stopped safely: {exc}"
        photo_grid_store.save(record)


async def start_calibrated_photo_grid() -> PhotoGridRecord:
    """Plan and start a reliable whole-bed grid without the legacy sequence."""
    global photo_grid_task
    if photo_grid_task is not None and not photo_grid_task.done():
        current = photo_grid_store.load()
        if current is not None:
            return current
        raise HomeAssistantError("A photo grid is already running")
    bot = await _require_grid_repair_capability(require_lighting=True)
    entry_id = settings.selected_config_entry_id
    calibration = calibration_store.get(entry_id)
    if calibration is None:
        raise HomeAssistantError("Save FarmBot camera calibration before starting a photo grid")

    soil = await client.soil_points(entry_id)
    x_bounds = soil.motion.axis_bounds.get("x")
    y_bounds = soil.motion.axis_bounds.get("y")
    if x_bounds is None or y_bounds is None:
        raise HomeAssistantError("FarmBot did not report usable X/Y axis bounds")
    inventory = await client.inventory(
        InventoryRequest(
            config_entry_id=entry_id,
            image_lookback_hours=min(720, max(72, settings.image_lookback_hours)),
        )
    )
    z_bounds = soil.motion.axis_bounds.get("z")
    if inventory.images:
        ordered_z = sorted(float(image.meta.z) for image in inventory.images)
        z = ordered_z[len(ordered_z) // 2]
    elif z_bounds is not None:
        z = min(z_bounds[1], max(z_bounds[0], 0.0))
    else:
        current_z = soil.motion.position.get("z")
        if current_z is None:
            raise HomeAssistantError("FarmBot did not report a safe photo Z coordinate")
        z = float(current_z)
    if z_bounds is not None:
        z = min(z_bounds[1], max(z_bounds[0], z))

    targets, footprint_width, footprint_height = plan_photo_grid(
        calibration,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z=z,
    )
    chunk_size = photo_grid_chunk_size(bot.capabilities)
    record = PhotoGridRecord(
        config_entry_id=entry_id,
        started_at=datetime.now(UTC),
        status="queued",
        message=f"Queued {len(targets)} calibrated photo coordinates",
        bed_bounds={"x": x_bounds, "y": y_bounds},
        footprint_width_mm=footprint_width,
        footprint_height_mm=footprint_height,
        calibration=calibration,
        targets=targets,
        chunk_size=chunk_size,
        indexed_targets=bot.supports("indexed_photo_grid_targets"),
        quality_repair_enabled=True,
    )
    if chunk_size < len(targets):
        LOGGER.warning(
            "Photo grid %s will be sent as %d separate calls: the loaded FarmBot "
            "integration (%s) does not advertise %s, so it runs the lighting and a "
            "return to the staging position once per call instead of once per grid. "
            "Update the integration to capture the route continuously.",
            record.session_id,
            math.ceil(len(targets) / chunk_size),
            bot.integration_version or "unknown version",
            PHOTO_GRID_CONTINUOUS_CAPABILITY,
        )
    photo_grid_store.save(record)
    photo_grid_task = asyncio.create_task(
        _photo_grid_worker(record), name=f"photo-grid-{record.session_id}"
    )
    return record


def _calibration_from_input(entry_id: str, values: FarmbotCalibrationInput) -> Calibration:
    """Build a processed-resolution calibration from stored FarmBot inputs."""
    resolution = settings.resolution
    return from_farmbot_calibration(
        coordinate_scale_mm_per_px=values.coordinate_scale,
        reference_width=values.reference_width,
        reference_height=values.reference_height,
        processed_width=resolution.width,
        processed_height=resolution.height,
        rotation_degrees=values.rotation_degrees,
        offset_x_mm=values.offset_x_mm,
        offset_y_mm=values.offset_y_mm,
        origin_location=values.origin_location,
        uncertainty_mm=settings.calibration_uncertainty_mm,
        analysis_resolution=resolution.value,
    )


def seed_calibration_from_store() -> None:
    """Restore the active DB calibration from the durable /data store on boot.

    The store is the master record of the FarmBot calibration the user entered;
    the SQLite active calibration is the runtime source the analysis pipeline
    reads. If a bot has a stored calibration but no active DB calibration (fresh
    container, wiped DB), re-derive and persist it so a restart never loses
    calibration and never requires re-entry.
    """
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        return
    stored = calibration_store.get(entry_id)
    if stored is None or database.active_calibration(entry_id) is not None:
        return
    try:
        database.save_calibration(entry_id, _calibration_from_input(entry_id, stored))
        LOGGER.info("Restored calibration for %s from the /data store", entry_id)
    except ValueError as exc:
        LOGGER.warning("Could not restore stored calibration for %s: %s", entry_id, exc)


def _normalize_leading_slashes(value: str) -> str:
    """Collapse only duplicate slashes at the beginning of an ASGI path."""

    return f"/{value.lstrip('/')}" if value.startswith("//") else value


class NormalizeIngressPathMiddleware:
    """Normalize duplicate leading slashes before FastAPI route matching."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in {"http", "websocket"}:
            path = scope.get("path", "")
            raw_path = scope.get("raw_path", b"")
            normalized_path = _normalize_leading_slashes(path)
            normalized_raw_path = (
                b"/" + raw_path.lstrip(b"/") if raw_path.startswith(b"//") else raw_path
            )
            if normalized_path != path or normalized_raw_path != raw_path:
                scope = dict(scope)
                scope["path"] = normalized_path
                scope["raw_path"] = normalized_raw_path
        await self.app(scope, receive, send)


# FarmBot-style composite calibration view. The selected whole-bed photo set is
# stitched in garden-coordinate space using FarmBot camera and Farm Designer
# transforms, with plant and weed centres overlaid so alignment across the whole
# bed can be verified at once. Vanilla JS on a canvas -- no frontend build
# toolchain (Part 5). The rotation direction here MUST match vision.ROTATION_SIGN
# and vision.garden_to_pixel.
_CALIBRATION_JS = r"""
(function(){
  const ROT_SIGN=-1;           // FarmBot Web App uses rotate(-camera_rotation)
  const MAX_CANVAS=2400;       // cap composite dimensions to bound memory
  const canvas=document.getElementById('canvas');
  const ctx=canvas.getContext('2d');
  const viewport=document.getElementById('calibration-grid-viewport');
  const status=document.getElementById('status');
  const ppmEl=document.getElementById('ppm');
  let scene={images:[],plants:[],weeds:[],bed_bounds:null}, current=null, pending=false;
  let viewZoom=1;

  function entry(){return document.getElementById('entry_id').value.trim();}
  function num(id){return parseFloat(document.getElementById(id).value)||0;}
  function checked(id){return document.getElementById(id).checked;}
  function origin(){return document.getElementById('origin').value;}
  function mapOrigin(){return document.getElementById('map_origin').value;}
  function rotateMap(){return checked('rotate_map');}
  function applyZoom(next){
    viewZoom=Math.max(1,Math.min(6,next));
    canvas.style.width=(viewZoom*100)+'%';
    canvas.style.height='auto';
    document.getElementById('zoom-value').textContent=Math.round(viewZoom*100)+'%';
    document.getElementById('zoom-out').disabled=viewZoom<=1;
    document.getElementById('zoom-in').disabled=viewZoom>=6;
  }
  function originSigns(o){
    return [(o==='top_right'||o==='bottom_right')?-1:1,
            (o==='bottom_left'||o==='bottom_right')?-1:1];
  }
  // FarmBot calibration inputs, or null when incomplete.
  function params(){
    const scale=num('fb_scale'), refw=num('fb_refw'), refh=num('fb_refh');
    if(!(scale>0&&refw>0&&refh>0)) return null;
    const s=originSigns(origin());
    return {scale:scale,refw:refw,refh:refh,sx:s[0],sy:s[1],
            rot:num('rotation')*Math.PI/180*ROT_SIGN,ox:num('offx'),oy:num('offy')};
  }
  // Pixels-per-mm of one processed image (its own natural size) under p.
  // FarmBot sizes map photos from the loaded image's actual dimensions. The
  // integration reports those pre-resize dimensions so the smaller JPEG can
  // reproduce the same physical footprint exactly.
  function imagePpm(p,iw,ih,captureW,captureH){
    return [iw/(p.scale*(captureW||p.refw)),ih/(p.scale*(captureH||p.refh))];
  }
  function croppedFootprint(p,captureW,captureH){
    let width=p.scale*(captureW||p.refw), height=p.scale*(captureH||p.refh);
    const angle=Math.abs(num('rotation'));
    let closest=angle%90;if(closest>45) closest=90-closest;
    const rotated90=(angle+45)%180>90;
    if(closest>40){
      const side=Math.min(width,height)/Math.sqrt(2);
      return [side,side];
    }
    let crop=0;
    if(closest>0){
      const factor=(5.61-.095*closest*closest+9.06*closest)/640;
      crop=Math.round(Math.max(width,height)*factor);
    }
    width=Math.max(1,width-crop);height=Math.max(1,height-crop);
    return rotated90?[height,width]:[width,height];
  }
  function calibrationTileBounds(records,p,xmin,xmax,ymin,ymax){
    const axis=function(values,lower,upper){
      const sorted=values.slice().sort((a,b)=>a-b),groups=[];
      sorted.forEach(function(value){
        const last=groups[groups.length-1];
        if(last&&Math.abs(value-last.center)<=25){
          last.values.push(value);
          last.center=last.values.reduce((a,b)=>a+b,0)/last.values.length;
        }else groups.push({values:[value],center:value});
      });
      return function(value){
        let nearest=0;
        groups.forEach(function(group,index){
          if(Math.abs(group.center-value)<Math.abs(groups[nearest].center-value)) nearest=index;
        });
        return [
          nearest===0?lower:(groups[nearest-1].center+groups[nearest].center)/2,
          nearest===groups.length-1?upper:(groups[nearest].center+groups[nearest+1].center)/2];
      };
    };
    const xCell=axis(records.map(r=>r.info.x+p.ox),xmin,xmax);
    const yCell=axis(records.map(r=>r.info.y+p.oy),ymin,ymax);
    const result={};
    records.forEach(function(r){
      const x=xCell(r.info.x+p.ox),y=yCell(r.info.y+p.oy);
      result[String(r.info.id)]=[x[0],y[0],x[1],y[1]];
    });
    return result;
  }
  function strokeCalibrationTiles(ctx,tiles,toCanvas,width,height){
    ctx.save();ctx.setTransform(1,0,0,1,0,0);
    ctx.strokeStyle='rgba(225,238,227,.32)';ctx.lineWidth=1;
    Object.values(tiles).forEach(function(cell){
      const points=[
        toCanvas(cell[0],cell[1]),toCanvas(cell[2],cell[1]),
        toCanvas(cell[2],cell[3]),toCanvas(cell[0],cell[3])];
      ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);
      points.slice(1).forEach(function(point){ctx.lineTo(point[0],point[1]);});
      ctx.closePath();ctx.stroke();
    });
    ctx.strokeStyle='#245b38';ctx.lineWidth=4;
    ctx.strokeRect(2,2,width-4,height-4);ctx.restore();
  }
  // Map a source pixel (u,v) of an image taken at (cx,cy) to a garden coord.
  // Inverse of vision.garden_to_pixel.
  function pixelToCoord(p,cx,cy,iw,ih,u,v,captureW,captureH){
    const ppm=imagePpm(p,iw,ih,captureW,captureH);
    const rx=u-iw/2, ry=v-ih/2;
    const c=Math.cos(p.rot), s=Math.sin(p.rot);
    const vx=c*rx - s*ry, vy=s*rx + c*ry;
    return [cx + vx/(p.sx*ppm[0])+p.ox, cy + vy/(p.sy*ppm[1])+p.oy];
  }
  function releaseImages(batch){
    if(!batch) return;
    batch.images.forEach(function(rec){
      if(rec.objectUrl) URL.revokeObjectURL(rec.objectUrl);
    });
  }
  function positiveHeader(response,name){
    const value=parseFloat(response.headers.get(name));
    return value>0?value:null;
  }
  function loadGridImages(){
    const p=params();
    ppmEl.textContent=p?('FarmBot coordinate scale: '+p.scale+' mm/capture pixel'):
                        'Enter the FarmBot pixel coordinate scale, and measured-at width/height';
    if(!scene.images.length){current=null;clearCanvas('No photos in this date range');return;}
    releaseImages(current);
    const batch={images:[]};
    current=batch;
    status.textContent='Loading '+scene.images.length+' grid photos…';
    scene.images.forEach(im=>{
      const image=new Image();
      const rec={info:im,img:image,loaded:false,captureW:null,captureH:null,objectUrl:null};
      batch.images.push(rec);
      const url='api/vision/image/'+im.id+'.jpg?entry_id='+encodeURIComponent(entry());
      fetch(url).then(function(response){
        if(!response.ok) throw new Error('HTTP '+response.status);
        rec.captureW=positiveHeader(response,'X-FarmBot-Oriented-Width');
        rec.captureH=positiveHeader(response,'X-FarmBot-Oriented-Height');
        return response.blob();
      }).then(function(blob){
        if(current!==batch) return;
        rec.objectUrl=URL.createObjectURL(blob);
        image.onload=function(){rec.loaded=true;render();};
        image.onerror=function(){status.textContent='Could not decode image #'+im.id;};
        image.src=rec.objectUrl;
      }).catch(function(error){
        if(current===batch){
          status.textContent='Could not load image #'+im.id+': '+error.message;
        }
      });
    });
  }
  function clearCanvas(msg){
    canvas.width=900;canvas.height=420;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#888';ctx.font='14px sans-serif';ctx.fillText(msg||'',12,28);
  }
  function scheduleRender(){
    if(pending) return; pending=true;
    requestAnimationFrame(function(){pending=false;render();});
  }
  function render(){
    const p=params();
    ppmEl.textContent=p
      ?('Using FarmBot scale verbatim: '+p.scale+' mm/capture pixel; '
        +'selected capture resolution '+p.refw+'×'+p.refh)
      :'Enter the FarmBot pixel coordinate scale and Camera settings resolution';
    document.getElementById('save').disabled=!(p&&checked('confirm'));
    if(!current){return;}
    const loaded=current.images.filter(r=>r.loaded&&r.img.naturalWidth>0);
    if(!p){clearCanvas('Enter FarmBot calibration values to build the composite');return;}
    if(!loaded.length){return;}
    // Resolution differences are diagnostic, not a reason to hide evidence.
    // FarmBot sizes every map photo from the loaded image itself, so render
    // every capture with its own actual dimensions and warn below the grid.
    const matchesCalibration=function(r){
      if(!(r.captureW&&r.captureH)) return true; // legacy integration fallback
      const direct=Math.abs(r.captureW-p.refw)<5&&Math.abs(r.captureH-p.refh)<5;
      const swapped=Math.abs(r.captureH-p.refw)<5&&Math.abs(r.captureW-p.refh)<5;
      return direct||swapped;
    };
    const ready=loaded;
    const mismatched=loaded.filter(function(r){return !matchesCalibration(r);});
    // Garden-space bounding box from every image's four corners.
    let gxmin=Infinity,gxmax=-Infinity,gymin=Infinity,gymax=-Infinity,ppmSum=0;
    ready.forEach(r=>{
      const iw=r.img.naturalWidth, ih=r.img.naturalHeight;
      const pp=imagePpm(p,iw,ih,r.captureW,r.captureH); ppmSum+=(pp[0]+pp[1])/2;
      [[0,0],[iw,0],[0,ih],[iw,ih]].forEach(c=>{
        const g=pixelToCoord(
          p,r.info.x,r.info.y,iw,ih,c[0],c[1],r.captureW,r.captureH);
        gxmin=Math.min(gxmin,g[0]);gxmax=Math.max(gxmax,g[0]);
        gymin=Math.min(gymin,g[1]);gymax=Math.max(gymax,g[1]);
      });
    });
    // Use the FarmBot bed bounds when the latest grid or motion state supplied
    // them. This makes the calibration canvas the same whole-bed view as the
    // analysis grid instead of shrinking around whichever photos happened to load.
    if(scene.bed_bounds&&scene.bed_bounds.x&&scene.bed_bounds.y){
      gxmin=scene.bed_bounds.x[0];gxmax=scene.bed_bounds.x[1];
      gymin=scene.bed_bounds.y[0];gymax=scene.bed_bounds.y[1];
    }
    let P=ppmSum/ready.length;
    const rangeX=Math.max(1,gxmax-gxmin), rangeY=Math.max(1,gymax-gymin);
    const displayRangeX=rotateMap()?rangeY:rangeX;
    const displayRangeY=rotateMap()?rangeX:rangeY;
    P=Math.min(P,MAX_CANVAS/displayRangeX,MAX_CANVAS/displayRangeY);
    canvas.width=Math.max(1,Math.round(displayRangeX*P));
    canvas.height=Math.max(1,Math.round(displayRangeY*P));
    const toCanvas=function(gx,gy){
      let x=gx-gxmin, y=gy-gymin;
      if(rotateMap()){const oldX=x;x=y;y=oldX;}
      if(mapOrigin()==='top_right'||mapOrigin()==='bottom_right'){
        x=displayRangeX-x;
      }
      if(mapOrigin()==='bottom_left'||mapOrigin()==='bottom_right'){
        y=displayRangeY-y;
      }
      return [x*P,y*P];
    };
    const tileBounds=calibrationTileBounds(ready,p,gxmin,gxmax,gymin,gymax);
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);
    // Paint each image via the affine that maps its source pixels into the
    // composite (three mapped points fully determine the affine).
    ctx.imageSmoothingEnabled=true;
    ready.forEach(r=>{
      const iw=r.img.naturalWidth, ih=r.img.naturalHeight;
      const p0=toCanvas.apply(null,pixelToCoord(
        p,r.info.x,r.info.y,iw,ih,0,0,r.captureW,r.captureH));
      const pu=toCanvas.apply(null,pixelToCoord(
        p,r.info.x,r.info.y,iw,ih,iw,0,r.captureW,r.captureH));
      const pv=toCanvas.apply(null,pixelToCoord(
        p,r.info.x,r.info.y,iw,ih,0,ih,r.captureW,r.captureH));
      const footprint=croppedFootprint(p,r.captureW,r.captureH);
      const opticalX=r.info.x+p.ox, opticalY=r.info.y+p.oy;
      const tile=tileBounds[String(r.info.id)];
      const crop=tile||[
        opticalX-footprint[0]/2,opticalY-footprint[1]/2,
        opticalX+footprint[0]/2,opticalY+footprint[1]/2];
      const clip=[
        toCanvas(crop[0],crop[1]),toCanvas(crop[2],crop[1]),
        toCanvas(crop[2],crop[3]),toCanvas(crop[0],crop[3])];
      ctx.setTransform(1,0,0,1,0,0);ctx.save();
      ctx.beginPath();ctx.moveTo(clip[0][0],clip[0][1]);
      clip.slice(1).forEach(function(point){ctx.lineTo(point[0],point[1]);});
      ctx.closePath();ctx.clip();
      ctx.setTransform((pu[0]-p0[0])/iw,(pu[1]-p0[1])/iw,
                       (pv[0]-p0[0])/ih,(pv[1]-p0[1])/ih,p0[0],p0[1]);
      ctx.drawImage(r.img,0,0);ctx.restore();
    });
    ctx.setTransform(1,0,0,1,0,0);
    strokeCalibrationTiles(ctx,tileBounds,toCanvas,canvas.width,canvas.height);
    if(checked('showoverlay')) drawOverlay(p,toCanvas,P);
    const dimensions=[...new Set(ready.map(function(r){
      return r.captureW&&r.captureH?(r.captureW+'×'+r.captureH):'legacy dimensions';
    }))].join(', ');
    const mismatchDimensions=[...new Set(mismatched.map(function(r){
      return r.captureW+'×'+r.captureH;
    }))].join(', ');
    const resolutionWarning=mismatched.length
      ?(' Warning: selected resolution '+p.refw+'×'+p.refh
        +', but '+mismatched.length+' loaded photo(s) report '+mismatchDimensions
        +'. All photos are still shown at their reported size.')
      :'';
    const bounds=scene.bed_bounds
      ?(' Bed: X '+gxmin+'–'+gxmax+' mm, Y '+gymin+'–'+gymax+' mm'
        +(scene.bed_bounds_source?(' ('+scene.bed_bounds_source+')'):'')+'.')
      :'';
    status.textContent='Full tessellated grid: '+ready.length+' photos'
      +'; actual capture '+dimensions+'. '+scene.plants.length+' plants, '
      +scene.weeds.length+' weeds. '
      +'Confirm centres sit on their plants across the bed.'+bounds+resolutionWarning;
  }
  function marker(p,toCanvas,P,pt,colour,label){
    const c=toCanvas(pt.x,pt.y);
    if(c[0]<-40||c[1]<-40||c[0]>canvas.width+40||c[1]>canvas.height+40) return;
    ctx.strokeStyle=colour;ctx.fillStyle=colour;ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(c[0],c[1],Math.max(4,(pt.radius||0)*P),0,7);ctx.stroke();
    ctx.beginPath();ctx.arc(c[0],c[1],2.5,0,7);ctx.fill();
    if(label&&checked('showlabels')){
      ctx.font='12px sans-serif';
      ctx.fillText(label,c[0]+5,c[1]-5);
    }
  }
  function drawOverlay(p,toCanvas,P){
    scene.plants.forEach(pl=>marker(p,toCanvas,P,pl,'#2ecc40',
      (pl.name||('#'+pl.id))+(pl.slug?(' ('+pl.slug+')'):'')));
    scene.weeds.forEach(w=>marker(p,toCanvas,P,w,'#ff4136',w.name||'Weed'));
  }

  document.getElementById('load').addEventListener('click',async function(){
    const dateFrom=document.getElementById('date_from').value;
    const dateTo=document.getElementById('date_to').value;
    if(!dateFrom||!dateTo){status.textContent='Choose both a from and to date';return;}
    if(new Date(dateFrom)>new Date(dateTo)){status.textContent='From must be before to';return;}
    status.textContent='Loading date-filtered grid…';
    try{
      const query=new URLSearchParams({entry_id:entry(),
        date_from:new Date(dateFrom).toISOString(),date_to:new Date(dateTo).toISOString()});
      const r=await fetch('api/vision/images?'+query.toString());
      if(!r.ok) throw new Error('HTTP '+r.status);
      scene=await r.json();
      scene.images=scene.images||[];scene.plants=scene.plants||[];scene.weeds=scene.weeds||[];
      // FarmBot reverses the API's newest-first collection before painting,
      // leaving the newest neighbouring capture on top where tiles overlap.
      scene.images.sort(function(a,b){
        return new Date(a.created_at)-new Date(b.created_at);
      });
      status.textContent=scene.images.length+' unique grid locations, '
        +scene.plants.length+' plants, '+scene.weeds.length+' weeds';
      if(scene.images.length) loadGridImages(); else clearCanvas('No images in this date range');
    }catch(err){status.textContent='Could not load inventory: '+err.message;}
  });
  ['date_from','date_to'].forEach(function(id){
    document.getElementById(id).addEventListener('change',function(){
      document.getElementById('load').click();
    });
  });
  ['fb_scale','fb_refw','fb_refh','rotation','origin','map_origin','offx','offy'].forEach(function(id){
    document.getElementById(id).addEventListener('input',scheduleRender);
    document.getElementById(id).addEventListener('change',scheduleRender);
  });
  ['rotate_map','showoverlay','showlabels','confirm'].forEach(function(id){
    document.getElementById(id).addEventListener('change',scheduleRender);
  });
  document.getElementById('zoom-in').addEventListener('click',function(){
    applyZoom(viewZoom+.5);
  });
  document.getElementById('zoom-out').addEventListener('click',function(){
    applyZoom(viewZoom-.5);
  });
  document.getElementById('zoom-reset').addEventListener('click',function(){
    applyZoom(1);viewport.scrollLeft=0;viewport.scrollTop=0;
  });
  document.getElementById('save').addEventListener('click',function(){
    const p=params();
    if(!p){status.textContent='Enter the FarmBot calibration values first';return;}
    const f=document.createElement('form');f.method='post';f.action='calibration';
    const fields={entry_id:entry(),coordinate_scale:num('fb_scale'),
      reference_width:num('fb_refw'),reference_height:num('fb_refh'),
      rotation:num('rotation'),origin_location:origin(),
      map_origin:mapOrigin(),rotate_map:rotateMap(),
      offset_x:num('offx'),offset_y:num('offy')};
    for(const k in fields){const i=document.createElement('input');i.type='hidden';
      i.name=k;i.value=fields[k];f.appendChild(i);}
    document.body.appendChild(f);f.submit();
  });
  clearCanvas('Loading the date-filtered photo grid…');
  applyZoom(1);
  if(entry()) document.getElementById('load').click();
})();
"""

_DASHBOARD_JS = r"""
(function(){
  const modal=document.getElementById('overlay-modal');
  const modalImg=document.getElementById('overlay-modal-img');
  const modalDetails=document.getElementById('overlay-modal-details');
  const closeButton=document.getElementById('overlay-modal-close');
  const counter=document.getElementById('overlay-modal-counter');
  const artifactControls=document.getElementById('artifact-controls');
  const overlayLegend=document.getElementById('overlay-modal-legend');
  const plantPhotoMode=document.getElementById('plant-photo-mode');
  const plantPhotoTab=document.getElementById('plant-photo-tab');
  const plantDiagnosticTab=document.getElementById('plant-diagnostic-tab');
  let artifacts=[], index=0, returnFocus=null;
  let plantComposite=null;
  const queueModal=document.getElementById('queue-modal');
  const queueRows=document.getElementById('queue-image-rows');
  const queueMessage=document.getElementById('queue-message');
  const gantryModal=document.getElementById('gantry-modal');
  const gantryOpen=document.getElementById('gantry-debug-open');
  const photoGridModal=document.getElementById('photo-grid-modal');
  const photoGridOpen=document.getElementById('photo-grid-open');
  const photoGridCanvas=document.getElementById('photo-grid-canvas');
  const photoGridStatus=document.getElementById('photo-grid-status');
  function gridOriginSigns(origin){
    return [(origin==='top_right'||origin==='bottom_right')?-1:1,
            (origin==='bottom_left'||origin==='bottom_right')?-1:1];
  }
  function gridPositiveHeader(response,name){
    const value=parseFloat(response.headers.get(name));
    return value>0?value:null;
  }
  function gridCroppedFootprint(calibration,captureW,captureH){
    let width=calibration.coordinate_scale
      *(captureW||calibration.reference_width);
    let height=calibration.coordinate_scale
      *(captureH||calibration.reference_height);
    const angle=Math.abs(calibration.rotation_degrees);
    let closest=angle%90;if(closest>45) closest=90-closest;
    const rotated90=(angle+45)%180>90;
    if(closest>40){
      const side=Math.min(width,height)/Math.sqrt(2);
      return [side,side];
    }
    let crop=0;
    if(closest>0){
      const factor=(5.61-.095*closest*closest+9.06*closest)/640;
      crop=Math.round(Math.max(width,height)*factor);
    }
    width=Math.max(1,width-crop);height=Math.max(1,height-crop);
    return rotated90?[height,width]:[width,height];
  }
  function gridTileBounds(record){
    const calibration=record.calibration;
    const axis=function(label,coordinate,offset,limits){
      const groups=new Map();
      (record.targets||[]).forEach(function(target){
        const key=String(target[label]);
        if(!groups.has(key)) groups.set(key,[]);
        groups.get(key).push(Number(target[coordinate])+offset);
      });
      const ordered=[...groups.entries()].map(function(entry){
        return {key:entry[0],center:entry[1].reduce((a,b)=>a+b,0)/entry[1].length};
      }).sort(function(a,b){return a.center-b.center;});
      const result={};
      ordered.forEach(function(item,index){
        result[item.key]=[
          index===0?limits[0]:(ordered[index-1].center+item.center)/2,
          index===ordered.length-1?limits[1]:(item.center+ordered[index+1].center)/2];
      });
      return result;
    };
    const xs=axis('column','x',calibration.offset_x_mm,record.bed_bounds.x);
    const ys=axis('row','y',calibration.offset_y_mm,record.bed_bounds.y);
    const result={};
    (record.targets||[]).forEach(function(target){
      const x=xs[String(target.column)],y=ys[String(target.row)];
      result[String(target.index)]=[x[0],y[0],x[1],y[1]];
    });
    return result;
  }
  function strokeGridTiles(ctx,tileBounds,project,canvasWidth,canvasHeight){
    ctx.save();ctx.setTransform(1,0,0,1,0,0);
    ctx.strokeStyle='rgba(225,238,227,.32)';ctx.lineWidth=1;
    Object.values(tileBounds).forEach(function(cell){
      const a=project(cell[0],cell[1]),b=project(cell[2],cell[3]);
      ctx.strokeRect(a[0]+.5,a[1]+.5,b[0]-a[0]-1,b[1]-a[1]-1);
    });
    ctx.strokeStyle='#245b38';ctx.lineWidth=4;
    ctx.strokeRect(2,2,canvasWidth-4,canvasHeight-4);ctx.restore();
  }
  /* Draws each frame's photo into ctx using the same calibrated, per-pixel
     projective transform regardless of whether the caller wants the whole
     bed (drawPhotoGrid) or the stored, tightly cropped plant review artifact.
     project(x,y) maps a garden-mm coordinate to canvas pixels; onDone(loaded,
     failed,dimensions) fires once every frame has resolved (loaded or failed). */
  function drawFramesInto(ctx,canvasWidth,canvasHeight,frames,calibration,configEntryId,
      project,tileBounds,onDone){
    if(!frames.length){onDone(0,0,[]);return;}
    const signs=gridOriginSigns(calibration.origin_location);
    const rotation=-calibration.rotation_degrees*Math.PI/180;
    const dimensions=new Set();
    const loadFrame=function(frame){
      const url='api/vision/image/'+frame.image_id+'.jpg?entry_id='
        +encodeURIComponent(configEntryId);
      return fetch(url).then(function(response){
        if(!response.ok) throw new Error('HTTP '+response.status);
        const captureW=gridPositiveHeader(response,'X-FarmBot-Oriented-Width');
        const captureH=gridPositiveHeader(response,'X-FarmBot-Oriented-Height');
        if(captureW&&captureH) dimensions.add(captureW+'×'+captureH);
        return response.blob().then(function(blob){
          return new Promise(function(resolve,reject){
            const image=new Image();
            const objectUrl=URL.createObjectURL(blob);
            image.onload=function(){
              resolve({frame:frame,image:image,captureW:captureW,captureH:captureH,
                       objectUrl:objectUrl});
            };
            image.onerror=function(){
              URL.revokeObjectURL(objectUrl);reject(new Error('Image decode failed'));
            };
            image.src=objectUrl;
          });
        });
      }).catch(function(){return null;});
    };
    Promise.all(frames.map(loadFrame)).then(function(results){
      let loaded=0;
      results.forEach(function(result){
        if(!result) return;
        const frame=result.frame, image=result.image;
        const captureW=result.captureW, captureH=result.captureH;
        const iw=image.naturalWidth, ih=image.naturalHeight;
        const ppmX=iw/(calibration.coordinate_scale
          *(captureW||calibration.reference_width));
        const ppmY=ih/(calibration.coordinate_scale
          *(captureH||calibration.reference_height));
        const garden=function(u,v){
          const rx=u-iw/2, ry=v-ih/2;
          const vx=Math.cos(rotation)*rx-Math.sin(rotation)*ry;
          const vy=Math.sin(rotation)*rx+Math.cos(rotation)*ry;
          return [frame.x+vx/(signs[0]*ppmX)+calibration.offset_x_mm,
                  frame.y+vy/(signs[1]*ppmY)+calibration.offset_y_mm];
        };
        const p0=project.apply(null,garden(0,0));
        const pu=project.apply(null,garden(iw,0));
        const pv=project.apply(null,garden(0,ih));
        const footprint=gridCroppedFootprint(calibration,captureW,captureH);
        const opticalX=frame.x+calibration.offset_x_mm;
        const opticalY=frame.y+calibration.offset_y_mm;
        const tile=tileBounds&&tileBounds[String(frame.target_index)];
        const crop=tile||[
          opticalX-footprint[0]/2,opticalY-footprint[1]/2,
          opticalX+footprint[0]/2,opticalY+footprint[1]/2];
        const clip=[
          project(crop[0],crop[1]),project(crop[2],crop[1]),
          project(crop[2],crop[3]),project(crop[0],crop[3])];
        ctx.save();
        ctx.beginPath();ctx.rect(0,0,canvasWidth,canvasHeight);ctx.clip();
        ctx.beginPath();ctx.moveTo(clip[0][0],clip[0][1]);
        clip.slice(1).forEach(function(point){ctx.lineTo(point[0],point[1]);});
        ctx.closePath();ctx.clip();
        ctx.globalAlpha=.96;
        ctx.setTransform((pu[0]-p0[0])/iw,(pu[1]-p0[1])/iw,
                         (pv[0]-p0[0])/ih,(pv[1]-p0[1])/ih,p0[0],p0[1]);
        ctx.drawImage(image,0,0);ctx.restore();
        URL.revokeObjectURL(result.objectUrl);
        loaded++;
      });
      onDone(loaded,frames.length-loaded,[...dimensions]);
    });
  }
  function drawPhotoGrid(data){
    const record=data.grid, bounds=record.bed_bounds, calibration=record.calibration;
    const layeredFrames=(record.frames||[]).concat(record.quality_overlay_frames||[]);
    const tileBounds=gridTileBounds(record);
    const spanX=bounds.x[1]-bounds.x[0], spanY=bounds.y[1]-bounds.y[0];
    const displayWidth=900;
    photoGridCanvas.width=displayWidth;
    photoGridCanvas.height=Math.max(240,Math.min(650,Math.round(displayWidth*spanY/spanX)));
    const ctx=photoGridCanvas.getContext('2d');
    const sx=photoGridCanvas.width/spanX, sy=photoGridCanvas.height/spanY;
    const project=function(x,y){return [(x-bounds.x[0])*sx,(y-bounds.y[0])*sy];};
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle='#283f30';ctx.fillRect(0,0,photoGridCanvas.width,photoGridCanvas.height);
    if(!layeredFrames.length){photoGridStatus.textContent='This grid has no verified photos yet';return;}
    drawFramesInto(ctx,photoGridCanvas.width,photoGridCanvas.height,layeredFrames,calibration,
      record.config_entry_id,project,tileBounds,function(loaded,failed,dimensions){
        ctx.setTransform(1,0,0,1,0,0);
        strokeGridTiles(
          ctx,tileBounds,project,photoGridCanvas.width,photoGridCanvas.height);
        (data.plants||[]).forEach(function(plant){
          const c=project(plant.x,plant.y);
          ctx.fillStyle='rgba(32,160,82,.18)';ctx.strokeStyle='#39d878';ctx.lineWidth=2;
          ctx.beginPath();ctx.arc(c[0],c[1],Math.max(4,(plant.radius||0)*(sx+sy)/2),0,Math.PI*2);
          ctx.fill();ctx.stroke();
        });
        (data.weeds||[]).forEach(function(weed){
          const c=project(weed.x,weed.y);
          ctx.fillStyle='#ef476f';ctx.beginPath();ctx.arc(c[0],c[1],4,0,Math.PI*2);ctx.fill();
        });
        const planned=gridCroppedFootprint(
          calibration,calibration.reference_width,calibration.reference_height);
        const legacySpacing=Math.abs(record.footprint_width_mm-planned[0])>1
          ||Math.abs(record.footprint_height_mm-planned[1])>1;
        photoGridStatus.textContent=record.message+' · '+loaded
          +' verified photos rendered as tessellated cells'
          +(dimensions.length?' · actual capture '+dimensions.join(', '):'')
          +(failed?' · '+failed+' image(s) could not be loaded':'')
          +(legacySpacing
            ?' · This grid uses older rotation spacing; start a new photo grid to close the gaps.'
            :'');
      });
  }
  function showPlantPhotoTab(showPhoto){
    if(plantComposite) showPlantComposite(!showPhoto);
    plantPhotoTab.setAttribute('aria-pressed',String(showPhoto));
    plantDiagnosticTab.setAttribute('aria-pressed',String(!showPhoto));
  }
  async function loadPhotoGrid(){
    photoGridStatus.textContent='Loading the verified grid…';
    try{
      const response=await fetch('api/photo-grid/latest');
      const data=await response.json();
      if(!response.ok) throw new Error(data.detail||('HTTP '+response.status));
      drawPhotoGrid(data);
    }catch(error){photoGridStatus.textContent='Could not load photo grid: '+error.message;}
  }
  async function loadQueueImages(){
    const dateFrom=document.getElementById('queue-from').value;
    const dateTo=document.getElementById('queue-to').value;
    if(!dateFrom||!dateTo){queueMessage.textContent='Choose both a from and to date';return;}
    if(new Date(dateFrom)>new Date(dateTo)){queueMessage.textContent='From must be before to';return;}
    queueMessage.textContent='Loading images…';
    try{
      const query=new URLSearchParams({date_from:new Date(dateFrom).toISOString(),
        date_to:new Date(dateTo).toISOString()});
      const response=await fetch('api/analysis/images?'+query.toString());
      const data=await response.json();
      if(!response.ok) throw new Error(data.detail||('HTTP '+response.status));
      queueRows.innerHTML=(data.images||[]).map(function(image){
        const plants=(image.plants||[]).map(p=>p.name+' (#'+p.id+')').join(', ')||'None';
        return '<tr><td><input class=queue-checkbox type=checkbox value="'+image.id+'"></td>'
          +'<td>'+image.x.toFixed(1)+', '+image.y.toFixed(1)+', '+image.z.toFixed(1)+'</td>'
          +'<td>'+plants+'</td><td>'+new Date(image.created_at).toLocaleString()+'</td></tr>';
      }).join('')||'<tr><td colspan=4>No images in this timeframe</td></tr>';
      queueMessage.textContent=data.images.length+' images found';
    }catch(error){queueMessage.textContent='Could not load images: '+error.message;}
  }
  function showArtifact(){
    if(!artifacts.length) return;
    modalImg.src=artifacts[index];
    counter.textContent=(index+1)+' / '+artifacts.length;
  }
  function closeModal(){
    modal.hidden=true; modalImg.removeAttribute('src');
    plantComposite=null;
    if(returnFocus) returnFocus.focus();
  }
  function showPlantComposite(withOverlay){
    if(!plantComposite) return;
    const useOverlay=withOverlay&&plantComposite.overlay;
    modalImg.src=useOverlay?plantComposite.overlay:plantComposite.clean;
  }
  const weedModal=document.getElementById('weed-modal');
  const weedImg=document.getElementById('weed-modal-img');
  const weedMarker=document.getElementById('weed-modal-marker');
  const weedDetails=document.getElementById('weed-modal-details');
  const weedGuess=document.getElementById('weed-modal-guess');
  const weedMessage=document.getElementById('weed-modal-message');
  const weedAccept=document.getElementById('weed-modal-accept');
  const weedAcceptAll=document.getElementById('weed-modal-accept-all');
  const weedRejectButtons=[...document.querySelectorAll('[data-weed-label]')];
  const weedPrevious=document.getElementById('weed-modal-prev-weed');
  const weedNext=document.getElementById('weed-modal-next-weed');
  const weedPreviousImage=document.getElementById('weed-modal-prev-image');
  const weedNextImage=document.getElementById('weed-modal-next-image');
  const weedCounter=document.getElementById('weed-modal-weed-counter');
  const weedImageCounter=document.getElementById('weed-modal-image-counter');
  const weedWithoutOverlay=document.getElementById('weed-modal-without-overlay');
  const weedWithOverlay=document.getElementById('weed-modal-with-overlay');
  const weedCloseUp=document.getElementById('weed-modal-closeup');
  const weedImageWrap=document.getElementById('weed-image-wrap');
  const weedLegend=document.getElementById('weed-modal-legend');
  const weedZoomRow=document.getElementById('weed-modal-zoom');
  const weedZoom=document.getElementById('weed-modal-zoom-level');
  const weedZoomValue=document.getElementById('weed-modal-zoom-value');
  const weedUnknown=document.getElementById('weed-modal-unknown');
  const weedActionButtons=[weedAccept,weedAcceptAll,weedUnknown,...weedRejectButtons];
  let weedData=null, weedReturnFocus=null, weedViewers=[], weedImageGroups=[];
  let weedIndex=0, weedImageIndex=0;
  /* Kept across navigation so a reviewer who chose close-up (or the overlay)
     stays in that view while working through a run of detections. */
  let weedViewMode='clean';
  function parseWeedViewer(viewer){
    try{return JSON.parse(viewer.dataset.weed||'null');}catch(_){return null;}
  }
  function refreshWeedNavigation(){
    weedViewers=[...document.querySelectorAll('.weed-view')];
    const groups=new Map();
    weedViewers.forEach(function(viewer){
      const data=parseWeedViewer(viewer);
      if(!data) return;
      const key=String(data.imageId);
      if(!groups.has(key)) groups.set(key,[]);
      groups.get(key).push(viewer);
    });
    weedImageGroups=[...groups.values()];
  }
  function updateWeedNavigation(){
    const group=weedImageGroups[weedImageIndex]||[];
    weedCounter.textContent=group.length?(weedIndex+1)+' / '+group.length:'0 / 0';
    weedImageCounter.textContent=weedImageGroups.length
      ?(weedImageIndex+1)+' / '+weedImageGroups.length:'0 / 0';
    weedPrevious.disabled=weedIndex<=0;
    weedNext.disabled=weedIndex<0||weedIndex>=group.length-1;
    weedPreviousImage.disabled=weedImageIndex<=0;
    weedNextImage.disabled=weedImageIndex<0||weedImageIndex>=weedImageGroups.length-1;
  }
  function setWeedMessage(text,isError){
    weedMessage.textContent=text||'';
    weedMessage.classList.toggle('notice',Boolean(text)&&!isError);
  }
  function hasWeedGeometry(data){
    return Boolean(data&&data.x!=null&&data.y!=null&&data.width&&data.height);
  }
  function canCloseUp(data){
    /* A close-up needs both the pixel geometry to centre on and a clean image
       to crop: zooming the overlay would only magnify the circles. */
    return hasWeedGeometry(data)&&Boolean(data.reviewArtifact);
  }
  function applyCloseUp(){
    /* Percentage translations resolve against the image's own box, so the
       framing stays correct once the image finishes loading and whenever the
       dialog is resized -- no pixel measurements needed. */
    const zoom=Number(weedZoom.value)||5;
    const px=weedData.x/weedData.width, py=weedData.y/weedData.height;
    const limit=1-1/zoom;
    const tx=Math.min(0,Math.max(-limit,0.5/zoom-px));
    const ty=Math.min(0,Math.max(-limit,0.5/zoom-py));
    weedImg.style.transform='scale('+zoom+') translate('+(tx*100)+'%,'+(ty*100)+'%)';
    weedZoomValue.textContent=zoom+'×';
  }
  function showWeedView(mode){
    if(!weedData) return;
    /* Remember what was asked for, not what this particular detection can
       show: an older weed with no clean image must not silently drop every
       later weed out of close-up. */
    weedViewMode=mode;
    const noCleanImage=!weedData.reviewArtifact;
    if(mode==='closeup'&&!canCloseUp(weedData)) mode='clean';
    if(mode==='clean'&&noCleanImage) mode='overlay';
    const closeUp=mode==='closeup';
    weedImg.src=mode==='overlay'?weedData.overlayArtifact:weedData.reviewArtifact;
    weedWithoutOverlay.setAttribute('aria-pressed',String(mode==='clean'));
    weedWithOverlay.setAttribute('aria-pressed',String(mode==='overlay'));
    weedCloseUp.setAttribute('aria-pressed',String(closeUp));
    weedCloseUp.disabled=!canCloseUp(weedData);
    weedImageWrap.classList.toggle('closeup',closeUp);
    weedZoomRow.hidden=!closeUp;
    if(closeUp) applyCloseUp(); else weedImg.style.transform='';
    /* The marker ring is the "which one is it" cue on the wide shot; in the
       close-up the weed already fills the frame, so the ring would only hide it. */
    weedMarker.hidden=closeUp||!hasWeedGeometry(weedData);
    weedLegend.textContent=closeUp
      ?'Close-up of the detected weed on the original image, without any marker or overlay.'
      :'Blue circle = the weed being reviewed; red circles = other detected weeds in this image.';
    setWeedMessage(
      (weedViewMode!=='overlay'&&noCleanImage)
        ?'No image without the overlay was saved for this older detection; showing the analysis overlay instead.'
        :'',
      true
    );
  }
  function closeWeedModal(){
    weedModal.hidden=true; weedImg.removeAttribute('src'); weedImg.style.transform='';
    weedImageWrap.classList.remove('closeup'); weedZoomRow.hidden=true; weedData=null;
    if(weedReturnFocus) weedReturnFocus.focus();
  }
  function showWeedEmptyState(){
    /* The dialog only closes when the reviewer says so, so an exhausted queue
       becomes an empty state rather than a dialog that vanishes under them. */
    weedData=null;
    weedImg.removeAttribute('src'); weedImg.style.transform='';
    weedImageWrap.classList.remove('closeup'); weedZoomRow.hidden=true;
    weedMarker.hidden=true;
    weedDetails.textContent='Every weed recommendation has been reviewed.';
    weedLegend.textContent='';
    weedGuess.hidden=true; weedGuess.textContent='';
    weedCounter.textContent='0 / 0'; weedImageCounter.textContent='0 / 0';
    [weedPrevious,weedNext,weedPreviousImage,weedNextImage].forEach(function(button){
      button.disabled=true;
    });
    weedActionButtons.forEach(function(button){button.disabled=true;});
    [weedWithoutOverlay,weedWithOverlay,weedCloseUp].forEach(function(button){
      button.disabled=true;
    });
    setWeedMessage('Nothing left to review. Close this dialog when you are ready.',false);
    weedModal.querySelector('.modal-close').focus();
  }
  function advanceAfterReview(){
    /* The reviewed row is already gone from the table, so rebuilding the
       groups from the DOM reflows the indices for us: the next weed in this
       image lands on the current index, and if the image emptied out, the
       next image slides into the current image index. */
    const imageId=weedData?String(weedData.imageId):null;
    const previousWeedIndex=weedIndex, previousImageIndex=weedImageIndex;
    refreshWeedNavigation();
    const sameImage=weedImageGroups.find(function(group){
      const data=parseWeedViewer(group[0]);
      return data&&String(data.imageId)===imageId;
    });
    const target=sameImage
      ? sameImage[Math.min(previousWeedIndex,sameImage.length-1)]
      : (weedImageGroups[previousImageIndex]||weedImageGroups[previousImageIndex-1]||[])[0];
    if(!target){showWeedEmptyState(); return;}
    openWeedModal(parseWeedViewer(target),weedReturnFocus,true);
  }
  function showWeedGuess(data){
    /* Describes what the verifier thinks the object is, which is what a
       reviewer actually needs on a borderline detection. It never contradicts
       the accept/reject gate -- that is the confidence figure above. */
    const guess=(data&&data.verifierGuess)||[];
    if(!guess.length){weedGuess.hidden=true; weedGuess.textContent=''; return;}
    weedGuess.textContent='Verifier best guess: '+guess.map(function(entry){
      return entry.label+' '+Math.round(entry.probability*100)+'%';
    }).join(' · ');
    weedGuess.hidden=false;
  }
  function openWeedModal(data,trigger,keepFocus){
    weedData=data;
    if(!keepFocus) weedReturnFocus=trigger;
    setWeedMessage('',true);
    weedActionButtons.forEach(function(button){button.disabled=false;});
    [weedWithoutOverlay,weedWithOverlay].forEach(function(button){button.disabled=false;});
    refreshWeedNavigation();
    weedImageIndex=weedImageGroups.findIndex(function(group){
      return group.some(function(viewer){
        const viewerData=parseWeedViewer(viewer);
        return viewerData&&String(viewerData.detectionId)===String(data.detectionId);
      });
    });
    if(weedImageIndex<0) weedImageIndex=0;
    const group=weedImageGroups[weedImageIndex]||[];
    weedIndex=Math.max(0,group.findIndex(function(viewer){
      const viewerData=parseWeedViewer(viewer);
      return viewerData&&String(viewerData.detectionId)===String(data.detectionId);
    }));
    if(hasWeedGeometry(data)){
      weedMarker.style.left=(data.x/data.width*100)+'%';
      weedMarker.style.top=(data.y/data.height*100)+'%';
    }
    showWeedView(weedViewMode);
    const others=Math.max(0,(data.siblings||[]).length-1);
    weedDetails.textContent='Area '+data.areaMm2.toFixed(1)+' mm² · confidence '+data.confidence.toFixed(2)
      +' · '+(data.observations||1)+' independent look(s)'
      +(data.verifierConfidence!=null?(' · verifier '+data.verifierConfidence.toFixed(2)):'')
      +(others?(' · '+others+' other weed(s) in this image'):'');
    showWeedGuess(data);
    updateWeedNavigation();
    /* Only grab focus when the dialog first appears. Advancing after a label
       must leave focus on the button that was pressed, so the reviewer can
       keep pressing Enter -- and never lands on Close by surprise. */
    const wasHidden=weedModal.hidden;
    weedModal.hidden=false;
    if(wasHidden) weedModal.querySelector('.modal-close').focus();
  }
  async function postWeedAction(id,action){
    try{
      const response=await fetch('weeds/'+id+'/'+action,{method:'POST',headers:{Accept:'application/json'}});
      const result=await response.json().catch(function(){return {};});
      const ok=response.ok&&(result.status==='applied'||result.status==='rejected'
        ||result.status==='dismissed');
      if(ok){const row=document.getElementById('weed-'+id); if(row) row.remove();}
      return {ok:ok,result:result};
    }catch(error){return {ok:false,result:{message:'Request failed: '+error.message}};}
  }
  async function reviewCurrentWeed(button,action,failureMessage){
    if(!weedData) return;
    button.disabled=true;
    let advanced=false;
    try{
      const result=await postWeedAction(weedData.detectionId,action);
      /* Advance rather than close: reviewing a long list should not cost a
         dialog dismissal and a fresh "View" click for every single weed. */
      if(result.ok){advanced=true; advanceAfterReview();}
      else setWeedMessage(result.result.message||failureMessage,true);
    }finally{
      /* advanceAfterReview() owns the button state from here -- re-enabled for
         the next weed, or left disabled once the queue is empty. */
      if(!advanced) button.disabled=false;
    }
  }
  weedAccept.addEventListener('click',function(){
    reviewCurrentWeed(weedAccept,'approve','Could not accept weed');
  });
  weedUnknown.addEventListener('click',function(){
    reviewCurrentWeed(weedUnknown,'dismiss','Could not discard weed');
  });
  weedRejectButtons.forEach(function(button){
    button.addEventListener('click',function(){
      reviewCurrentWeed(button,'label/'+button.dataset.weedLabel,'Could not reject weed');
    });
  });
  weedAcceptAll.addEventListener('click',async function(){
    if(!weedData) return;
    weedAcceptAll.disabled=true;
    let advanced=false;
    try{
      refreshWeedNavigation();
      const ids=weedViewers.map(function(viewer){
        const data=parseWeedViewer(viewer);
        return data&&String(data.imageId)===String(weedData.imageId)?data.detectionId:null;
      }).filter(Boolean);
      if(!ids.length) ids.push(weedData.detectionId);
      const response=await fetch('weeds/accept-all',{
        method:'POST',headers:{Accept:'application/json','Content-Type':'application/json'},
        body:JSON.stringify({detection_ids:[...new Set(ids)]})
      });
      const result=await response.json().catch(function(){return {};});
      const acceptedIds=result.accepted_ids||[];
      const failedIds=result.failed_ids||[];
      if(response.ok&&(result.status==='applied'||result.status==='partial')){
        acceptedIds.forEach(function(id){
          const row=document.getElementById('weed-'+id); if(row) row.remove();
        });
        if(!failedIds.length){advanced=true; advanceAfterReview();}
        else{
          refreshWeedNavigation(); updateWeedNavigation();
          setWeedMessage(failedIds.length+' weed(s) could not be accepted: '
            +(result.message||'check the individual detections'),true);
        }
      }else setWeedMessage(result.message||'Could not accept weeds',true);
    }finally{if(!advanced) weedAcceptAll.disabled=false;}
  });
  weedPrevious.addEventListener('click',function(){
    const group=weedImageGroups[weedImageIndex]||[], target=group[weedIndex-1];
    if(target) openWeedModal(parseWeedViewer(target),weedReturnFocus,true);
  });
  weedNext.addEventListener('click',function(){
    const group=weedImageGroups[weedImageIndex]||[], target=group[weedIndex+1];
    if(target) openWeedModal(parseWeedViewer(target),weedReturnFocus,true);
  });
  weedPreviousImage.addEventListener('click',function(){
    const target=weedImageGroups[weedImageIndex-1];
    if(target&&target.length) openWeedModal(parseWeedViewer(target[0]),weedReturnFocus,true);
  });
  weedNextImage.addEventListener('click',function(){
    const target=weedImageGroups[weedImageIndex+1];
    if(target&&target.length) openWeedModal(parseWeedViewer(target[0]),weedReturnFocus,true);
  });
  weedWithoutOverlay.addEventListener('click',function(){showWeedView('clean');});
  weedWithOverlay.addEventListener('click',function(){showWeedView('overlay');});
  weedCloseUp.addEventListener('click',function(){showWeedView('closeup');});
  weedZoom.addEventListener('input',function(){if(weedData&&weedViewMode==='closeup') applyCloseUp();});
  document.getElementById('weed-modal-close').addEventListener('click',closeWeedModal);
  weedModal.addEventListener('click',function(event){if(event.target===weedModal) closeWeedModal();});
  plantPhotoTab.addEventListener('click',function(){showPlantPhotoTab(true);});
  plantDiagnosticTab.addEventListener('click',function(){showPlantPhotoTab(false);});
  document.addEventListener('click',async function(event){
    const weedViewer=event.target.closest('.weed-view');
    if(weedViewer){
      const data=parseWeedViewer(weedViewer);
      if(data) openWeedModal(data,weedViewer);
      return;
    }
    const plantViewer=event.target.closest('[data-composite-clean]');
    const artifactViewer=plantViewer?null:event.target.closest('[data-artifacts]');
    const trigger=plantViewer||artifactViewer;
    if(trigger){
      if(plantViewer){
        plantComposite={
          clean:plantViewer.dataset.compositeClean,
          overlay:plantViewer.dataset.compositeOverlay||null
        };
        artifactControls.hidden=true;
        overlayLegend.textContent='White cross = known center; cyan = current radius; red = proposed radius. Neighbour labels and circles stay fixed; diagnostic mode only adds the target mask.';
        if(plantComposite.clean) showPlantComposite(false); else modalImg.removeAttribute('src');
      } else {
        try{artifacts=JSON.parse(artifactViewer.dataset.artifacts||'[]');}catch(_){artifacts=[];}
        index=0;
        artifactControls.hidden=!artifacts.length;
        overlayLegend.textContent='Cyan circle = original radius; red circle = planned radius.';
        if(artifacts.length) showArtifact(); else modalImg.removeAttribute('src');
      }
      returnFocus=trigger;
      let details={}; try{details=JSON.parse(trigger.dataset.details||'{}');}catch(_){}
      modalDetails.textContent=details.formula||'';
      modal.hidden=false; closeButton.focus();
      if(plantViewer&&plantComposite){
        plantPhotoMode.hidden=false;
        showPlantPhotoTab(true);
      } else {
        plantPhotoMode.hidden=true;
        showPlantPhotoTab(false);
      }
      return;
    }
    const action=event.target.closest('.review-action');
    if(action){
      event.preventDefault();
      const row=action.closest('.review-item');
      const message=row&&row.querySelector('.action-message');
      action.disabled=true;
      try{
        const response=await fetch(action.dataset.url,{method:'POST',headers:{Accept:'application/json'}});
        const result=await response.json();
        const explicitReject=/\/(reject|keep)$/.test(action.dataset.url);
        if(response.ok&&(result.status==='applied'||(result.status==='rejected'&&explicitReject))) row.remove();
        else if(message) message.textContent=result.message||('HTTP '+response.status);
      }catch(error){if(message) message.textContent='Request failed: '+error.message;}
      finally{action.disabled=false;}
      return;
    }
    const curveAction=event.target.closest('.curve-action');
    if(curveAction){
      event.preventDefault();
      const row=curveAction.closest('.review-item');
      const message=row.querySelector('.action-message');
      const data=new FormData();
      if(curveAction.dataset.action==='apply'){
        const input=row.querySelector('.curve-value'); data.append('value',input.value);
        if(!window.confirm('Apply this curve value? Flagged values override the automatic gate.')) return;
        data.append('confirm_override','true');
      }
      curveAction.disabled=true;
      try{
        const response=await fetch(curveAction.dataset.url,{method:'POST',headers:{Accept:'application/json'},body:data});
        const result=await response.json();
        if(response.ok&&(result.status==='applied'||result.status==='rejected')) row.remove();
        else message.textContent=result.message||('HTTP '+response.status);
      }catch(error){message.textContent='Request failed: '+error.message;}
      finally{curveAction.disabled=false;}
    }
  });
  closeButton.addEventListener('click',closeModal);
  modal.addEventListener('click',function(event){if(event.target===modal) closeModal();});
  document.getElementById('overlay-modal-prev').addEventListener('click',function(){
    index=(index-1+artifacts.length)%artifacts.length;showArtifact();
  });
  document.getElementById('overlay-modal-next').addEventListener('click',function(){
    index=(index+1)%artifacts.length;showArtifact();
  });
  document.addEventListener('keydown',function(event){
    if(event.key!=='Escape') return;
    if(!modal.hidden) closeModal();
    if(!weedModal.hidden) closeWeedModal();
    if(!gantryModal.hidden) gantryModal.hidden=true;
  });
  document.getElementById('queue-open').addEventListener('click',function(){
    const to=new Date(), from=new Date(to.getTime()-72*60*60*1000);
    function localValue(value){
      const shifted=new Date(value.getTime()-value.getTimezoneOffset()*60000);
      return shifted.toISOString().slice(0,16);
    }
    if(!document.getElementById('queue-to').value){
      document.getElementById('queue-from').value=localValue(from);
      document.getElementById('queue-to').value=localValue(to);
    }
    queueModal.hidden=false;loadQueueImages();
  });
  document.getElementById('queue-close').addEventListener('click',function(){queueModal.hidden=true;});
  if(gantryOpen) gantryOpen.addEventListener('click',function(){
    gantryModal.hidden=false;
    gantryModal.querySelector('.modal-close').focus();
  });
  document.getElementById('gantry-close').addEventListener('click',function(){
    gantryModal.hidden=true;
  });
  gantryModal.addEventListener('click',function(event){
    if(event.target===gantryModal) gantryModal.hidden=true;
  });
  if(photoGridOpen) photoGridOpen.addEventListener('click',function(){
    photoGridModal.hidden=false;loadPhotoGrid();
    photoGridModal.querySelector('.modal-close').focus();
  });
  document.getElementById('photo-grid-close').addEventListener('click',function(){
    photoGridModal.hidden=true;
  });
  photoGridModal.addEventListener('click',function(event){
    if(event.target===photoGridModal) photoGridModal.hidden=true;
  });
  document.getElementById('queue-refresh').addEventListener('click',loadQueueImages);
  document.getElementById('queue-select-all').addEventListener('change',function(){
    document.querySelectorAll('.queue-checkbox').forEach(box=>box.checked=this.checked);
  });
  document.getElementById('queue-add').addEventListener('click',async function(){
    const ids=[...document.querySelectorAll('.queue-checkbox:checked')].map(box=>+box.value);
    if(!ids.length){queueMessage.textContent='Select at least one image';return;}
    const response=await fetch('analysis/queue',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image_ids:ids})});
    const data=await response.json();
    if(response.ok){document.getElementById('queue-count').textContent=data.queue_length;
      queueMessage.textContent=ids.length+' images added';}
    else queueMessage.textContent=data.detail||'Could not add images';
  });
})();
"""

# Boundaries and exclusion zones. The add form only shows the geometry fields of
# the selected shape, permissions start at the sensible polarity for the chosen
# kind, and a top-down garden map draws every zone (optionally with the bot's
# plants and weeds) so a zone can be checked before it starts gating writes.
_ZONES_JS = r"""
(function(){
  const shape=document.getElementById('shape');
  const kind=document.getElementById('kind');
  const canvas=document.getElementById('zone-map');
  const ctx=canvas.getContext('2d');
  const status=document.getElementById('zone-map-status');
  let zones=[]; try{zones=JSON.parse(canvas.dataset.zones||'[]');}catch(_){zones=[];}
  let items={plants:[],weeds:[]};

  function showShapeFields(){
    ['rectangle','circle','polygon'].forEach(function(name){
      const box=document.getElementById('fields-'+name);
      const active=(shape.value===name);
      box.hidden=!active;
      box.querySelectorAll('input,textarea').forEach(function(field){field.disabled=!active;});
    });
  }
  // Boundaries usually permit everything inside; exclusion zones usually
  // forbid everything. Both stay editable afterwards.
  function applyKindDefaults(){
    const allow=(kind.value==='boundary');
    ['allow_weeds','allow_plant_centers','allow_plant_radius'].forEach(function(name){
      document.getElementById('new_'+name).checked=allow;
    });
  }
  function zonePoints(zone){
    if(zone.shape==='rectangle')
      return [[zone.min_x,zone.min_y],[zone.max_x,zone.min_y],
              [zone.max_x,zone.max_y],[zone.min_x,zone.max_y]];
    if(zone.shape==='circle')
      return [[zone.center_x-zone.radius_mm,zone.center_y-zone.radius_mm],
              [zone.center_x+zone.radius_mm,zone.center_y+zone.radius_mm]];
    return zone.points||[];
  }
  function bounds(){
    let xs=[],ys=[];
    zones.forEach(function(zone){zonePoints(zone).forEach(function(p){xs.push(p[0]);ys.push(p[1]);});});
    items.plants.concat(items.weeds).forEach(function(p){xs.push(p.x);ys.push(p.y);});
    if(!xs.length) return null;
    let minX=Math.min.apply(null,xs), maxX=Math.max.apply(null,xs);
    let minY=Math.min.apply(null,ys), maxY=Math.max.apply(null,ys);
    const padX=Math.max(50,(maxX-minX)*0.08), padY=Math.max(50,(maxY-minY)*0.08);
    return {minX:minX-padX,maxX:maxX+padX,minY:minY-padY,maxY:maxY+padY};
  }
  function drawZone(zone,project,scale){
    const boundary=(zone.kind==='boundary');
    ctx.save();
    ctx.setLineDash(zone.enabled?[]:[6,4]);
    ctx.lineWidth=2;
    ctx.strokeStyle=boundary?'#2ecc40':'#ff4136';
    ctx.fillStyle=boundary?'rgba(46,204,64,.12)':'rgba(255,65,54,.16)';
    ctx.beginPath();
    if(zone.shape==='circle'){
      const c=project(zone.center_x,zone.center_y);
      ctx.arc(c[0],c[1],Math.max(2,zone.radius_mm*scale),0,Math.PI*2);
    } else {
      const pts=zonePoints(zone);
      pts.forEach(function(p,i){
        const c=project(p[0],p[1]);
        if(i===0) ctx.moveTo(c[0],c[1]); else ctx.lineTo(c[0],c[1]);
      });
      ctx.closePath();
    }
    ctx.fill();ctx.stroke();
    const label=zonePoints(zone)[0]||[zone.center_x,zone.center_y];
    const anchor=project(zone.shape==='circle'?zone.center_x:label[0],
                         zone.shape==='circle'?zone.center_y:label[1]);
    ctx.setLineDash([]);
    ctx.fillStyle='#17221b';ctx.font='12px system-ui';
    ctx.fillText(zone.name+(zone.enabled?'':' (off)'),anchor[0]+6,anchor[1]-6);
    ctx.restore();
  }
  function render(){
    const box=bounds();
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#fbfdfb';ctx.fillRect(0,0,canvas.width,canvas.height);
    if(!box){
      ctx.fillStyle='#74817a';ctx.font='14px system-ui';
      ctx.fillText('Add a zone to see the garden map',14,26);
      return;
    }
    const scale=Math.min(canvas.width/(box.maxX-box.minX),canvas.height/(box.maxY-box.minY));
    const project=function(x,y){return [(x-box.minX)*scale,(y-box.minY)*scale];};
    zones.forEach(function(zone){drawZone(zone,project,scale);});
    items.plants.forEach(function(plant){
      const c=project(plant.x,plant.y);
      ctx.strokeStyle='#1a7f4b';ctx.lineWidth=1.5;
      ctx.beginPath();ctx.arc(c[0],c[1],Math.max(3,(plant.radius||0)*scale),0,Math.PI*2);ctx.stroke();
      ctx.fillStyle='#1a7f4b';ctx.beginPath();ctx.arc(c[0],c[1],2.5,0,Math.PI*2);ctx.fill();
    });
    items.weeds.forEach(function(weed){
      const c=project(weed.x,weed.y);
      ctx.fillStyle='#b3002d';ctx.beginPath();ctx.arc(c[0],c[1],3,0,Math.PI*2);ctx.fill();
    });
    ctx.fillStyle='#74817a';ctx.font='12px system-ui';
    ctx.fillText('X '+Math.round(box.minX)+'…'+Math.round(box.maxX)+' mm, Y '
      +Math.round(box.minY)+'…'+Math.round(box.maxY)+' mm (Y increases downwards)',10,
      canvas.height-10);
  }
  document.getElementById('zone-load-items').addEventListener('click',async function(){
    const entry=canvas.dataset.entry||'';
    if(!entry){status.textContent='Select a FarmBot in the app options first';return;}
    status.textContent='Loading plants and weeds…';
    try{
      const response=await fetch('api/vision/images?entry_id='+encodeURIComponent(entry));
      if(!response.ok) throw new Error('HTTP '+response.status);
      const data=await response.json();
      items={plants:data.plants||[],weeds:data.weeds||[]};
      status.textContent=items.plants.length+' plants and '+items.weeds.length+' FarmBot weeds shown';
      render();
    }catch(err){status.textContent='Could not load garden items: '+err.message;}
  });
  shape.addEventListener('change',showShapeFields);
  kind.addEventListener('change',applyKindDefaults);
  showShapeFields();applyKindDefaults();render();
})();
"""


async def event_listener() -> None:
    async for event in client.vision_events():
        # Grid images are quality-gated as one completed set. In particular,
        # a washed-out frame must be deleted and excluded before any new-image
        # event can derive analysis from it.
        active_grid = photo_grid_task
        if event.image_id is not None and active_grid is not None and not active_grid.done():
            await asyncio.shield(active_grid)
            record = photo_grid_store.load()
            if (
                record is not None
                and record.config_entry_id == event.config_entry_id
                and event.image_id in record.excluded_image_ids
            ):
                LOGGER.info(
                    "Skipped analysis event for discarded photo-grid image %d",
                    event.image_id,
                )
                continue
        # Await each automatic request so photos cannot be silently discarded
        # merely because the previous image is still being analysed.
        await jobs.run(
            entry_id=event.config_entry_id,
            mode=OperatingMode(event.mode) if event.mode is not None else settings.mode,
            plant_ids=event.plant_ids,
            image_ids=[event.image_id] if event.image_id is not None else None,
            trigger="new_image" if event.image_id is not None else "event",
            queue_if_busy=True,
        )


async def heartbeat() -> None:
    while True:
        if settings.selected_config_entry_id:
            if jobs.lock.locked():
                try:
                    job_id = UUID(str(jobs.current.get("id")))
                except (TypeError, ValueError):
                    job_id = None
                await jobs._status(
                    settings.selected_config_entry_id,
                    job_id,
                    "running",
                    str(jobs.current.get("progress") or "analysing")[:240],
                )
            else:
                await jobs._status(settings.selected_config_entry_id, None, "idle", "ready")
        # Older installations may retain the former 15-minute option. Cap the
        # effective interval so they also stay inside the integration's
        # ten-minute availability window after upgrading.
        await asyncio.sleep(min(settings.heartbeat_minutes, 5) * 60)


async def resolve_config_entry() -> None:
    """Select the only loaded FarmBot automatically when no ID was configured."""
    if settings.selected_config_entry_id:
        return
    try:
        bots = (await client.list_bots()).bots
    except HomeAssistantError as exc:
        LOGGER.warning("Could not discover FarmBot config entries at startup: %s", exc)
        return
    if len(bots) == 1:
        settings.selected_config_entry_id = bots[0].config_entry_id
        LOGGER.info(
            "Automatically selected the only loaded FarmBot config entry: %s",
            settings.selected_config_entry_id,
        )
    elif len(bots) > 1:
        LOGGER.warning(
            "Multiple FarmBots are loaded; select one in the add-on options to enable heartbeats"
        )


async def scheduler() -> None:
    last_run_date = None
    last_repair_completed_at: datetime | None = None
    while True:
        now = datetime.now().astimezone()
        if (
            settings.schedule_enabled
            and settings.selected_config_entry_id
            and now.strftime("%H:%M") == settings.schedule_time
            and now.date() != last_run_date
            and database.active_calibration(settings.selected_config_entry_id)
        ):
            last_run_date = now.date()
            await jobs.run(trigger="schedule")
        repair_settings = grid_repair_settings_store.load()
        if repair_settings.enabled and settings.selected_config_entry_id:
            # inspect_photo_grid() caches its inventory lookup for 5 minutes,
            # so polling every tick here does not add extra Home Assistant load.
            candidate_run = await inspect_photo_grid()
            if (
                candidate_run is not None
                and candidate_run.completed_at != last_repair_completed_at
                and now - candidate_run.completed_at.astimezone()
                >= timedelta(minutes=repair_settings.delay_minutes)
            ):
                last_repair_completed_at = candidate_run.completed_at
                try:
                    result = await start_photo_grid_repair()
                    LOGGER.info("Automatic photo-grid repair: %s", result.get("message"))
                except HomeAssistantError as exc:
                    LOGGER.warning("Automatic photo-grid repair failed to start: %s", exc)
        await asyncio.sleep(30)


async def retention_cleanup() -> None:
    while True:
        artifacts = settings.data_dir / "artifacts"
        now = datetime.now().astimezone()
        if artifacts.exists():
            for path in artifacts.glob("*"):
                days = (
                    settings.successful_mask_retention_days
                    if path.name.endswith("-mask.png")
                    else settings.diagnostic_retention_days
                )
                cutoff = now - timedelta(days=days)
                if datetime.fromtimestamp(path.stat().st_mtime).astimezone() < cutoff:
                    path.unlink(missing_ok=True)
        await asyncio.sleep(6 * 60 * 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await resolve_config_entry()
    LOGGER.info(
        "FarmBot Vision %s starting: selected_config_entry_id=%s mode=%s analysis_resolution=%s",
        __version__,
        settings.selected_config_entry_id or "(not set)",
        settings.mode.value,
        settings.resolution.label,
    )
    if not settings.selected_config_entry_id:
        LOGGER.warning(
            "No FarmBot config entry ID configured; scheduled/heartbeat status reports and "
            "the calibration page will not work until one is set in the add-on options"
        )
    seed_calibration_from_store()
    interrupted_grid = photo_grid_store.load()
    if interrupted_grid is not None and interrupted_grid.status in {
        "queued",
        "running",
        "retrying",
    }:
        interrupted_grid.status = "failed"
        interrupted_grid.completed_at = datetime.now(UTC)
        interrupted_grid.message = (
            "Photo grid was interrupted by an app restart; verified photos were preserved. "
            "Start a fresh grid to replace it."
        )
        photo_grid_store.save(interrupted_grid)
    tasks = [
        asyncio.create_task(event_listener(), name="event_listener"),
        asyncio.create_task(heartbeat(), name="heartbeat"),
        asyncio.create_task(scheduler(), name="scheduler"),
        asyncio.create_task(retention_cleanup(), name="retention_cleanup"),
    ]

    def _log_task_failure(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            LOGGER.error(
                "Background task %s crashed and will not restart: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    for task in tasks:
        task.add_done_callback(_log_task_failure)
    yield
    for task in tasks:
        task.cancel()
    if grid_repair_task is not None and not grid_repair_task.done():
        grid_repair_task.cancel()
        await asyncio.gather(grid_repair_task, return_exceptions=True)
    if photo_grid_task is not None and not photo_grid_task.done():
        photo_grid_task.cancel()
        await asyncio.gather(photo_grid_task, return_exceptions=True)
    await asyncio.gather(*tasks, return_exceptions=True)
    await soil_jobs.close()
    await client.close()


app = FastAPI(
    title="FarmBot Vision", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
)
app.add_middleware(NormalizeIngressPathMiddleware)


def ingress_base(request: Request) -> str:
    value = request.headers.get("X-Ingress-Path", "./").strip()
    if value in {"", ".", "./"}:
        return "./"
    value = _normalize_leading_slashes(value).rstrip("/")
    return f"{value}/"


def layout(request: Request, body: str, title: str = "FarmBot Vision") -> HTMLResponse:
    base = escape(ingress_base(request), quote=True)
    return HTMLResponse(
        f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><base href="{base}">
<title>{escape(title)}</title><style>
:root{{--green:#52b788;--dark:#17221b;--muted:#74817a}}*{{box-sizing:border-box}}
body{{font:15px system-ui;margin:0;background:#f3f7f4;color:var(--dark)}}header{{background:#173f2c;color:white;padding:1rem 4vw}}
main{{max-width:1100px;margin:auto;padding:1.2rem}}nav a{{color:white;margin-right:1rem}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem}}
.calibration-grid{{grid-template-columns:minmax(240px,320px) minmax(0,1fr)}}@media(max-width:760px){{.calibration-grid{{grid-template-columns:1fr}}}}
.card{{background:white;border-radius:10px;padding:1rem;box-shadow:0 1px 4px #0002;overflow:auto}}table{{width:100%;border-collapse:collapse}}td,th{{padding:.5rem;text-align:left;border-bottom:1px solid #ddd}}
button{{background:var(--green);border:0;border-radius:6px;padding:.65rem 1rem;cursor:pointer}}.warn{{color:#9b4b00}}.muted{{color:var(--muted)}}input,select{{padding:.5rem;max-width:100%}}img{{max-width:100%}}
.action-message{{display:block;color:#a40000;max-width:24rem}}.action-message.notice{{color:var(--muted)}}.overlay-modal[hidden]{{display:none}}
.overlay-modal{{position:fixed;inset:0;z-index:1000;background:#000b;display:flex;align-items:center;justify-content:center;padding:1rem}}
.overlay-modal figure{{position:relative;background:white;border-radius:10px;margin:0;padding:1rem;width:min(95vw,1000px);max-height:95vh;overflow:auto}}
.overlay-modal img{{display:block;width:100%;height:auto;max-height:72vh;object-fit:contain;margin:auto;background:#111}}.modal-close{{position:absolute;right:.5rem;top:.5rem;font-size:1.5rem;z-index:2}}
.overlay-modal canvas{{display:block;max-height:70vh;max-width:100%;margin:auto;background:#111;border-radius:6px}}
.gantry-gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;min-width:min(85vw,900px)}}
.gantry-gallery figure{{margin:0;padding:0;overflow:hidden}}.gantry-gallery img{{width:100%;height:180px;object-fit:contain;background:#111}}
.gantry-gallery figcaption{{padding:.5rem 0}}.gantry-target{{color:#a40000;font-weight:bold}}
.photo-grid-dialog{{width:min(96vw,1000px)}}.photo-grid-dialog canvas{{display:block;width:100%;max-height:72vh;
background:#243a2c;border:1px solid #bcc9c0;border-radius:6px}}
.photo-grid-card form{{margin:0}}
.calibration-grid-viewport{{width:100%;max-height:72vh;overflow:auto;background:#111;
border:1px solid #ccc;border-radius:6px}}.calibration-grid-viewport canvas{{display:block;
width:100%;height:auto;max-width:none}}
.modal-controls{{display:flex;gap:.5rem;align-items:center;justify-content:center;margin-top:.6rem}}.legend{{font-size:.9rem;color:var(--muted)}}
.modal-controls[hidden]{{display:none}}
.queue-dialog{{width:min(95vw,900px)}}.button-row{{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}}
.clear-actions{{margin-top:.5rem}}.clear-actions form{{margin:0}}
button.clear-button{{background:#a40000;color:white}}
.training-crop{{width:96px;height:96px;object-fit:contain;background:#181d19;border-radius:6px}}
.weed-dialog{{width:min(95vw,900px)}}.weed-navigation{{justify-content:space-between}}
.weed-actions fieldset{{border:1px solid #d5ded8;border-radius:6px;margin-top:.7rem}}
.weed-image-wrap{{position:relative;display:inline-block;margin:auto}}
.weed-image-wrap.closeup{{overflow:hidden;background:#111}}
.weed-image-wrap.closeup img{{transform-origin:0 0;image-rendering:auto}}
.weed-zoom[hidden]{{display:none}}.weed-zoom input[type=range]{{flex:1;max-width:16rem}}
button.unknown-button{{background:#e4ede7;color:var(--dark);box-shadow:inset 0 0 0 1px #b6c6bb}}
.weed-marker{{position:absolute;width:34px;height:34px;margin-left:-17px;margin-top:-17px;pointer-events:none;
border-radius:50%;border:3px solid #168cff;box-shadow:0 0 0 1px #001b3d}}
.weed-view-toggle button[aria-pressed=true]{{background:#1672c4;color:white;box-shadow:inset 0 0 0 2px #0b4779}}
.plant-view-toggle button[aria-pressed=true]{{background:#1672c4;color:white;box-shadow:inset 0 0 0 2px #0b4779}}
@media (max-width:600px){{.overlay-modal{{padding:.35rem;align-items:flex-start}}.overlay-modal figure{{width:99vw;max-width:99vw;padding:.55rem;margin-top:.35rem}}.overlay-modal img{{max-height:68vh}}.modal-controls{{flex-wrap:wrap}}}}
td.actions{{min-width:9rem}}.actions-group{{display:flex;flex-direction:column;align-items:stretch;gap:.4rem}}
.actions-group form{{margin:0}}.actions-group button{{width:100%;padding:.45rem .8rem;font-size:.9rem}}
.actions-group button[data-artifacts]{{background:#e4ede7;color:var(--dark)}}
.hint{{display:inline-flex;align-items:center;justify-content:center;width:1.1em;height:1.1em;
border-radius:50%;background:var(--muted);color:white;font-size:.72em;font-weight:bold;
margin-left:.3em;cursor:help;vertical-align:middle;line-height:1}}
</style></head><body><header><h1>🌱 FarmBot Vision</h1><nav><a href="./">Analysis</a><a href="soil-height">Soil height</a><a href="settings">Calibration</a><a href="weed-settings">Weed settings</a><a href="canopy-settings">Canopy fusion</a><a href="zones">Boundaries &amp; zones</a><a href="api/health">Health JSON</a></nav></header>
<main>{body}</main></body></html>"""
    )


def hint(text: str) -> str:
    """A small hover-tooltip badge ("?") explaining a nearby form field."""
    return f'<span class=hint tabindex=0 title="{escape(text, quote=True)}">?</span>'


@app.get("/health")
@app.get("/api/health")
async def health() -> JSONResponse:
    artifacts = settings.data_dir / "artifacts"
    artifact_bytes = (
        sum(p.stat().st_size for p in artifacts.glob("*") if p.is_file())
        if artifacts.exists()
        else 0
    )
    resolution = settings.resolution
    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "algorithm_version": ALGORITHM_VERSION,
            "contract_version": CONTRACT_VERSION,
            "minimum_integration_version": MINIMUM_INTEGRATION_VERSION,
            "opencv_threads": cv2.getNumThreads(),
            "analysis_resolution": resolution.value,
            "analysis_width": resolution.width,
            "analysis_height": resolution.height,
            "analysis_pixels": resolution.pixel_count,
            "relative_workload": resolution.relative_workload,
            "job": jobs.current,
            "last_job": jobs.last,
            "canopy_fusion": canopy_fusion_settings_store.load().model_dump(),
            "database": database.stats(),
            "artifact_bytes": artifact_bytes,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    grid_run = await inspect_photo_grid()
    photo_grid_record = photo_grid_store.load()
    repair_values = grid_repair_settings_store.load()
    rows = database.pending_measurements()
    crop_slugs = sorted({row["crop_slug"] for row in rows})
    curves = {
        slug: fit_monotonic_curve(
            database.measurements_for_crop(slug), safety_margin_mm=settings.safety_margin_mm
        )
        for slug in crop_slugs
    }

    def _artifact_button(r: dict) -> str:
        center_x = r.get("recorded_center_x")
        center_y = r.get("recorded_center_y")
        center = (
            f"({float(center_x):.1f}, {float(center_y):.1f})"
            if center_x is not None and center_y is not None
            else "(unavailable)"
        )
        details = {
            "formula": (
                f"Current {r['current_radius_mm']:.1f} mm; Recommended = "
                f"{r['recommended_protection_radius_mm']:.1f} mm. "
                f"Plant center = {center}; crop: {r.get('crop_slug', 'unknown')}. "
                f"Evidence: {int(r.get('used_measurement_count', 1))} used, "
                f"{int(r.get('useful_measurement_count', 1))} useful, "
                f"{int(r.get('measurement_count', 1))} candidates; "
                f"visible outer boundary "
                f"{float(r.get('visible_boundary_coverage', r.get('boundary_coverage', 0)) or 0):.0%}. "
                + (
                    f"Fused from {r.get('fusion_view_count', 0)} calibrated views; "
                    f"angular coverage {float(r.get('fusion_angular_coverage') or 0):.0%}; "
                    f"corroborated pixels "
                    f"{float(r.get('fusion_corroborated_fraction') or 0):.0%}."
                    if r.get("fused_canopy")
                    else "Measured from one image or consolidated per-image radii."
                )
            )
        }
        details_json = escape(json.dumps(details, separators=(",", ":")), quote=True)
        plant_attrs = ""
        if r.get("plant_id") is not None and center_x is not None and center_y is not None:
            plant_entry_id = r.get("config_entry_id") or settings.selected_config_entry_id
            unlinked = sorted(database.unlinked_image_ids(plant_entry_id, int(r["plant_id"])))
            unlinked_json = escape(json.dumps(unlinked, separators=(",", ":")), quote=True)
            plant_attrs = (
                f' data-plant-id="{int(r["plant_id"])}" '
                f'data-plant-x="{float(center_x)}" data-plant-y="{float(center_y)}" '
                f'data-plant-radius="{float(r.get("current_radius_mm") or 0)}" '
                f'data-unlinked-images="{unlinked_json}"'
            )
        if r.get("composite_path"):
            clean_url = escape(
                f"artifact/{Path(r['composite_path']).name}",
                quote=True,
            )
            overlay_url = (
                escape(
                    f"artifact/{Path(r['composite_overlay_path']).name}",
                    quote=True,
                )
                if r.get("composite_overlay_path")
                else ""
            )
            composite_button = (
                f'<button type=button data-composite-clean="{clean_url}" '
                f'data-composite-overlay="{overlay_url}" '
                f'data-details="{details_json}"{plant_attrs}>View</button>'
            )
            if r.get("fusion_diagnostic_path"):
                fusion_url = escape(
                    json.dumps(
                        [f"artifact/{Path(r['fusion_diagnostic_path']).name}"],
                        separators=(",", ":"),
                    ),
                    quote=True,
                )
                composite_button += (
                    f'<button type=button data-artifacts="{fusion_url}" '
                    f'data-details="{details_json}">Fusion</button>'
                )
            return composite_button
        paths = r.get("artifact_paths") or []
        if not paths and r.get("overlay_path"):
            paths = [r["overlay_path"]]
        if r.get("fusion_diagnostic_path"):
            paths = [*paths, r["fusion_diagnostic_path"]]
        urls = [f"artifact/{Path(path).name}" for path in paths if path]
        if not urls:
            if not plant_attrs:
                return "<span class=muted>None</span>"
            return (
                f'<button type=button data-artifacts="[]" '
                f'data-details="{details_json}"{plant_attrs}>View</button>'
            )
        artifacts_json = escape(json.dumps(urls, separators=(",", ":")), quote=True)
        return (
            f'<button type=button data-artifacts="{artifacts_json}" '
            f'data-details="{details_json}"{plant_attrs}>View</button>'
        )

    def _format_center(
        x: object,
        y: object,
        *,
        fallback: str = "Unavailable for older result",
    ) -> str:
        if x is None or y is None:
            return f"<span class=muted>{escape(fallback)}</span>"
        try:
            return f"X {float(x):.1f}, Y {float(y):.1f}"
        except (TypeError, ValueError):
            return f"<span class=muted>{escape(fallback)}</span>"

    def _format_coordinates(x: object, y: object) -> str:
        if x is None or y is None:
            return "<span class=muted>Unavailable</span>"
        return f"({float(x):.1f}, {float(y):.1f})"

    def _review_controls(r: dict) -> str:
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if not r.get("calibrated", 1):
            return (
                "<span class=warn>Calibration required to apply a radius</span>"
                f'<form method=post action="recommendations/{r["measurement_id"]}/reject">'
                f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/reject">'
                "Reject</button></form><small class=action-message></small>"
            )
        if r["recommended_protection_radius_mm"] > r["current_radius_mm"]:
            approve_label = "Apply radius"
        elif r["recommended_protection_radius_mm"] < r["current_radius_mm"]:
            approve_label = "Apply smaller radius"
        else:
            approve_label = "Approve observation"
        return (
            f'<form method=post action="recommendations/{r["measurement_id"]}/approve">'
            f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/approve">'
            f"{approve_label}</button></form>"
            f'<form method=post action="recommendations/{r["measurement_id"]}/reject">'
            f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/reject">Reject</button></form>'
            + (
                f'<form method=post action="recommendations/{r["measurement_id"]}/move-center">'
                f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/move-center">'
                "Move center</button></form>"
                if r.get("center_misaligned")
                else ""
            )
            + "<small class=action-message></small>"
        )

    measurement_rows = "".join(
        f'<tr class=review-item id="measurement-{r["measurement_id"]}"><td>{escape(r["crop_slug"])}</td>'
        f"<td>{_format_coordinates(r.get('recorded_center_x'), r.get('recorded_center_y'))}</td>"
        f"<td>{r['current_radius_mm']:.1f}</td>"
        f"<td>{r['maximum_accepted_canopy_radius_mm']:.1f}</td><td>{r['recommended_protection_radius_mm']:.1f}</td>"
        f"<td>{r['confidence']:.2f}</td>"
        f"<td>{escape(r['decision'])}</td><td>{escape(r['reason'])}</td>"
        f"<td class=actions><div class=actions-group>{_artifact_button(r)}{_review_controls(r)}</div></td></tr>"
        for r in rows
        if not r.get("vegetation_absent")
    )
    last = jobs.last
    curve_rows = "".join(
        f"<tr><td>{escape(slug)}</td><td>{escape(str(curve))}</td><td>diameter mm</td></tr>"
        for slug, curve in curves.items()
    )

    removal_rows = "".join(
        f'<tr class=review-item id="measurement-{r["measurement_id"]}">'
        f"<td>{escape(r['crop_slug'])}</td>"
        f"<td>{_format_center(r.get('recorded_center_x'), r.get('recorded_center_y'))}</td>"
        f"<td>{_format_center(r.get('recommended_center_x'), r.get('recommended_center_y'), fallback='No move suggested')}</td>"
        f"<td>{r['absent_observations']}</td><td>{r['confidence']:.2f}</td>"
        f"<td>{escape(r['reason'])}</td><td>{_artifact_button(r)}</td><td>"
        f'<form method=post action="removals/{r["measurement_id"]}/approve"><button class=review-action '
        f'data-url="removals/{r["measurement_id"]}/approve">Approve removal</button></form>'
        f'<form method=post action="removals/{r["measurement_id"]}/keep"><button class=review-action '
        f'data-url="removals/{r["measurement_id"]}/keep">Keep plant</button></form>'
        + (
            f'<form method=post action="removals/{r["measurement_id"]}/move-center"><button class=review-action '
            f'data-url="removals/{r["measurement_id"]}/move-center">Move center</button></form>'
            if r.get("center_misaligned")
            else ""
        )
        + "<small class=action-message></small></td></tr>"
        for r in rows
        if r.get("vegetation_absent")
    )
    proposal_rows = []
    for proposal in database.curve_proposals():
        previous = json.loads(proposal["previous_data_json"] or "{}")
        proposed = json.loads(proposal["data_json"] or "{}")
        day = int(proposal["plant_age_days"])
        value = float(proposed.get(str(day), 0))
        diagnostic = _artifact_button(
            {
                "artifact_paths": [proposal["overlay_path"]] if proposal["overlay_path"] else [],
                "current_radius_mm": value / 2,
                "typical_canopy_radius_mm": value / 2,
                "maximum_accepted_canopy_radius_mm": value / 2,
                "recommended_protection_radius_mm": value / 2,
                "safety_margin_mm": 0,
                "calibration_uncertainty_mm": 0,
            }
        )
        proposal_rows.append(
            f'<tr class=review-item id="curve-proposal-{proposal["id"]}"><td>{proposal["plant_id"]}</td>'
            f"<td>{escape(str(previous))}</td><td>day {day}: "
            f'<input class=curve-value form="curve-apply-{proposal["id"]}" name=value '
            f'type=number min=0 step=any value="{value:g}"> mm diameter</td>'
            f"<td>{escape(proposal['reason'] or '')}; old conflict "
            f"day {escape(str(proposal['conflict_day']))} = {escape(str(proposal['conflict_old_diameter']))}</td>"
            f"<td>{diagnostic}</td><td>"
            f'<form id="curve-apply-{proposal["id"]}" method=post action="curve-proposals/{proposal["id"]}/apply">'
            "<input type=hidden name=confirm_override value=true>"
            f'<button class=curve-action data-action=apply data-url="curve-proposals/{proposal["id"]}/apply">Use value</button></form>'
            f'<form method=post action="curve-proposals/{proposal["id"]}/discard-new"><button class=curve-action '
            f'data-action=discard-new data-url="curve-proposals/{proposal["id"]}/discard-new">Discard new</button></form>'
            f'<form method=post action="curve-proposals/{proposal["id"]}/discard-old"><button class=curve-action '
            f'data-action=discard-old data-url="curve-proposals/{proposal["id"]}/discard-old">Discard old</button></form>'
            "<small class=action-message></small></td></tr>"
        )
    flagged_curve_rows = "".join(proposal_rows)
    decision_rows = "".join(
        f"<tr><td>{escape(row['created_at'])}</td><td>{escape(row['measurement_id'])}</td>"
        f"<td>{escape(row['action'])}</td></tr>"
        for row in database.recent_decisions()
    )
    pending_weeds = database.pending_weed_detections()
    weeds_by_image: dict[int, list[dict]] = {}
    for w in pending_weeds:
        weeds_by_image.setdefault(w["image_id"], []).append(w)

    def _verifier_guess(w: dict) -> list[dict[str, object]]:
        try:
            return _verifier_label_guess(json.loads(w.get("features_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            return []

    def _weed_view_button(w: dict) -> str:
        if not w.get("overlay_path"):
            return "<span class=muted>None</span>"
        siblings = [str(other["detection_id"]) for other in weeds_by_image.get(w["image_id"], [])]
        marker = {
            "overlayArtifact": f"artifact/{Path(w['overlay_path']).name}",
            "reviewArtifact": (
                f"artifact/{Path(w['review_path']).name}" if w.get("review_path") else None
            ),
            "imageId": w["image_id"],
            "x": w.get("center_px_x"),
            "y": w.get("center_px_y"),
            "width": w.get("processed_width"),
            "height": w.get("processed_height"),
            "detectionId": str(w["detection_id"]),
            "siblings": siblings,
            "areaMm2": w["area_mm2"],
            "confidence": w["confidence"],
            "observations": w.get("observation_count", 1),
            "verifierConfidence": w.get("verifier_confidence"),
            "verifierGuess": _verifier_guess(w),
        }
        marker_json = escape(json.dumps(marker, separators=(",", ":")), quote=True)
        return f'<button type=button class=weed-view data-weed="{marker_json}">View</button>'

    def _weed_row(w: dict) -> str:
        verifier = (
            f"{float(w['verifier_confidence']):.2f}"
            if w.get("verifier_confidence") is not None
            else "—"
        )
        return (
            f'<tr class=review-item id="weed-{w["detection_id"]}"><td>{w["image_id"]}</td>'
            f"<td>{w['x']:.1f}, {w['y']:.1f}, {w['z']:.1f}</td>"
            f"<td>{w['area_mm2']:.1f}</td><td>{w.get('observation_count', 1)}</td>"
            f"<td>{float(w.get('heuristic_confidence') or w['confidence']):.2f}</td>"
            f"<td>{verifier}</td><td>{_weed_view_button(w)}</td></tr>"
        )

    weed_rows = "".join(_weed_row(w) for w in pending_weeds)
    resolution = settings.resolution
    if photo_grid_record is None:
        photo_grid_summary = "No calibrated grid has been captured yet"
        photo_grid_details = "Save calibration, then start the first reliable whole-bed grid."
        photo_grid_view_disabled = " disabled"
    else:
        photo_grid_summary = escape(photo_grid_record.message or photo_grid_record.status)
        photo_grid_details = (
            f"{len(photo_grid_record.frames)} of {len(photo_grid_record.targets)} verified photos"
            f" · started "
            f"{escape(photo_grid_record.started_at.astimezone().strftime('%d %b %Y %H:%M'))}"
        )
        photo_grid_view_disabled = "" if photo_grid_record.frames else " disabled"
    photo_grid_running = photo_grid_task is not None and not photo_grid_task.done()
    photo_grid_start_disabled = " disabled" if photo_grid_running else ""
    photo_grid_message = escape(request.query_params.get("photo_grid", ""))

    if grid_run is None:
        repair_summary = escape(
            str(grid_repair_state.get("error") or "No recent photo-grid run found")
        )
        repair_details = (
            "A grid is identified when most coordinates form a 2-D lattice within one hour."
        )
        repair_disabled = " disabled"
    else:
        missing_count = sum(target.reason == "missing" for target in grid_run.targets)
        gantry_count = sum(target.reason == "gantry" for target in grid_run.targets)
        repair_summary = (
            f"<b>{grid_run.filled_count} of {grid_run.expected_count} grid cells found</b>"
        )
        repair_details = (
            f"{missing_count} missing · {gantry_count} gantry photo"
            f"{'s' if gantry_count != 1 else ''} · "
            f"last run {escape(grid_run.completed_at.astimezone().strftime('%d %b %Y %H:%M'))}"
        )
        repair_disabled = " disabled" if not grid_run.targets else ""
    repair_checked = " checked" if repair_values.enabled else ""
    repair_message = escape(
        request.query_params.get("repair", "") or str(grid_repair_state.get("message") or "")
    )
    gantry_ids = {
        image_id
        for image_id in grid_repair_state.get("gantry_image_ids", ())
        if isinstance(image_id, int)
    }
    gantry_target_ids = (
        {
            target.image_id
            for target in grid_run.targets
            if target.reason == "gantry" and target.image_id is not None
        }
        if grid_run is not None
        else set()
    )
    gantry_images = (
        [image for image in grid_run.images if image.id in gantry_ids]
        if grid_run is not None
        else []
    )
    gantry_entry = quote(settings.selected_config_entry_id or "")
    gantry_cards = (
        "".join(
            (
                "<figure>"
                f'<img loading=lazy src="api/vision/image/{image.id}.jpg?entry_id={gantry_entry}" '
                f'alt="Gantry classifier photo {image.id}">'
                "<figcaption>"
                f"<b>Image #{image.id}</b><br>"
                f"X {image.meta.x:.1f}, Y {image.meta.y:.1f}, Z {image.meta.z:.1f} mm<br>"
                + (
                    "<span class=gantry-target>Interior repair target</span>"
                    if image.id in gantry_target_ids
                    else "<span class=muted>Perimeter positive — ignored for repair</span>"
                )
                + "</figcaption></figure>"
            )
            for image in gantry_images
        )
        or "<p>No images in the latest grid were classified as gantry photos.</p>"
    )
    gantry_debug_disabled = " disabled" if not gantry_images else ""

    def _dims(value: object) -> str:
        if isinstance(value, list) and len(value) == 2 and value[0] is not None:
            return f"{value[0]}x{value[1]}"
        return "—"

    warnings = last.get("calibration_warnings") or []
    warning_html = (
        "".join(f"<li class=warn>{escape(str(w))}</li>" for w in warnings)
        if warnings
        else "<li class=muted>None</li>"
    )
    skip_reasons = last.get("skip_reasons") or {}
    skip_html = (
        "".join(
            f"<li>Plant {escape(str(pid))}: {escape(str(reason))}</li>"
            for pid, reason in skip_reasons.items()
        )
        if skip_reasons
        else "<li class=muted>None</li>"
    )
    body = f"""<div class=grid><section class=card><h2>Health</h2><b>{escape(jobs.current["status"])}</b>
<p>{escape(jobs.current.get("progress", ""))}</p><p class=muted>App {__version__} · {ALGORITHM_VERSION} · {CONTRACT_VERSION}</p></section>
<section class=card><h2>FarmBot</h2><p>{escape(settings.selected_config_entry_id or "Not selected")}</p>
<p>Mode: {settings.mode.value}</p></section>
<section class=card><h2>Analysis resolution</h2><p><b>{escape(resolution.label)}</b></p>
<p class=muted>{resolution.pixel_count:,} px · restart to change</p></section>
<section class="card photo-grid-card"><h2>Photo grid</h2>
<p><b>{photo_grid_summary}</b></p><p class=muted>{photo_grid_details}</p>
<div class=button-row>
<form method=post action="photo-grid/start"><button{photo_grid_start_disabled}>Start photo grid</button></form>
<button id=photo-grid-open type=button{photo_grid_view_disabled}>View most recent grid</button>
</div>
<small class=action-message>{photo_grid_message}</small>
<p class=muted><small>Uses the saved scale, reference image size, rotation and camera
offsets with the FarmBot axis limits. Every uploaded photo is coordinate-checked before
the bot advances.</small></p></section>
<section class=card><h2>Repair photo grid</h2><p>{repair_summary}</p>
<p class=muted>{repair_details}</p>
<form method=post action="grid-repair/settings">
<label><input type=checkbox name=enabled value=true{repair_checked}> Repair automatically</label>
<input type=number name=delay_minutes min=1 max=1440 step=1
value="{repair_values.delay_minutes}" required> minutes after the photo grid completes
<button>Save</button></form>
<form method=post action="grid-repair/run"><button{repair_disabled}>Repair now</button></form>
<form method=post action="grid-repair/recheck"><button>Recheck grid</button></form>
<button id=gantry-debug-open type=button{gantry_debug_disabled}>View gantry photos ({len(gantry_images)})</button>
<small class=action-message>{repair_message}</small></section>
<section class=card><h2>Analysis</h2><p><span id=queue-count>{len(jobs.queued_image_ids)}</span> queued</p>
<div class=button-row><form method=post action="analyse"><button>Analyse queue</button></form>
<button id=queue-open type=button>Add to queue</button></div>
<div class="button-row clear-actions">
<form method=post action="analysis/clear-recommendations" onsubmit="return confirm('Clear all pending recommendations?');">
<button type=submit class=clear-button>Clear all recommendations</button></form>
<form method=post action="analysis/clear-weeds" onsubmit="return confirm('Clear all pending weed recommendations?');">
<button type=submit class=clear-button>Clear all weed recommendations</button></form>
<form method=post action="analysis/clear-measurements" onsubmit="return confirm('Clear all pending measurements?');">
<button type=submit class=clear-button>Clear all measurements</button></form></div></section></div>
<section class=card><h2>Last job</h2>
<p>{escape(last.get("message", "Never run"))}</p>
<div class=grid>
<div><b>Timing</b><p class=muted>Duration {last.get("duration_seconds", "—")} s · CPU {last.get("cpu_seconds", "—")} s · peak {last.get("peak_memory_mb", "—")} MB</p></div>
<div><b>Images</b><p class=muted>{last.get("images_processed", "—")} processed · {last.get("uncalibrated_images", 0)} uncalibrated</p></div>
<div><b>Plants</b><p class=muted>{last.get("plants_measured", "—")} measured · {last.get("uncertain", "—")} uncertain · {last.get("skipped", "—")} skipped</p></div>
<div><b>Dimensions</b><p class=muted>source {escape(_dims(last.get("source_dimensions")))} · oriented {escape(_dims(last.get("oriented_dimensions")))} · processed {escape(_dims(last.get("processed_dimensions")))}</p></div>
<div><b>Calibration</b><p class=muted>source {escape(str(last.get("calibration_source") or "—"))}</p></div>
<div><b>Contract</b><p class=muted>{escape(str(last.get("contract_version") or CONTRACT_VERSION))} · min integration {MINIMUM_INTEGRATION_VERSION}</p></div>
<div><b>Zones</b><p class=muted>{last.get("zone_blocked_weeds", 0)} weeds · {last.get("zone_blocked_radius", 0)} radius increases blocked</p></div>
</div>
<p><b>Calibration warnings</b></p><ul>{warning_html}</ul>
<p><b>Skip reasons</b></p><ul>{skip_html}</ul></section>
<section class=card><h2>Measurements</h2><table><thead><tr><th>Crop</th><th>Coordinates (x, y)</th><th>Current</th><th>Max leaf</th><th>Recommended</th><th>Confidence</th><th>Decision</th><th>Reason</th><th>Actions</th></tr></thead><tbody>{measurement_rows or "<tr><td colspan=9>No measurements yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Removed / missing plants</h2><table><thead><tr><th>Crop</th><th>Recorded center (X, Y mm)</th><th>Move center to (X, Y mm)</th><th>Absent looks</th><th>Confidence</th><th>Reason</th><th>Diagnostic</th><th>Review</th></tr></thead><tbody>{removal_rows or "<tr><td colspan=8>No confirmed missing plants</td></tr>"}</tbody></table></section>
<section class=card><h2>Detected weeds</h2><p class=muted>Unowned vegetation outside known plant protection areas.</p>
<table><thead><tr><th>Image</th><th>Coordinates</th><th>Area mm²</th><th>Looks</th>
<th>Heuristic</th><th>Verifier</th><th>View</th></tr></thead>
<tbody>{weed_rows or "<tr><td colspan=7>No weed recommendations</td></tr>"}</tbody></table></section>
<section class=card><h2>Growth-curve updates</h2><p class=muted>Flagged per-plant diameter points require review.</p><table><tbody>{flagged_curve_rows or "<tr><td>No flagged curve updates</td></tr>"}</tbody></table></section>
<section class=card><h2>Crop protection spread proposals</h2><p class=muted>Monotonic and limited to 10 points. FarmBot values are diameters; assignment requires approval.</p><table><tbody>{curve_rows or "<tr><td>No curve is ready</td></tr>"}</tbody></table></section>
<section class=card><h2>Approval and rollback history</h2><table><tbody>{decision_rows or "<tr><td>No decisions yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Safety warning</h2><p class=warn>Early experimental vision results must not be the sole basis for destructive automatic weeding.</p></section>
<div id=overlay-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Analysis diagnostic"><figure>
<button id=overlay-modal-close class=modal-close type=button aria-label=Close>&times;</button>
<div id=plant-photo-mode class="modal-controls plant-view-toggle" role=group aria-label="View mode" hidden>
<button id=plant-photo-tab type=button aria-pressed=true>Standard view</button>
<button id=plant-diagnostic-tab type=button aria-pressed=false>Diagnostic mask</button>
</div>
<img id=overlay-modal-img alt="Plant analysis diagnostic"><figcaption id=overlay-modal-details></figcaption>
<p id=overlay-modal-legend class=legend>Cyan circle = original radius; red circle = planned radius.</p>
<div id=artifact-controls class=modal-controls><button id=overlay-modal-prev type=button>Previous</button><span id=overlay-modal-counter></span><button id=overlay-modal-next type=button>Next</button></div>
</figure></div>
<div id=weed-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Weed review">
<figure class=weed-dialog><button id=weed-modal-close class=modal-close type=button aria-label=Close>&times;</button>
<div class="modal-controls weed-view-toggle" role=group aria-label="Weed image view">
<button id=weed-modal-without-overlay type=button aria-pressed=true>Without overlay</button>
<button id=weed-modal-with-overlay type=button aria-pressed=false>With overlay</button>
<button id=weed-modal-closeup type=button aria-pressed=false>Close-up</button>
</div>
<div class="modal-controls weed-zoom" id=weed-modal-zoom hidden>
<label for=weed-modal-zoom-level>Zoom</label>
<input id=weed-modal-zoom-level type=range min=2 max=12 step=1 value=5>
<span id=weed-modal-zoom-value>5&times;</span>
</div>
<div id=weed-image-wrap class=weed-image-wrap><img id=weed-modal-img alt="Weed detection"><div id=weed-modal-marker class=weed-marker hidden></div></div>
<figcaption id=weed-modal-details></figcaption>
<p id=weed-modal-guess class=legend hidden></p>
<p id=weed-modal-legend class=legend>Blue circle = the weed being reviewed; red circles = other detected weeds in this image.</p>
<div class="modal-controls weed-navigation" aria-label="Navigate weeds in this image">
<button id=weed-modal-prev-weed type=button>Previous weed</button>
<span>Weed <span id=weed-modal-weed-counter>0 / 0</span></span>
<button id=weed-modal-next-weed type=button>Next weed</button>
</div>
<div class="modal-controls weed-navigation" aria-label="Navigate weed images">
<button id=weed-modal-prev-image type=button>Previous image</button>
<span>Image <span id=weed-modal-image-counter>0 / 0</span></span>
<button id=weed-modal-next-image type=button>Next image</button>
</div>
<div class="weed-actions">
<div class=button-row><button id=weed-modal-accept type=button>Accept</button>
<button id=weed-modal-accept-all type=button>Accept all</button>
<button id=weed-modal-unknown type=button class=unknown-button>Unknown</button></div>
<fieldset><legend>Reject as</legend><div class=button-row>
<button type=button data-weed-label=crop>Crop</button>
<button type=button data-weed-label=fallen_leaf>Fallen leaf</button>
<button type=button data-weed-label=mushroom>Mushroom</button>
<button type=button data-weed-label=moss>Moss</button>
<button type=button data-weed-label=soil>Soil</button>
<button type=button data-weed-label=hardware>Hardware</button>
</div></fieldset>
<p class=muted><small>Fallen leaf identifies an isolated leaf that has detached from a plant
and teaches the verifier that it is not a weed. Unknown discards an unclear detection without
accepting, rejecting, or adding it to the verifier's training data. Reviewing keeps the dialog
open and moves on to the next weed.</small></p>
</div>
<small id=weed-modal-message class=action-message></small>
</figure></div>
<div id=queue-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Add images to analysis queue">
<figure class=queue-dialog><button id=queue-close class=modal-close type=button aria-label=Close>&times;</button>
<h2>Add images to analysis queue</h2>
<div class=button-row><label>From <input id=queue-from type=datetime-local></label>
<label>To <input id=queue-to type=datetime-local></label>
<button id=queue-refresh type=button>Refresh</button><label><input id=queue-select-all type=checkbox> Select all</label></div>
<p id=queue-message class=muted></p><table><thead><tr><th>Select</th><th>Coordinates (x, y, z)</th>
<th>Plants present</th><th>Date taken</th></tr></thead><tbody id=queue-image-rows></tbody></table>
<div class=button-row><button id=queue-add type=button>Add selected images to queue</button></div>
</figure></div>
<div id=gantry-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Gantry classifier debug photos">
<figure><button id=gantry-close class=modal-close type=button aria-label=Close>&times;</button>
<h2>Gantry classifier debug</h2>
<p class=muted>Every positive from the latest photo batch is shown. Perimeter positives are displayed for diagnosis but are not repair targets.</p>
<div class=gantry-gallery>{gantry_cards}</div>
</figure></div>
<div id=photo-grid-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Most recent photo grid">
<figure class=photo-grid-dialog><button id=photo-grid-close class=modal-close type=button aria-label=Close>&times;</button>
<h2>Most recent photo grid</h2>
<canvas id=photo-grid-canvas width=900 height=420 aria-label="Birds-eye photo grid"></canvas>
<p id=photo-grid-status class=muted>Loading the verified grid…</p>
<p class=muted><small>The canvas uses the same calibrated garden-coordinate transform as
analysis. Green circles are FarmBot plants and red dots are FarmBot weed points.</small></p>
</figure></div><script>{_DASHBOARD_JS}</script>"""  # noqa: S608 - HTML template
    return layout(request, body)


@app.post("/photo-grid/start")
async def run_calibrated_photo_grid() -> RedirectResponse:
    try:
        record = await start_calibrated_photo_grid()
        message = record.message
    except (HomeAssistantError, ValueError) as exc:
        message = f"Could not start photo grid: {exc}"
    return RedirectResponse(f"../?photo_grid={quote(message)}", status_code=303)


@app.get("/api/photo-grid/latest")
async def latest_calibrated_photo_grid() -> JSONResponse:
    record = photo_grid_store.load()
    if record is None:
        raise HTTPException(404, "No calibrated photo grid has been captured")
    plants: list[dict[str, object]] = []
    weeds: list[dict[str, object]] = []
    try:
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=record.config_entry_id,
                image_lookback_hours=min(720, max(72, settings.image_lookback_hours)),
            )
        )
        plants = [
            {
                "id": plant.id,
                "name": plant.name,
                "x": plant.x,
                "y": plant.y,
                "radius": plant.radius,
            }
            for plant in inventory.plants
        ]
        weeds = [
            {"id": weed.id, "name": weed.name, "x": weed.x, "y": weed.y, "radius": weed.radius}
            for weed in inventory.weeds
        ]
    except HomeAssistantError:
        # The verified mosaic remains viewable while FarmBot is temporarily
        # offline; live plant/weed markers are supplementary.
        pass
    return JSONResponse({"grid": record.model_dump(mode="json"), "plants": plants, "weeds": weeds})


@app.post("/grid-repair/settings")
async def save_grid_repair_settings(
    delay_minutes: Annotated[int, Form()],
    enabled: Annotated[bool | None, Form()] = None,
) -> RedirectResponse:
    values = GridRepairSettings(enabled=bool(enabled), delay_minutes=delay_minutes)
    grid_repair_settings_store.save(values)
    return RedirectResponse(
        f"../?repair={quote('Photo-grid repair schedule saved')}", status_code=303
    )


@app.post("/grid-repair/run")
async def run_grid_repair() -> RedirectResponse:
    try:
        result = await start_photo_grid_repair()
        message = str(result.get("message") or result.get("status") or "Repair requested")
    except HomeAssistantError as exc:
        message = f"Could not start repair: {exc}"
    return RedirectResponse(f"../?repair={quote(message)}", status_code=303)


@app.post("/grid-repair/recheck")
async def recheck_grid_repair() -> RedirectResponse:
    run = await inspect_photo_grid(force=True)
    if run is None:
        message = str(grid_repair_state.get("error") or "No recent photo-grid run was found")
    elif not run.targets:
        message = "Photo grid is complete"
    else:
        missing = sum(target.reason == "missing" for target in run.targets)
        gantry = sum(target.reason == "gantry" for target in run.targets)
        message = f"Found {missing} missing and {gantry} gantry photo(s) to repair"
    return RedirectResponse(f"../?repair={quote(message)}", status_code=303)


@app.post("/analyse")
async def analyse(background: BackgroundTasks) -> RedirectResponse:
    background.add_task(jobs.run, trigger="manual")
    return RedirectResponse("./", status_code=303)


@app.get("/api/analysis/images")
async def analysis_images(
    date_from: datetime | None = None, date_to: datetime | None = None
) -> JSONResponse:
    if not settings.selected_config_entry_id:
        raise HTTPException(400, "Select a FarmBot before loading images")
    now = datetime.now(UTC)
    date_to = date_to or now
    date_from = date_from or (date_to - timedelta(hours=72))
    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=UTC)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=UTC)
    if date_from > date_to:
        raise HTTPException(422, "From must be before to")
    hours = max(1, min(720, int((now - date_from).total_seconds() / 3600) + 1))
    try:
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=settings.selected_config_entry_id,
                image_lookback_hours=hours,
            )
        )
    except HomeAssistantError as exc:
        raise HTTPException(502, str(exc)) from exc
    calibration = database.active_calibration(settings.selected_config_entry_id)
    width, height = settings.analysis_width, settings.analysis_height
    items = []
    for image in sorted(inventory.images, key=lambda item: item.created_at, reverse=True):
        if not date_from <= image.created_at <= date_to:
            continue
        present = []
        for plant in inventory.plants:
            if calibration is not None:
                px, py = garden_to_pixel(
                    plant.x,
                    plant.y,
                    image.meta.x,
                    image.meta.y,
                    width,
                    height,
                    calibration,
                )
                is_present = (
                    -plant.radius * calibration.pixels_per_mm_x
                    <= px
                    <= width + plant.radius * calibration.pixels_per_mm_x
                    and -plant.radius * calibration.pixels_per_mm_y
                    <= py
                    <= height + plant.radius * calibration.pixels_per_mm_y
                )
            else:
                # Useful conservative fallback before calibration: images still
                # remain selectable and nearby plants are listed approximately.
                is_present = (
                    abs(plant.x - image.meta.x) <= 500 and abs(plant.y - image.meta.y) <= 400
                )
            if is_present:
                present.append({"id": plant.id, "name": plant.name})
        items.append(
            {
                "id": image.id,
                "created_at": image.created_at.isoformat(),
                "x": image.meta.x,
                "y": image.meta.y,
                "z": image.meta.z,
                "plants": present,
            }
        )
    return JSONResponse(
        {"images": items, "date_from": date_from.isoformat(), "date_to": date_to.isoformat()}
    )


@app.post("/analysis/queue")
async def add_analysis_queue(request: QueueImagesRequest) -> JSONResponse:
    return JSONResponse({"queue_length": jobs.add_to_queue(request.image_ids)})


@app.post("/analysis/clear-recommendations")
async def clear_all_recommendations() -> RedirectResponse:
    database.clear_pending_measurements()
    database.clear_pending_weed_detections()
    database.clear_flagged_curve_proposals()
    return RedirectResponse("../", status_code=303)


@app.post("/analysis/clear-weeds")
async def clear_weed_recommendations() -> RedirectResponse:
    database.clear_pending_weed_detections()
    return RedirectResponse("../", status_code=303)


@app.post("/analysis/clear-measurements")
async def clear_measurements() -> RedirectResponse:
    database.clear_pending_measurements()
    return RedirectResponse("../", status_code=303)


def _soil_artifacts(paths: list[str]) -> str:
    links = []
    for path in paths:
        if not path:
            continue
        url = f"artifact/{Path(path).name}"
        label = Path(path).stem.rsplit("-", 1)[-1]
        links.append(
            f'<a href="{escape(url, quote=True)}" target=_blank rel=noopener>{escape(label)}</a>'
        )
    return " ".join(links) or "<span class=muted>None</span>"


@app.get("/soil-height", response_class=HTMLResponse)
async def soil_height_page(request: Request) -> HTMLResponse:
    entry_id = settings.selected_config_entry_id
    inventory = None
    sites = []
    inventory_error = ""
    calibration = database.active_soil_calibration(entry_id) if entry_id else None
    planning_baseline = calibration.baseline_mm if calibration else 15
    if entry_id:
        try:
            inventory, sites = await soil_jobs.safe_sites(entry_id, planning_baseline)
        except HomeAssistantError as exc:
            inventory_error = str(exc)
    measurements = database.recent_soil_measurements(entry_id, 200)
    persisted_job = database.latest_soil_job(entry_id)
    current_job = soil_jobs.current if soil_jobs.running else (persisted_job or soil_jobs.current)

    latest_by_point: dict[int, dict] = {}
    for measurement in measurements:
        latest_by_point.setdefault(int(measurement["point_id"]), measurement)

    point_rows = ""
    point_options = ""
    retry_ids: list[int] = []
    if inventory:
        for site in sites:
            measurement = latest_by_point.get(site.point_id)
            status = measurement["status"] if measurement else "not measured"
            proposed = (
                f"{measurement['proposed_z_mm']:.0f} mm"
                if measurement and measurement["proposed_z_mm"] is not None
                else "—"
            )
            uncertainty = (
                f"±{measurement['uncertainty_mm']:.1f} mm"
                if measurement and measurement["uncertainty_mm"] is not None
                else "—"
            )
            confidence = f"{100 * measurement['confidence']:.0f}%" if measurement else "—"
            reason = escape(measurement["reason"] if measurement else "")
            diagnostics = _soil_artifacts(measurement["artifact_paths"] if measurement else [])
            if measurement and measurement["status"] == "failed":
                retry_ids.append(site.point_id)
            apply_control = ""
            if measurement and measurement["status"] == "valid":
                measurement_id = escape(measurement["measurement_id"], quote=True)
                apply_control = (
                    f'<form method=post action="soil/measurements/{measurement_id}/apply">'
                    "<button type=submit>Apply</button></form>"
                    f'<form method=post action="soil/measurements/{measurement_id}/reject">'
                    "<button type=submit>Reject</button></form>"
                )
            point_rows += (
                "<tr>"
                f'<td><input form=measure-points type=checkbox name=point_ids value="{site.point_id}"></td>'
                f"<td>{site.point_id}</td><td>{escape(site.point_name)}</td>"
                f"<td>{site.expected_x:.1f}, {site.expected_y:.1f}</td>"
                f"<td>{site.capture_x:.1f}, {site.capture_y:.1f}</td>"
                f"<td>{site.relocation_distance_mm:.1f} mm</td>"
                f"<td>{site.point_updated_at.date().isoformat()}</td>"
                f"<td>{site.expected_z:.1f} mm</td>"
                f"<td>{proposed}</td><td>{uncertainty}</td><td>{confidence}</td>"
                f"<td>{escape(status)}</td><td>{reason}</td><td>{diagnostics}</td>"
                f"<td>{apply_control}</td></tr>"
            )
            point_options += (
                f'<option value="{site.point_id}">{escape(site.point_name)}: clear soil '
                f"({site.capture_x:.0f}, {site.capture_y:.0f})</option>"
            )

    valid_measurements = [
        item
        for item in measurements
        if item["status"] == "valid"
        and item.get("capture_x") is not None
        and item.get("capture_y") is not None
        and item.get("point_updated_at")
    ]
    measurement_rows = "".join(
        "<tr>"
        f"<td><input form=apply-selected type=checkbox name=measurement_ids "
        f'value="{escape(item["measurement_id"], quote=True)}"></td>'
        f"<td>{escape(item['point_name'])}</td>"
        f"<td>{item['expected_x']:.1f}, {item['expected_y']:.1f}</td>"
        f"<td>{item['capture_x']:.1f}, {item['capture_y']:.1f}</td>"
        f"<td>{item['old_z_mm']:.1f} mm</td>"
        f"<td>{item['proposed_z_mm']:.0f} mm</td>"
        f"<td>{100 * item['confidence']:.0f}%</td>"
        f"<td>{escape(item['reason'])}</td></tr>"
        for item in valid_measurements
    )
    point_count = len(inventory.points) if inventory else 0
    site_count = len(sites)
    warning = (
        "<p class=warn>Fewer than three stale soil points currently have a nearby "
        "clear-soil replacement. FarmBot soil-height interpolation needs at least "
        "three measured points.</p>"
        if site_count < 3
        else ""
    )
    motion = inventory.motion if inventory else None
    motion_summary = (
        f"connected={motion.connected}, busy={motion.busy}, emergency stop={motion.locked}, "
        f"position={escape(json.dumps(motion.position))}"
        if motion
        else "unavailable"
    )
    calibration_summary = (
        f"Active calibration #{calibration.calibration_id}: "
        f"{calibration.processed_width}×{calibration.processed_height}, "
        f"{calibration.baseline_mm:.0f} mm baseline, "
        f"{calibration.residual_mm:.1f} mm residual"
        if calibration
        else "No active soil calibration. Complete the guided calibration before measuring."
    )
    default_capture_z = calibration.capture_z if calibration else 0
    default_baseline = calibration.baseline_mm if calibration else 15
    capture_z_hint = hint(
        "The FarmBot Z-axis (height) position the gantry moves to before taking soil "
        "photos. During calibration the bot also steps down 25 mm and 50 mm from this "
        "height to build the depth curve; during measurement it only captures here. "
        "Use the same Capture Z every time — changing it invalidates the calibration."
    )
    baseline_hint = hint(
        "How far, in mm, the camera shifts sideways (along Y) between the shots taken "
        "at each point. This lateral shift is the 'virtual stereo' separation used to "
        "compute soil depth from the difference between images, similar to the "
        "distance between two eyes. It must match the value used for calibration — "
        "changing it requires recalibrating."
    )
    job_message = escape(str(current_job.get("message", "Not run")))
    job_status = escape(str(current_job.get("status", "idle")))
    retry_values = "".join(
        f'<input type=hidden name=point_ids value="{point_id}">' for point_id in retry_ids
    )
    live_refresh = (
        "<script>setTimeout(()=>location.reload(),3000)</script>" if soil_jobs.running else ""
    )
    body = f"""
<h2>Supplemental soil-height measurement</h2>
<p>Finds plant- and weed-free soil within 200 mm of FarmBot soil points that have
not been updated for more than 14 days. Measurements are captured at those clear
locations and, after review, replace the assigned stale point.</p>
{warning}
<section class=grid>
 <div class=card><h3>Bot</h3><p>{escape(entry_id or "No FarmBot selected")}</p>
 <p class=muted>{motion_summary}</p><p>{escape(inventory_error)}</p></div>
 <div class=card><h3>Calibration</h3><p>{escape(calibration_summary)}</p>
 <p class=warn>Recalibrate after moving, rotating, or refocusing the camera.</p></div>
 <div class=card><h3>Current job</h3><p><strong>{job_status}</strong>: {job_message}</p>
 <form method=post action=soil/stop><button type=submit>Stop after current point</button></form></div>
</section>
<section class=card>
 <h3>Guided calibration</h3>
 <p>Choose one of the calculated clear-soil sites. Enter the manually measured
camera-to-soil distance at the capture Z, then confirm that a 50 mm movement
toward the soil is safe.</p>
 <form method=post action=soil/calibrate>
  <label>Clear soil site <select name=point_id required>{point_options}</select></label>
  <label>Camera-to-soil distance (mm) <input type=number min=1 step=0.1
   name=reference_distance_mm required></label>
  <label>Capture Z (mm){capture_z_hint} <input type=number step=0.1 name=capture_z value=0 required></label>
  <label>Baseline (mm){baseline_hint} <input type=number min=5 max=30 step=0.1
   name=baseline_mm value=15 required></label>
  <label><input type=checkbox name=safety_confirm required> I confirm the automated
   50 mm movement toward the soil is safe</label>
  <button type=submit>Calibrate</button>
 </form>
</section>
<section class=card>
 <h3>Clear-soil replacements ({site_count} from {point_count} existing points)</h3>
 <p class=muted>Each candidate has a 75 mm clear-soil margin, expanded for the
stereo movement, around all current FarmBot plants and weeds, the latest
detected plant canopies, and pending or created Vision weeds. Fresh points and
points without a trustworthy update date are not replaced.</p>
 <form id=measure-points method=post action=soil/measure>
  <label>Capture Z (mm){capture_z_hint} <input type=number step=0.1 name=capture_z
   value="{default_capture_z:g}" required></label>
  <label>Baseline (mm){baseline_hint} <input type=number min=5 max=30 step=0.1 name=baseline_mm
   value="{default_baseline:g}" required></label>
  <button type=submit name=mode value=selected>Measure selected</button>
  <button type=submit name=mode value=all>Measure all</button>
 </form>
 <form method=post action=soil/measure>{retry_values}
  <input type=hidden name=capture_z value="{default_capture_z:g}">
  <input type=hidden name=baseline_mm value="{default_baseline:g}">
  <button type=submit name=mode value=retry {"disabled" if not retry_ids else ""}>Retry failed</button>
 </form>
 <table><thead><tr><th>Select</th><th>ID</th><th>Replaces</th><th>Old X, Y</th>
 <th>Clear X, Y</th><th>Move</th><th>Last updated</th><th>Current Z</th>
 <th>Proposed Z</th><th>Uncertainty</th><th>Confidence</th>
 <th>Status</th><th>Message</th><th>Diagnostics</th><th>Review</th></tr></thead>
 <tbody>{point_rows or "<tr><td colspan=15>No stale point has a safe clear-soil site within 200 mm.</td></tr>"}</tbody></table>
</section>
<section class=card>
 <h3>Pending valid results</h3>
 <form id=apply-selected method=post action=soil/apply-selected>
  <button type=submit>Apply selected</button>
 </form>
 <table><thead><tr><th>Select</th><th>Point</th><th>Old X, Y</th><th>New X, Y</th>
 <th>Old Z</th><th>Proposed Z</th>
 <th>Confidence</th><th>Quality result</th></tr></thead>
 <tbody>{measurement_rows or "<tr><td colspan=8>No unapplied valid results.</td></tr>"}</tbody></table>
</section>
 {live_refresh}"""  # noqa: S608 - HTML template; no SQL is constructed here.
    return layout(request, body, "Soil height · FarmBot Vision")


@app.get("/api/soil/points")
async def soil_points_api() -> JSONResponse:
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        raise HTTPException(409, "No FarmBot config entry is selected")
    calibration = database.active_soil_calibration(entry_id)
    inventory, sites = await soil_jobs.safe_sites(
        entry_id, calibration.baseline_mm if calibration else 15
    )
    return JSONResponse(
        {
            "inventory": inventory.model_dump(mode="json"),
            "safe_sites": [site.model_dump(mode="json") for site in sites],
        }
    )


@app.get("/api/soil/job")
async def soil_job_api() -> JSONResponse:
    return JSONResponse(soil_jobs.current)


@app.post("/soil/calibrate")
async def start_soil_calibration(
    point_id: int = Form(...),
    reference_distance_mm: float = Form(...),
    capture_z: float = Form(0),
    baseline_mm: float = Form(15),
    safety_confirm: bool = Form(False),
) -> RedirectResponse:
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        raise HTTPException(409, "No FarmBot config entry is selected")
    if not safety_confirm:
        raise HTTPException(422, "Confirm that the 50 mm calibration movement is safe")
    _inventory, sites = await soil_jobs.safe_sites(entry_id, baseline_mm)
    site = next((item for item in sites if item.point_id == point_id), None)
    if site is None:
        raise HTTPException(404, "Clear-soil calibration site not found")
    try:
        soil_jobs.start_calibration(
            config_entry_id=entry_id,
            point_id=point_id,
            capture_z=capture_z,
            baseline_mm=baseline_mm,
            reference_distance_mm=reference_distance_mm,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return RedirectResponse("../soil-height", status_code=303)


@app.post("/soil/measure")
async def start_soil_measurement(
    point_ids: Annotated[list[int] | None, Form()] = None,
    mode: str = Form("selected"),
    capture_z: float = Form(0),
    baseline_mm: float = Form(15),
) -> RedirectResponse:
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        raise HTTPException(409, "No FarmBot config entry is selected")
    if mode == "all":
        _inventory, sites = await soil_jobs.safe_sites(entry_id, baseline_mm)
        point_ids = [site.point_id for site in sites]
    point_ids = point_ids or []
    try:
        soil_jobs.start_measurements(
            config_entry_id=entry_id,
            point_ids=point_ids,
            capture_z=capture_z,
            baseline_mm=baseline_mm,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return RedirectResponse("../soil-height", status_code=303)


@app.post("/soil/stop")
async def stop_soil_measurement() -> RedirectResponse:
    if soil_jobs.running:
        soil_jobs.request_stop()
    return RedirectResponse("../soil-height", status_code=303)


async def _apply_soil_measurement(measurement_id: str) -> dict:
    measurement = database.soil_measurement(measurement_id)
    if (
        measurement is None
        or measurement["status"] != "valid"
        or measurement["proposed_z_mm"] is None
        or measurement.get("capture_x") is None
        or measurement.get("capture_y") is None
        or not measurement.get("point_updated_at")
    ):
        raise HTTPException(404, "Applicable soil result not found")
    apply_request = ApplySoilHeightRequest(
        config_entry_id=measurement["config_entry_id"],
        point_id=measurement["point_id"],
        measurement_id=measurement["measurement_id"],
        expected_x=measurement["expected_x"],
        expected_y=measurement["expected_y"],
        expected_z=measurement["old_z_mm"],
        expected_updated_at=measurement["point_updated_at"],
        recommended_x=measurement["capture_x"],
        recommended_y=measurement["capture_y"],
        recommended_z_mm=measurement["proposed_z_mm"],
        confidence=measurement["confidence"],
        apply=True,
        human_approved=True,
    )
    try:
        response = await client.apply_soil_height(apply_request)
    except HomeAssistantError as exc:
        response = {"status": "conflict", "message": str(exc)}
    response_status = str(response.get("status") or "rejected")
    if response_status == "applied":
        status, action = "applied", "approve"
    elif response_status == "conflict":
        status, action = "conflict", "stale_conflict"
    else:
        status, action = "rejected", "rejected_write"
    reason = str(response.get("message") or response.get("status") or status)[:240]
    database.update_soil_measurement_status(measurement_id, status, reason)
    database.record_soil_decision(measurement_id, action, response)
    return response


@app.post("/soil/measurements/{measurement_id}/apply")
async def apply_soil_measurement(measurement_id: UUID) -> RedirectResponse:
    await _apply_soil_measurement(str(measurement_id))
    return RedirectResponse("../../../soil-height", status_code=303)


@app.post("/soil/apply-selected")
async def apply_selected_soil_measurements(
    measurement_ids: Annotated[list[str] | None, Form()] = None,
) -> RedirectResponse:
    for measurement_id in measurement_ids or []:
        try:
            UUID(measurement_id)
        except ValueError as exc:
            raise HTTPException(422, "Malformed soil measurement ID") from exc
        await _apply_soil_measurement(measurement_id)
    return RedirectResponse("../soil-height", status_code=303)


@app.post("/soil/measurements/{measurement_id}/reject")
async def reject_soil_measurement(measurement_id: UUID) -> RedirectResponse:
    measurement = database.soil_measurement(str(measurement_id))
    if measurement is None or measurement["status"] != "valid":
        raise HTTPException(404, "Reviewable soil result not found")
    database.update_soil_measurement_status(
        str(measurement_id), "rejected", "Rejected during human review"
    )
    database.record_soil_decision(str(measurement_id), "reject", {"status": "rejected"})
    return RedirectResponse("../../../soil-height", status_code=303)


@app.get("/canopy-settings", response_class=HTMLResponse)
async def canopy_settings_page(request: Request) -> HTMLResponse:
    values = canopy_fusion_settings_store.load()

    def checked(value: bool) -> str:
        return " checked" if value else ""

    body = f"""<section class=card><h2>Multi-image canopy fusion</h2>
<p>Plant segmentation still runs on each original image. When a plant reaches an image edge,
the resulting ownership masks are aligned in calibrated garden coordinates and fused before
its radius is measured. This avoids seams and duplicate leaves from an RGB panorama.</p>
<form method=post action="canopy-settings">
<fieldset><legend>Activation</legend>
<label><input type=checkbox name=enabled value=true{checked(values.enabled)}> Enable calibrated mask fusion</label><br>
<label><input type=checkbox name=always_fuse_when_available value=true{checked(values.always_fuse_when_available)}> Fuse whenever enough views are available</label><br>
<label>Fuse below visible fraction <input type=number name=activation_visible_fraction min=0 max=1 step=.01 value="{values.activation_visible_fraction:g}"></label><br>
<label>Minimum views <input type=number name=minimum_views min=2 max=20 step=1 value="{values.minimum_views}"></label><br>
<label>Maximum time gap (hours) <input type=number name=maximum_time_gap_hours min=.1 max=720 step=.1 value="{values.maximum_time_gap_hours:g}"></label>
</fieldset>
<fieldset><legend>Evidence acceptance</legend>
<label>Minimum per-view confidence <input type=number name=minimum_view_confidence min=0 max=1 step=.01 value="{values.minimum_view_confidence:g}"></label><br>
<label>Supporting views required per pixel <input type=number name=minimum_supporting_views min=1 max=10 step=1 value="{values.minimum_supporting_views}"></label><br>
<label>Single-view pixel confidence <input type=number name=single_view_acceptance_confidence min=0 max=1 step=.01 value="{values.single_view_acceptance_confidence:g}"></label><br>
<label>Source-edge evidence margin (mm) <input type=number name=source_edge_margin_mm min=0 max=250 step=1 value="{values.source_edge_margin_mm:g}"></label>
</fieldset>
<fieldset><legend>Radius measurement</legend>
<label>Outer radial percentile <input type=number name=radial_percentile min=80 max=100 step=.1 value="{values.radial_percentile:g}"></label><br>
<label>Angular sectors <input type=number name=angular_sectors min=12 max=360 step=1 value="{values.angular_sectors}"></label><br>
<label>Maximum fusion canvas (pixels) <input type=number name=maximum_canvas_pixels min=480 max=6000 step=10 value="{values.maximum_canvas_pixels}"></label>
</fieldset>
<fieldset><legend>Automatic-action guardrails</legend>
<label><input type=checkbox name=automatic_requires_reliable_fusion value=true{checked(values.automatic_requires_reliable_fusion)}> Require reliable fusion when partial views are present</label><br>
<label>Minimum angular coverage <input type=number name=minimum_angular_coverage min=0 max=1 step=.01 value="{values.minimum_angular_coverage:g}"></label><br>
<label>Minimum corroborated mask fraction <input type=number name=minimum_corroborated_fraction min=0 max=1 step=.01 value="{values.minimum_corroborated_fraction:g}"></label><br>
<label>Maximum disagreement with per-image estimate (mm) <input type=number name=maximum_automatic_disagreement_mm min=0 max=500 step=1 value="{values.maximum_automatic_disagreement_mm:g}"></label><br>
<label><input type=checkbox name=save_diagnostics value=true{checked(values.save_diagnostics)}> Save fusion diagnostics for review</label>
</fieldset>
<button>Save canopy fusion settings</button></form>
<p class=muted>Disabling a guardrail permits more automation but does not remove the normal
confidence, calibration, zone, or plant-safety checks.</p></section>"""
    return layout(request, body, "Canopy fusion")


@app.post("/canopy-settings")
async def save_canopy_settings(
    enabled: bool = Form(False),
    always_fuse_when_available: bool = Form(False),
    activation_visible_fraction: float = Form(0.92),
    minimum_views: int = Form(2),
    maximum_time_gap_hours: float = Form(6),
    minimum_view_confidence: float = Form(0.35),
    minimum_supporting_views: int = Form(2),
    single_view_acceptance_confidence: float = Form(0.82),
    source_edge_margin_mm: float = Form(20),
    radial_percentile: float = Form(97),
    angular_sectors: int = Form(72),
    minimum_angular_coverage: float = Form(0.70),
    minimum_corroborated_fraction: float = Form(0.05),
    maximum_automatic_disagreement_mm: float = Form(35),
    automatic_requires_reliable_fusion: bool = Form(False),
    maximum_canvas_pixels: int = Form(2400),
    save_diagnostics: bool = Form(False),
) -> RedirectResponse:
    try:
        values = CanopyFusionSettings(
            enabled=enabled,
            always_fuse_when_available=always_fuse_when_available,
            activation_visible_fraction=activation_visible_fraction,
            minimum_views=minimum_views,
            maximum_time_gap_hours=maximum_time_gap_hours,
            minimum_view_confidence=minimum_view_confidence,
            minimum_supporting_views=minimum_supporting_views,
            single_view_acceptance_confidence=single_view_acceptance_confidence,
            source_edge_margin_mm=source_edge_margin_mm,
            radial_percentile=radial_percentile,
            angular_sectors=angular_sectors,
            minimum_angular_coverage=minimum_angular_coverage,
            minimum_corroborated_fraction=minimum_corroborated_fraction,
            maximum_automatic_disagreement_mm=maximum_automatic_disagreement_mm,
            automatic_requires_reliable_fusion=automatic_requires_reliable_fusion,
            maximum_canvas_pixels=maximum_canvas_pixels,
            save_diagnostics=save_diagnostics,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if values.minimum_supporting_views > values.minimum_views:
        raise HTTPException(422, "Supporting views per pixel cannot exceed the minimum view count")
    canopy_fusion_settings_store.save(values)
    return RedirectResponse("canopy-settings", status_code=303)


@app.get("/weed-settings", response_class=HTMLResponse)
async def weed_settings_page(request: Request) -> HTMLResponse:
    values = weed_settings_store.load()
    labels = database.weed_training_summary()
    training_samples = database.weed_training_samples()
    weed_verifier.reload()
    model = weed_verifier.model

    def checked(value: bool) -> str:
        return " checked" if value else ""

    if not model:
        model_status = "No trained verifier model yet."
        threshold_html = ""
    else:
        origin = (
            "Bundled starter model shipped with this add-on"
            if weed_verifier.is_bundled
            else f"Trained {escape(str(model['created_at']))}"
        )
        held_out = (
            f"{model.get('validation_groups', 0)} held-out image(s)"
            if model.get("grouped_validation")
            else "the training data itself (too few distinct images to hold any out)"
        )
        suggested = weed_verifier.suggested_threshold
        model_status = (
            f"{origin} from {model['sample_count']} labels "
            f"({model['positive_count']} weed / {model['negative_count']} non-weed). "
            f"Measured on {held_out}: precision {model['metrics']['precision']:.1%} "
            f"(at least {float(model['metrics'].get('precision_lower_bound', 0)):.0%} with 90% "
            f"confidence), recall {model['metrics']['recall']:.1%}"
            + (f" at threshold {suggested:.2f}." if suggested is not None else ".")
        )
        # The threshold that hits the target precision is the one number worth
        # acting on, so offer it as a single click rather than asking the user
        # to transcribe it into the field above.
        threshold_html = ""
        if suggested is not None:
            matches = abs(suggested - values.visual_verifier_minimum_confidence) < 5e-3
            threshold_html = (
                f"<p>Suggested verifier threshold for "
                f"{float(model.get('target_precision', 0.95)):.0%} precision: "
                f"<b>{suggested:.2f}</b> — currently set to "
                f"{values.visual_verifier_minimum_confidence:.2f}."
                + (
                    " Already applied.</p>"
                    if matches
                    else '</p><form method=post action="weed-model/apply-threshold">'
                    "<button>Apply suggested threshold</button></form>"
                )
            )
    training_notice = request.query_params.get("training")
    training_notice_html = f"<p class=warn>{escape(training_notice)}</p>" if training_notice else ""

    def training_sample_row(sample: dict) -> str:
        detection_id = str(sample["detection_id"])
        crop_path = sample.get("crop_path")
        crop = (
            f'<img class=training-crop src="artifact/{escape(Path(crop_path).name, quote=True)}" '
            f'alt="Candidate crop {escape(detection_id, quote=True)}">'
            if crop_path
            else "<span class=muted>No crop image</span>"
        )
        options = "".join(
            f'<option value="{escape(label, quote=True)}"'
            f"{' selected' if sample['label'] == label else ''}>"
            f"{escape(label.replace('_', '/'))}</option>"
            for label in TRAINING_LABELS
        )
        return (
            "<tr>"
            f"<td>{crop}</td>"
            f"<td><code>{escape(detection_id)}</code></td>"
            f"<td>{escape(str(sample['created_at']))}</td>"
            f'<td><form method=post action="weed-model/samples/{escape(detection_id, quote=True)}">'
            f'<select name=label aria-label="Training label for {escape(detection_id, quote=True)}">'
            f"{options}</select> <button>Save tag</button></form></td>"
            "</tr>"
        )

    training_sample_rows = "".join(training_sample_row(sample) for sample in training_samples)
    training_samples_html = (
        "<p>No labeled training images.</p>"
        if not training_samples
        else f"<table><thead><tr><th>Image</th><th>Detection</th><th>Labeled</th>"
        f"<th>Tag</th></tr></thead><tbody>{training_sample_rows}</tbody></table>"
    )
    # Labelling a candidate the model is already sure about teaches it nothing.
    # Rank the unlabelled backlog by how close each one sits to the operating
    # threshold and offer the most ambiguous handful first.
    boundary = weed_verifier.suggested_threshold or values.visual_verifier_minimum_confidence
    uncertain: list[tuple[float, dict, list[dict[str, object]]]] = []
    if weed_verifier.available:
        for candidate in database.unlabelled_weed_detections():
            try:
                features = json.loads(candidate.get("features_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            score = weed_verifier.predict(features) if isinstance(features, dict) else None
            if score is None:
                continue
            uncertain.append((abs(score - boundary), candidate, _verifier_label_guess(features)))
        uncertain.sort(key=lambda item: item[0])
    del uncertain[8:]

    def uncertain_row(candidate: dict, guess: list[dict[str, object]]) -> str:
        detection_id = str(candidate["detection_id"])
        crop_path = candidate.get("crop_path")
        crop = (
            f'<img class=training-crop src="artifact/{escape(Path(crop_path).name, quote=True)}" '
            f'alt="Candidate crop {escape(detection_id, quote=True)}">'
            if crop_path
            else "<span class=muted>No crop image</span>"
        )
        guess_text = (
            " · ".join(
                f"{escape(str(entry['label']))} {float(entry['probability']):.0%}"
                for entry in guess
            )
            or "<span class=muted>no per-category heads yet</span>"
        )
        options = "".join(
            f'<option value="{escape(label, quote=True)}">{escape(label.replace("_", "/"))}</option>'
            for label in TRAINING_LABELS
        )
        return (
            f"<tr><td>{crop}</td><td>image {candidate['image_id']}<br>"
            f"<span class=muted>{guess_text}</span></td>"
            f'<td><form method=post action="weed-model/label/{escape(detection_id, quote=True)}">'
            f'<select name=label aria-label="Label for {escape(detection_id, quote=True)}">'
            f"{options}</select> <button>Save label</button></form></td></tr>"
        )

    uncertain_html = (
        "<p class=muted>Nothing to suggest yet — train the verifier first, then the most "
        "ambiguous unlabelled candidates appear here.</p>"
        if not uncertain
        else "<table><thead><tr><th>Image</th><th>Detection</th><th>Label</th></tr></thead>"
        f"<tbody>{''.join(uncertain_row(row, guess) for _, row, guess in uncertain)}"
        "</tbody></table>"
    )

    body = f"""<section class=card><h2>Weed detection and automation</h2>
<p>Every stage is configurable. Start in review/shadow mode, label real examples, train the
local verifier, then enable enforcement or automatic FarmBot creation when its validation
results and field behaviour are satisfactory.</p>
<form method=post action="weed-settings">
<fieldset><legend>Operation</legend>
<label><input type=checkbox name=enabled value=true{checked(values.enabled)}> Enable weed detection</label><br>
<label><input type=checkbox name=automatic_creation value=true{checked(values.automatic_creation)}>
Automatically create detected weeds in FarmBot</label><br>
<label>Review/recommendation confidence <input type=number name=minimum_confidence min=0 max=1 step=.01 value="{values.minimum_confidence:g}"></label><br>
<label>Automatic creation confidence <input type=number name=automatic_creation_confidence min=0 max=1 step=.01 value="{values.automatic_creation_confidence:g}"></label><br>
<label>Created weed radius (mm) <input type=number name=weed_radius_mm min=1 step=1 value="{values.weed_radius_mm:g}"></label>
</fieldset>
<fieldset><legend>Candidate size, colour and shape</legend>
<label>Minimum weed area (mm²) <input type=number name=minimum_area_mm2 min=5 step=1 value="{values.minimum_area_mm2:g}"></label><br>
<label>Maximum weed area (mm²) <input type=number name=maximum_area_mm2 min=10 step=1 value="{values.maximum_area_mm2:g}"></label><br>
<label><input type=checkbox name=shape_filter_enabled value=true{checked(values.shape_filter_enabled)}> Enable colour/shape filter</label><br>
<label>Strong-green hue range <input type=number name=green_hue_min min=0 max=179 step=1 value="{values.green_hue_min}"> to
<input type=number name=green_hue_max min=0 max=179 step=1 value="{values.green_hue_max}"></label><br>
<label>Strong-green minimum saturation <input type=number name=strong_green_minimum_saturation min=0 max=255 step=1 value="{values.strong_green_minimum_saturation}"></label><br>
<label>Strong-green minimum Excess Green <input type=number name=strong_green_minimum_excess_green min=-255 max=510 step=1 value="{values.strong_green_minimum_excess_green}"></label><br>
<label>Minimum strong-green fraction <input type=number name=minimum_green_purity min=0 max=1 step=.01 value="{values.minimum_green_purity:g}"></label><br>
<label>Minimum solidity <input type=number name=minimum_solidity min=0 max=1 step=.01 value="{values.minimum_solidity:g}"></label><br>
<label>Minimum circularity <input type=number name=minimum_circularity min=0 max=1 step=.01 value="{values.minimum_circularity:g}"></label><br>
<label>Maximum aspect ratio <input type=number name=maximum_aspect_ratio min=1 max=50 step=.1 value="{values.maximum_aspect_ratio:g}"></label>
</fieldset>
<fieldset><legend>Known crop protection</legend>
<label><input type=checkbox name=crop_protection_enabled value=true{checked(values.crop_protection_enabled)}> Protect all known and previously observed crops</label><br>
<label>Canopy radius multiplier <input type=number name=crop_support_radius_multiplier min=.5 max=5 step=.05 value="{values.crop_support_radius_multiplier:g}"></label><br>
<label>Minimum extra canopy support (mm) <input type=number name=crop_support_extra_mm min=0 max=500 step=1 value="{values.crop_support_extra_mm:g}"></label><br>
<label>Extra exclusion around plants (mm) <input type=number name=plant_exclusion_margin_mm min=0 step=1 value="{values.plant_exclusion_margin_mm:g}"></label>
</fieldset>
<fieldset><legend>Multi-image confirmation</legend>
<label><input type=checkbox name=temporal_confirmation_enabled value=true{checked(values.temporal_confirmation_enabled)}> Enable temporal confirmation</label><br>
<label>Looks before recommendation <input type=number name=recommendation_min_observations min=1 max=20 step=1 value="{values.recommendation_min_observations}"></label><br>
<label>Looks before automatic creation <input type=number name=automatic_min_observations min=1 max=20 step=1 value="{values.automatic_min_observations}"></label><br>
<label>Position matching distance (mm) <input type=number name=temporal_match_distance_mm min=1 max=250 step=1 value="{values.temporal_match_distance_mm:g}"></label><br>
<label>Maximum gap between looks (hours) <input type=number name=temporal_max_gap_hours min=1 max=8760 step=1 value="{values.temporal_max_gap_hours}"></label>
</fieldset>
<fieldset><legend>Learned visual verifier</legend>
<label><input type=checkbox name=visual_verifier_enabled value=true{checked(values.visual_verifier_enabled)}> Enable learned verifier</label><br>
<label><input type=checkbox name=visual_verifier_shadow_mode value=true{checked(values.visual_verifier_shadow_mode)}> Shadow mode (score but do not reject)</label><br>
<label><input type=checkbox name=visual_verifier_required_for_automatic value=true{checked(values.visual_verifier_required_for_automatic)}> Require verifier approval for automatic creation</label><br>
<label>Verifier confidence threshold <input type=number name=visual_verifier_minimum_confidence min=0 max=1 step=.01 value="{values.visual_verifier_minimum_confidence:g}"></label><br>
<label>Candidate recall boost while the verifier scores <input type=number name=candidate_recall_boost min=.1 max=1 step=.05 value="{values.candidate_recall_boost:g}"></label><br>
<span class=muted>Relaxes the colour/shape gates above by this factor once a trained verifier is
enforcing, so borderline weeds reach the verifier instead of being dropped by rules that cannot
tell a weed from moss. 1 keeps the gates unchanged.</span><br>
<label>Minimum weed and non-weed labels for training <input type=number name=training_minimum_per_class min=2 step=1 value="{values.training_minimum_per_class}"></label><br>
<label><input type=checkbox name=automatic_retraining value=true{checked(values.automatic_retraining)}> Retrain automatically after each new label once enough labels exist</label><br>
<label><input type=checkbox name=candidate_crop_storage_enabled value=true{checked(values.candidate_crop_storage_enabled)}> Store candidate crops for review/training</label>
</fieldset>
<fieldset><legend>Existing weed maintenance</legend>
<label><input type=checkbox name=automatic_radius_adjustment value=true{checked(values.automatic_radius_adjustment)}>
Automatically increase the radius of a matching known weed</label><br>
<label>Radius adjustment confidence <input type=number name=radius_adjustment_confidence min=0 max=1 step=.01 value="{values.radius_adjustment_confidence:g}"></label><br>
<label><input type=checkbox name=automatic_removal value=true{checked(values.automatic_removal)}>
Automatically remove known weeds that disappear</label><br>
<label>Removal confidence <input type=number name=removal_confidence min=0 max=1 step=.01 value="{values.removal_confidence:g}"></label><br>
<label>Absent images before removal <input type=number name=removal_min_consecutive_absent min=1 max=10 step=1 value="{values.removal_min_consecutive_absent}"></label>
</fieldset>
<button>Save all weed settings</button></form></section>
<section class=card><h2>Verifier training</h2>{training_notice_html}<p>{model_status}</p>
{threshold_html}
<p>Labels: {labels["weed"]} weeds · {labels["crop"]} crops ·
{labels["fallen_leaf"]} fallen leaves · {labels["mushroom"]} mushrooms ·
{labels["moss"]} moss · {labels["soil"]} soil · {labels["hardware"]} hardware.</p>
<div class=button-row><form method=post action="weed-model/train"><button>Train verifier now</button></form>
<form method=get action="weed-model/export"><button>Export labels and model</button></form>
<form method=get action="weed-model/export"><input type=hidden name=crops value=true>
<button>Export with crop images</button></form>
<form method=post action="weed-model/clear" onsubmit="return confirm('Clear all labeled training images and the trained verifier?');">
<button type=submit>Clear all training images</button></form></div>
<form method=post action="weed-model/import" enctype="multipart/form-data" class=button-row>
<label>Import a training bundle <input type=file name=bundle accept="application/json,.json" required></label>
<label><input type=checkbox name=replace value=true> Replace existing labels</label>
<button>Import</button></form>
<p class=muted>Export produces a JSON bundle of every label and its features. It is the format to
share a starter model between installs: drop an exported bundle in as
<code>bundled_weed_model.json</code> beside the source, or import it here and retrain. Crop images
are omitted unless you ask for them, because they make the file large and are only needed for
re-checking labels by eye.</p>
<p class=muted>Accepting a weed records a positive label. Rejection and the category buttons on
the Analysis page record hard negative examples from this FarmBot. Edit a tag below if a
review decision was wrong. Saving a tag does not change the detection review status.</p>
{training_samples_html}</section>
<section class=card><h2>Most informative to label next</h2>
<p class=muted>These unlabelled candidates sit closest to the verifier's decision boundary, so each
label here moves the model more than another obvious weed does.</p>
{uncertain_html}</section>"""
    return layout(request, body, "Weed settings")


@app.post("/weed-settings")
async def save_weed_settings(
    enabled: bool = Form(False),
    automatic_creation: bool = Form(False),
    automatic_radius_adjustment: bool = Form(False),
    radius_adjustment_confidence: float = Form(0.55),
    automatic_removal: bool = Form(False),
    removal_confidence: float = Form(0.6),
    removal_min_consecutive_absent: int = Form(1),
    minimum_area_mm2: float = Form(75),
    maximum_area_mm2: float = Form(2500),
    plant_exclusion_margin_mm: float = Form(35),
    crop_protection_enabled: bool = Form(False),
    crop_support_radius_multiplier: float = Form(1.2),
    crop_support_extra_mm: float = Form(25),
    shape_filter_enabled: bool = Form(False),
    green_hue_min: int = Form(25),
    green_hue_max: int = Form(100),
    strong_green_minimum_saturation: int = Form(45),
    strong_green_minimum_excess_green: int = Form(20),
    minimum_green_purity: float = Form(0.45),
    minimum_solidity: float = Form(0.25),
    minimum_circularity: float = Form(0.03),
    maximum_aspect_ratio: float = Form(7),
    minimum_confidence: float = Form(0.70),
    automatic_creation_confidence: float = Form(0.90),
    temporal_confirmation_enabled: bool = Form(False),
    recommendation_min_observations: int = Form(1),
    automatic_min_observations: int = Form(3),
    temporal_match_distance_mm: float = Form(25),
    temporal_max_gap_hours: int = Form(168),
    visual_verifier_enabled: bool = Form(False),
    visual_verifier_shadow_mode: bool = Form(False),
    visual_verifier_required_for_automatic: bool = Form(False),
    visual_verifier_minimum_confidence: float = Form(0.85),
    candidate_recall_boost: float = Form(0.6),
    training_minimum_per_class: int = Form(10),
    automatic_retraining: bool = Form(False),
    candidate_crop_storage_enabled: bool = Form(False),
    weed_radius_mm: float = Form(15),
) -> RedirectResponse:
    try:
        values = WeedSettings(
            enabled=enabled,
            automatic_creation=automatic_creation,
            automatic_radius_adjustment=automatic_radius_adjustment,
            radius_adjustment_confidence=radius_adjustment_confidence,
            automatic_removal=automatic_removal,
            removal_confidence=removal_confidence,
            removal_min_consecutive_absent=removal_min_consecutive_absent,
            minimum_area_mm2=minimum_area_mm2,
            maximum_area_mm2=maximum_area_mm2,
            plant_exclusion_margin_mm=plant_exclusion_margin_mm,
            crop_protection_enabled=crop_protection_enabled,
            crop_support_radius_multiplier=crop_support_radius_multiplier,
            crop_support_extra_mm=crop_support_extra_mm,
            shape_filter_enabled=shape_filter_enabled,
            green_hue_min=green_hue_min,
            green_hue_max=green_hue_max,
            strong_green_minimum_saturation=strong_green_minimum_saturation,
            strong_green_minimum_excess_green=strong_green_minimum_excess_green,
            minimum_green_purity=minimum_green_purity,
            minimum_solidity=minimum_solidity,
            minimum_circularity=minimum_circularity,
            maximum_aspect_ratio=maximum_aspect_ratio,
            minimum_confidence=minimum_confidence,
            automatic_creation_confidence=automatic_creation_confidence,
            temporal_confirmation_enabled=temporal_confirmation_enabled,
            recommendation_min_observations=recommendation_min_observations,
            automatic_min_observations=automatic_min_observations,
            temporal_match_distance_mm=temporal_match_distance_mm,
            temporal_max_gap_hours=temporal_max_gap_hours,
            visual_verifier_enabled=visual_verifier_enabled,
            visual_verifier_shadow_mode=visual_verifier_shadow_mode,
            visual_verifier_required_for_automatic=visual_verifier_required_for_automatic,
            visual_verifier_minimum_confidence=visual_verifier_minimum_confidence,
            candidate_recall_boost=candidate_recall_boost,
            training_minimum_per_class=training_minimum_per_class,
            automatic_retraining=automatic_retraining,
            candidate_crop_storage_enabled=candidate_crop_storage_enabled,
            weed_radius_mm=weed_radius_mm,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if values.minimum_area_mm2 > values.maximum_area_mm2:
        raise HTTPException(422, "Minimum weed area cannot exceed maximum weed area")
    if values.green_hue_min > values.green_hue_max:
        raise HTTPException(422, "Minimum green hue cannot exceed maximum green hue")
    if values.recommendation_min_observations > values.automatic_min_observations:
        raise HTTPException(
            422, "Looks before recommendation cannot exceed looks before automatic creation"
        )
    weed_settings_store.save(values)
    return RedirectResponse("weed-settings", status_code=303)


async def _train_weed_verifier() -> dict:
    values = weed_settings_store.load()
    model = await asyncio.to_thread(
        weed_verifier.train,
        database.weed_training_samples(),
        values.training_minimum_per_class,
    )
    database.record_weed_model_run(model)
    return model


async def _record_weed_label(detection_id: UUID, label: str) -> None:
    if label not in ALL_LABELS:
        raise HTTPException(422, "Unsupported training label")
    if not database.label_weed_detection(str(detection_id), label):
        raise HTTPException(404, "Weed detection not found")
    if weed_settings_store.load().automatic_retraining:
        try:
            await _train_weed_verifier()
        except ValueError:
            # Label collection intentionally starts before the minimum dataset
            # exists. The settings page shows the live counts.
            pass


@app.post("/weed-model/train")
async def train_weed_model() -> RedirectResponse:
    try:
        model = await _train_weed_verifier()
    except ValueError as exc:
        return RedirectResponse(f"../weed-settings?training={quote(str(exc))}", status_code=303)
    message = f"Trained from {model['sample_count']} labels"
    return RedirectResponse(
        f"../weed-settings?training={quote(message)}",
        status_code=303,
    )


@app.post("/weed-model/samples/{detection_id}")
async def edit_weed_training_sample(detection_id: UUID, label: str = Form(...)) -> RedirectResponse:
    if label not in ALL_LABELS:
        raise HTTPException(422, "Unsupported training label")
    if not database.update_weed_training_sample_label(str(detection_id), label):
        raise HTTPException(404, "Training sample not found")

    message = f"Updated training tag to {label.replace('_', '/')}"
    if weed_settings_store.load().automatic_retraining:
        try:
            model = await _train_weed_verifier()
            message += f" and retrained from {model['sample_count']} labels"
        except ValueError:
            message += "; retraining will start when both classes have enough labels"
    # Two levels up: this route lives at /weed-model/samples/{detection_id}.
    return RedirectResponse(f"../../weed-settings?training={quote(message)}", status_code=303)


@app.post("/weed-model/label/{detection_id}")
async def label_weed_candidate(detection_id: UUID, label: str = Form(...)) -> RedirectResponse:
    """Label an unreviewed candidate straight from the uncertainty queue.

    Unlike the review actions this only teaches the verifier; it deliberately
    leaves the detection's own review status alone so a candidate can be used
    as training data without being accepted into or removed from FarmBot.
    """
    await _record_weed_label(detection_id, label)
    message = f"Labelled candidate as {label.replace('_', '/')}"
    # Two levels up: this route lives at /weed-model/label/{detection_id}.
    return RedirectResponse(f"../../weed-settings?training={quote(message)}", status_code=303)


@app.post("/weed-model/apply-threshold")
async def apply_suggested_threshold() -> RedirectResponse:
    weed_verifier.reload()
    suggested = weed_verifier.suggested_threshold
    if suggested is None:
        return RedirectResponse(
            "../weed-settings?training=" + quote("No suggested threshold available yet"),
            status_code=303,
        )
    values = weed_settings_store.load()
    weed_settings_store.save(
        values.model_copy(update={"visual_verifier_minimum_confidence": suggested})
    )
    message = f"Verifier threshold set to {suggested:.2f}"
    return RedirectResponse(f"../weed-settings?training={quote(message)}", status_code=303)


@app.get("/weed-model/export")
async def export_weed_training(crops: bool = False) -> JSONResponse:
    """Export every label with its features, and the trained model if present.

    This is the portable artifact: features are what training consumes, so a
    bundle from one FarmBot can seed another install or be shipped as the
    add-on's starter model. Crop images are optional because they multiply the
    file size and are only useful for re-checking labels by eye.
    """
    weed_verifier.reload()
    samples = []
    for sample in database.weed_training_samples():
        entry = {
            "detection_id": sample["detection_id"],
            "label": sample["label"],
            "features": sample["features"],
            "created_at": str(sample["created_at"]),
            "image_id": sample.get("image_id"),
        }
        crop_path = sample.get("crop_path")
        if crops and crop_path:
            try:
                entry["crop_base64"] = base64.b64encode(Path(crop_path).read_bytes()).decode(
                    "ascii"
                )
            except OSError:
                LOGGER.warning("Could not read crop %s for export", crop_path)
        samples.append(entry)
    bundle = {
        "bundle_version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "app_version": __version__,
        "feature_names": list(FEATURE_NAMES),
        "samples": samples,
        "model": None if weed_verifier.is_bundled else weed_verifier.model,
    }
    filename = f"farmbot-weed-training-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    return JSONResponse(
        bundle,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/weed-model/import")
async def import_weed_training(
    bundle: Annotated[UploadFile, File()], replace: bool = Form(False)
) -> RedirectResponse:
    try:
        payload = json.loads((await bundle.read()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(422, f"Bundle is not valid JSON: {exc}") from exc
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise HTTPException(422, "Bundle contains no samples")
    accepted = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        label = sample.get("label")
        features = sample.get("features")
        detection_id = sample.get("detection_id")
        if label not in ALL_LABELS or not isinstance(features, dict) or not detection_id:
            continue
        accepted.append((str(detection_id), str(label), features))
    if not accepted:
        raise HTTPException(422, "Bundle contains no usable labelled samples")
    if replace:
        _remove_weed_training_crops(database.clear_weed_training_samples())
    imported = database.import_weed_training_samples(accepted)
    message = f"Imported {imported} labelled sample(s)"
    if weed_settings_store.load().automatic_retraining:
        try:
            model = await _train_weed_verifier()
            message += f" and retrained from {model['sample_count']} labels"
        except ValueError:
            message += "; retraining will start when both classes have enough labels"
    return RedirectResponse(f"../weed-settings?training={quote(message)}", status_code=303)


def _remove_weed_training_crops(paths: list[str]) -> int:
    """Delete only candidate crops owned by the app's artifact directory."""
    artifact_dir = (settings.data_dir / "artifacts").resolve()
    removed = 0
    for raw_path in set(paths):
        try:
            path = Path(raw_path).resolve()
        except (OSError, RuntimeError):
            continue
        if path.parent != artifact_dir:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            LOGGER.warning("Could not remove cleared weed training crop %s", path)
            continue
        removed += 1
    return removed


@app.post("/weed-model/clear")
async def clear_weed_training() -> RedirectResponse:
    paths = database.clear_weed_training_samples()
    weed_verifier.clear()
    removed = _remove_weed_training_crops(paths)
    message = f"Cleared {len(paths)} labeled training image(s) and unloaded the verifier"
    if removed != len(set(paths)):
        message += f"; removed {removed} stored crop file(s)"
    return RedirectResponse(f"../weed-settings?training={quote(message)}", status_code=303)


async def _approve_weed_detection(
    detection_id: UUID, *, allow_already_accepted: bool = False
) -> tuple[dict, int]:
    detection = database.weed_detection(str(detection_id))
    if detection is None:
        return {"status": "not_found", "message": "Weed recommendation not found"}, 404
    if allow_already_accepted and detection["status"] in ("created", "labelled"):
        return {"status": "applied", "message": "Weed was already accepted"}, 200
    if detection["status"] not in ("recommended", "observing"):
        return {"status": "not_found", "message": "Weed recommendation not found"}, 404
    verdict = zone_verdict(ZoneAspect.WEEDS, detection["x"], detection["y"])
    if not verdict.allowed:
        # The recommendation stays pending so the zones can be corrected instead
        # of losing the detection.
        return {
            "status": "conflict",
            "message": f"Weeds are not allowed at this position: {verdict.reason}",
        }, 409
    try:
        result = await client.create_weed(
            CreateWeedRequest(
                config_entry_id=detection["config_entry_id"],
                detection_id=detection_id,
                x=detection["x"],
                y=detection["y"],
                z=detection["z"],
                radius=detection["radius_mm"],
                confidence=detection["confidence"],
                apply=True,
                human_approved=True,
            )
        )
    except HomeAssistantError as exc:
        return {"status": "error", "message": f"Could not create weed: {exc}"}, 502
    if result.get("status") == "applied":
        database.update_weed_detection(str(detection_id), "created")
        await _record_weed_label(detection_id, "weed")
    return result, 200


@app.post("/weeds/{detection_id}/approve")
async def approve_weed(detection_id: UUID) -> JSONResponse:
    result, status_code = await _approve_weed_detection(detection_id)
    if status_code == 404:
        raise HTTPException(404, result["message"])
    return JSONResponse(result, status_code=status_code)


@app.post("/weeds/accept-all")
async def accept_all_weeds(request: WeedBulkAcceptRequest) -> JSONResponse:
    accepted_ids: list[str] = []
    failed: list[dict[str, str]] = []
    for detection_id in dict.fromkeys(request.detection_ids):
        result, _status_code = await _approve_weed_detection(
            detection_id, allow_already_accepted=True
        )
        detection_id_text = str(detection_id)
        if result.get("status") == "applied":
            accepted_ids.append(detection_id_text)
        else:
            failed.append(
                {
                    "detection_id": detection_id_text,
                    "message": str(result.get("message") or "Could not accept weed"),
                }
            )
    failed_ids = [item["detection_id"] for item in failed]
    if failed:
        message = f"{len(failed)} weed(s) could not be accepted"
        if accepted_ids:
            message = f"{len(accepted_ids)} weed(s) accepted; {message}"
        status = "partial"
    else:
        message = f"{len(accepted_ids)} weed(s) accepted"
        status = "applied"
    return JSONResponse(
        {
            "status": status,
            "accepted_ids": accepted_ids,
            "failed_ids": failed_ids,
            "failures": failed,
            "message": message,
        }
    )


@app.post("/weeds/{detection_id}/reject")
async def reject_weed(detection_id: UUID) -> JSONResponse:
    detection = database.weed_detection(str(detection_id))
    if detection is None:
        raise HTTPException(404, "Weed recommendation not found")
    database.reject_weed_detection(
        str(detection_id), max(20.0, float(detection["radius_mm"]) * 1.5)
    )
    await _record_weed_label(detection_id, "soil")
    return JSONResponse({"status": "rejected", "message": "Weed recommendation rejected"})


@app.post("/weeds/{detection_id}/dismiss")
async def dismiss_weed(detection_id: UUID) -> JSONResponse:
    """Discard an ambiguous detection: neither accepted, rejected, nor labelled.

    Some candidates cannot honestly be called: clipped at the image edge, or
    only a few pixels across. Forcing a decision would poison the verifier's
    training set, so nothing is recorded beyond suppressing the position.
    """
    detection = database.weed_detection(str(detection_id))
    if detection is None:
        raise HTTPException(404, "Weed recommendation not found")
    database.dismiss_weed_detection(
        str(detection_id), max(20.0, float(detection["radius_mm"]) * 1.5)
    )
    return JSONResponse(
        {"status": "dismissed", "message": "Weed recommendation discarded as unknown"}
    )


@app.post("/weeds/{detection_id}/label/{label}")
async def label_weed(detection_id: UUID, label: str) -> JSONResponse:
    detection = database.weed_detection(str(detection_id))
    if detection is None:
        raise HTTPException(404, "Weed detection not found")
    await _record_weed_label(detection_id, label)
    if label == "weed":
        database.update_weed_detection(str(detection_id), "labelled")
    else:
        database.reject_weed_detection(
            str(detection_id), max(20.0, float(detection["radius_mm"]) * 1.5)
        )
    return JSONResponse(
        {"status": "applied", "message": f"Saved {label.replace('_', '/')} training label"}
    )


def zone_verdict(aspect: ZoneAspect, x: float, y: float, radius_mm: float = 0.0) -> ZoneVerdict:
    """Evaluate a placement against the persisted zones (empty config allows)."""
    return evaluate(zone_store.zones(), aspect, x, y, radius_mm)


def _parse_polygon_points(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line in raw.replace(";", "\n").splitlines():
        entry = line.strip()
        if not entry:
            continue
        parts = [part for part in entry.replace("\t", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise HTTPException(422, f"Could not read the polygon point '{entry}'; use 'x, y'")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise HTTPException(422, f"Polygon point '{entry}' is not numeric") from exc
    return points


def _zone_json(zones: list[Zone]) -> str:
    payload = [
        {
            "zone_id": zone.zone_id,
            "name": zone.name,
            "kind": zone.kind.value,
            "shape": zone.shape.value,
            "enabled": zone.enabled,
            "min_x": zone.min_x,
            "min_y": zone.min_y,
            "max_x": zone.max_x,
            "max_y": zone.max_y,
            "center_x": zone.center_x,
            "center_y": zone.center_y,
            "radius_mm": zone.radius_mm,
            "points": [list(point) for point in zone.points],
        }
        for zone in zones
    ]
    return escape(json.dumps(payload, separators=(",", ":")), quote=True)


@app.get("/zones", response_class=HTMLResponse)
async def zones_page(request: Request) -> HTMLResponse:
    zones = zone_store.zones()

    def flag(zone: Zone, field: str, label: str) -> str:
        state = " checked" if getattr(zone, field) else ""
        return (
            f'<label><input type=checkbox form="zone-form-{zone.zone_id}" name={field} '
            f"value=true{state}> {escape(label)}</label>"
        )

    zone_forms = "".join(
        f'<form id="zone-form-{zone.zone_id}" method=post action="zones/{zone.zone_id}/update"></form>'
        f'<form id="zone-delete-{zone.zone_id}" method=post action="zones/{zone.zone_id}/delete"></form>'
        for zone in zones
    )
    zone_rows = "".join(
        f"<tr><td>{escape(zone.name)}</td>"
        f"<td>{'Boundary' if zone.kind is ZoneKind.BOUNDARY else 'Exclusion zone'}</td>"
        f"<td>{escape(zone.describe_geometry())}</td>"
        f"<td>{flag(zone, 'allow_weeds', 'Weeds allowed')}</td>"
        f"<td>{flag(zone, 'allow_plant_centers', 'Centres allowed')}</td>"
        f"<td>{flag(zone, 'allow_plant_radius', 'Radius allowed')}</td>"
        f"<td>{flag(zone, 'enabled', 'Active')}</td>"
        f'<td><div class=button-row><button form="zone-form-{zone.zone_id}">Save</button>'
        f'<button form="zone-delete-{zone.zone_id}">Delete</button></div></td></tr>'
        for zone in zones
    )
    body = f"""<section class=card><h2>Boundaries and exclusion zones</h2>
<p>Zones are areas of the garden in FarmBot coordinates (millimetres). A
<b>boundary</b> encloses where things are allowed; an <b>exclusion zone</b> marks
an area to keep clear. For each zone you choose independently whether weeds may
be placed there, whether a plant centre may be moved there, and whether a plant's
protection radius may extend into it.</p>
<p class=muted>Overlaps resolve in a fixed order: an exclusion zone that allows an
aspect is an explicit exception and wins; otherwise any zone that forbids the
aspect and is touched by the position denies it; otherwise, if at least one
boundary allows that aspect, the position must fall inside one of them. With no
zones configured nothing is restricted. Weeds, plant centres, and a boundary's
test of a protection radius are all point tests: a radius may extend past a
boundary's edge, since only the plant itself has to stay inside the growing
area. Exclusion zones are different -- they mark real hazards, so the full
protection disc must not overlap a forbidding exclusion zone.</p>
<p class=warn>Zones gate both automatic writes and manual approvals: a blocked
weed is never created, a blocked centre move and a blocked radius increase are
refused with the zone's name.</p></section>
<section class=card><h2>Add a zone</h2>
<form method=post action="zones">
<div class=grid>
<div>
<label>Name<br><input name=name maxlength=60 required placeholder="Bed 1"></label><br>
<label>Type<br><select id=kind name=kind>
<option value=boundary selected>Boundary — things may go inside</option>
<option value=exclusion>Exclusion zone — keep this area clear</option></select></label><br>
<label>Shape<br><select id=shape name=shape>
<option value=rectangle selected>Rectangle</option>
<option value=circle>Circle</option>
<option value=polygon>Polygon</option></select></label>
</div>
<div>
<div id=fields-rectangle>
<label>Corner 1 X (mm)<br><input type=number step=any name=min_x value=0></label><br>
<label>Corner 1 Y (mm)<br><input type=number step=any name=min_y value=0></label><br>
<label>Corner 2 X (mm)<br><input type=number step=any name=max_x value=1000></label><br>
<label>Corner 2 Y (mm)<br><input type=number step=any name=max_y value=1000></label>
</div>
<div id=fields-circle hidden>
<label>Centre X (mm)<br><input type=number step=any name=center_x value=0></label><br>
<label>Centre Y (mm)<br><input type=number step=any name=center_y value=0></label><br>
<label>Radius (mm)<br><input type=number step=any min=1 name=radius_mm value=500></label>
</div>
<div id=fields-polygon hidden>
<label>Points, one "X, Y" pair per line (at least three)<br>
<textarea name=points rows=6 cols=24 placeholder="0, 0&#10;1200, 0&#10;1200, 800"></textarea></label>
</div>
</div>
<div>
<p><b>Inside this zone</b></p>
<label><input type=checkbox id=new_allow_weeds name=allow_weeds value=true checked>
Weeds may be placed</label><br>
<label><input type=checkbox id=new_allow_plant_centers name=allow_plant_centers value=true checked>
Plant centres may be moved here</label><br>
<label><input type=checkbox id=new_allow_plant_radius name=allow_plant_radius value=true checked>
A plant radius may extend into it</label>
<p class=muted>Clearing a box on a boundary carves a hole in it; ticking one on an
exclusion zone makes that aspect an allowed exception inside it.</p>
<p><button>Add zone</button></p>
</div>
</div>
</form></section>
<section class=card><h2>Configured zones</h2>{zone_forms}
<table><thead><tr><th>Name</th><th>Type</th><th>Area</th><th>Weeds</th><th>Plant centres</th>
<th>Plant radius</th><th>Active</th><th>Actions</th></tr></thead>
<tbody>{zone_rows or "<tr><td colspan=8>No zones yet; nothing is restricted</td></tr>"}</tbody></table>
<p class=muted>Changing a tick box takes effect after Save.</p></section>
<section class=card><h2>Garden map</h2>
<canvas id=zone-map width=900 height=600 data-zones="{_zone_json(zones)}"
 data-entry="{escape(settings.selected_config_entry_id or "", quote=True)}"
 style="width:100%;border:1px solid #ccc"></canvas>
<div class=button-row><button id=zone-load-items type=button>Show plants &amp; FarmBot weeds</button></div>
<p id=zone-map-status class=muted></p>
<p class=legend>Green outline = boundary, red outline = exclusion zone, dashed = inactive.
Green circles = plants with their protection radius, red dots = FarmBot weeds.</p>
</section>
<script>{_ZONES_JS}</script>"""
    return layout(request, body, "Boundaries and zones")


@app.get("/api/zones")
async def zones_api() -> JSONResponse:
    return JSONResponse(zone_store.load().model_dump(mode="json"))


@app.post("/zones")
async def create_zone(
    name: str = Form(...),
    kind: str = Form("boundary"),
    shape: str = Form("rectangle"),
    allow_weeds: bool = Form(False),
    allow_plant_centers: bool = Form(False),
    allow_plant_radius: bool = Form(False),
    min_x: float = Form(0),
    min_y: float = Form(0),
    max_x: float = Form(0),
    max_y: float = Form(0),
    center_x: float = Form(0),
    center_y: float = Form(0),
    radius_mm: float = Form(0),
    points: str = Form(""),
) -> RedirectResponse:
    try:
        zone_kind, zone_shape = ZoneKind(kind), ZoneShape(shape)
    except ValueError as exc:
        raise HTTPException(400, "Unknown zone type or shape") from exc
    try:
        zone = Zone(
            name=name.strip(),
            kind=zone_kind,
            shape=zone_shape,
            allow_weeds=allow_weeds,
            allow_plant_centers=allow_plant_centers,
            allow_plant_radius=allow_plant_radius,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            center_x=center_x,
            center_y=center_y,
            radius_mm=radius_mm,
            points=_parse_polygon_points(points) if zone_shape is ZoneShape.POLYGON else [],
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    zone_store.add(zone)
    LOGGER.info("Added %s zone '%s' (%s)", zone.kind.value, zone.name, zone.describe_geometry())
    return RedirectResponse("zones", status_code=303)


@app.post("/zones/{zone_id}/update")
async def update_zone(
    zone_id: str,
    allow_weeds: bool = Form(False),
    allow_plant_centers: bool = Form(False),
    allow_plant_radius: bool = Form(False),
    enabled: bool = Form(False),
) -> RedirectResponse:
    updated = zone_store.update(
        zone_id,
        allow_weeds=allow_weeds,
        allow_plant_centers=allow_plant_centers,
        allow_plant_radius=allow_plant_radius,
        enabled=enabled,
    )
    if updated is None:
        raise HTTPException(404, "Zone not found")
    return RedirectResponse("../../zones", status_code=303)


@app.post("/zones/{zone_id}/delete")
async def delete_zone(zone_id: str) -> RedirectResponse:
    if not zone_store.delete(zone_id):
        raise HTTPException(404, "Zone not found")
    return RedirectResponse("../../zones", status_code=303)


def _calibration_warnings(calibration: Calibration | None) -> list[str]:
    """Warnings when an existing calibration may not fit the current setup."""
    warnings: list[str] = []
    if calibration is None:
        return warnings
    resolution = settings.resolution
    if calibration.processed_width and (
        calibration.processed_width != resolution.width
        or calibration.processed_height != resolution.height
    ):
        warnings.append(
            f"Calibration belongs to {calibration.processed_width}x{calibration.processed_height}; "
            f"the app is configured for {resolution.width}x{resolution.height}."
        )
    elif not calibration.processed_width:
        warnings.append(
            "Calibration has no recorded resolution and cannot be verified against the "
            "current preset; recalibration is recommended."
        )
    if calibration.source == "manual_transformed":
        warnings.append(
            "This calibration was mathematically transformed from another resolution; "
            "verify plant-centre alignment before trusting it."
        )
    return warnings


def _origin_options(selected: str) -> str:
    labels = {
        "top_left": "Top left",
        "top_right": "Top right",
        "bottom_left": "Bottom left",
        "bottom_right": "Bottom right",
    }
    return "".join(
        f"<option value={value}{' selected' if value == selected else ''}>{escape(label)}</option>"
        for value, label in labels.items()
    )


def _calibration_grid_range(entry_id: str) -> tuple[datetime, datetime]:
    """Default the preview to the latest captured grid's time window."""
    record = photo_grid_store.load()
    if record is not None and record.config_entry_id == entry_id and record.frames:
        end = record.completed_at or record.started_at + timedelta(hours=1)
        return record.started_at - timedelta(minutes=5), end + timedelta(minutes=5)
    end = datetime.now(UTC)
    return end - timedelta(hours=72), end


@app.get("/settings", response_class=HTMLResponse)
async def calibration_page(request: Request) -> HTMLResponse:
    entry_id = settings.selected_config_entry_id
    calibration = database.active_calibration(entry_id)
    resolution = settings.resolution
    warnings = _calibration_warnings(calibration)
    warning_html = "".join(f"<p class=warn>⚠ {escape(w)}</p>" for w in warnings)
    current = "none"
    if calibration is not None:
        current = (
            f"source={calibration.source}, "
            f"{calibration.pixels_per_mm_x:.4f}×{calibration.pixels_per_mm_y:.4f} px/mm, "
            f"resolution={calibration.processed_width}x{calibration.processed_height}, "
            f"rotation={calibration.rotation_degrees}°, "
            f"origin={calibration.origin_location}, "
            f"offsets=({calibration.offset_x_mm},{calibration.offset_y_mm}) mm"
        )
    # Prefill the form with the durable stored inputs so a restart shows the last
    # saved calibration ready to edit (persistence is /data-backed, not options).
    stored = calibration_store.get(entry_id) if entry_id else None
    v_scale = "" if stored is None else stored.coordinate_scale
    v_refw = 2592 if stored is None else stored.reference_width
    v_refh = 1944 if stored is None else stored.reference_height
    v_rot = 0 if stored is None else stored.rotation_degrees
    v_ox = 0 if stored is None else stored.offset_x_mm
    v_oy = 0 if stored is None else stored.offset_y_mm
    v_origin = "top_left" if stored is None else str(stored.origin_location)
    v_map_origin = "top_left" if stored is None else str(stored.map_origin)
    v_rotate_map = False if stored is None else stored.rotate_map
    grid_from, grid_to = _calibration_grid_range(entry_id)
    grid_from_value = grid_from.astimezone().strftime("%Y-%m-%dT%H:%M")
    grid_to_value = grid_to.astimezone().strftime("%Y-%m-%dT%H:%M")
    body = f"""<section class=card><h2>FarmBot calibration</h2>
<p>Copy the values from FarmBot's own camera calibration (Photos → Camera
calibration), then verify alignment against a date-filtered whole-bed grid. The app rescales
FarmBot's mm/pixel scale (measured at its native frame) to the configured
analysis resolution ({escape(resolution.label)}). Values are saved to the app's
persistent storage and restored automatically after a restart — no external
tools needed.</p>
{warning_html}
<p class=muted>Current active calibration: {escape(current)}</p>
<div class="grid calibration-grid">
<div>
<label>FarmBot config entry ID<br><input id=entry_id value="{escape(entry_id)}"></label>
<div class=button-row><label>Photos from<br><input id=date_from type=datetime-local value="{grid_from_value}"></label>
<label>Photos to<br><input id=date_to type=datetime-local value="{grid_to_value}"></label></div>
<p><button type=button id=load>Load photo grid</button></p>
<hr>
<p class=muted>Copy scale, rotation, origin, and offsets exactly as shown in FarmBot.
Copy the selected resolution from Photos → Camera settings. The preview reports each
downloaded photo's actual dimensions and warns about differences, but always shows the
photos so the values can be corrected interactively.</p>
<label>Pixel coordinate scale (mm/pixel)<br><input id=fb_scale type=number min=0 step=any value="{v_scale}"></label>
<label>FarmBot capture width (px)<br><input id=fb_refw type=number min=1 step=1 value="{v_refw}"></label>
<label>FarmBot capture height (px)<br><input id=fb_refh type=number min=1 step=1 value="{v_refh}"></label>
<p id=ppm class=muted>Enter the FarmBot pixel coordinate scale and Camera settings resolution</p>
<label>Camera rotation (degrees)<br><input id=rotation type=number step=any value="{v_rot}"></label>
<label>Origin location in image<br><select id=origin>{_origin_options(v_origin)}</select></label>
<label>Map origin<br><select id=map_origin>{_origin_options(v_map_origin)}</select></label>
<label><input type=checkbox id=rotate_map{" checked" if v_rotate_map else ""}>
 Rotate map (swap X and Y)</label>
<p class=muted>Copy both Farm Designer controls independently. Map origin reflects the
selected axes; Rotate map swaps X and Y. Together they orient the complete bed,
including photos and markers, without changing the camera transform.</p>
<label>Offset X (mm)<br><input id=offx type=number step=any value="{v_ox}"></label>
<label>Offset Y (mm)<br><input id=offy type=number step=any value="{v_oy}"></label>
<p class=muted>Copy both FarmBot camera offsets unchanged. FarmBot places the optical
centre at the photo's recorded gantry coordinate plus these offsets.</p>
<label><input type=checkbox id=showoverlay checked> Overlay plant &amp; weed centres</label><br>
<label><input type=checkbox id=showlabels checked> Show labels (name / weed)</label>
<p><label><input type=checkbox id=confirm> Centres align across the full grid</label></p>
<p><button type=button id=save disabled>Save calibration</button></p>
<p id=status class=muted></p>
</div>
<div>
<div class=button-row aria-label="Calibration grid zoom controls">
<button type=button id=zoom-out aria-label="Zoom out">−</button>
<button type=button id=zoom-in aria-label="Zoom in">+</button>
<button type=button id=zoom-reset>Reset zoom</button>
<span id=zoom-value class=muted>100%</span></div>
<div id=calibration-grid-viewport class=calibration-grid-viewport>
<canvas id=canvas width=900 height=420 aria-label="Calibration birds-eye photo grid"
 style="display:block;width:100%;background:#111"></canvas></div>
<p class=muted>Green = known plants (name · crop). Red = FarmBot weeds. Adjust the
values above and the complete grid updates live. Within the selected range, only the
newest photo at each FarmBot coordinate is shown. Photos are cropped at shared
camera-centre midpoints into gap-free rectangular cells, with the outside cells clipped
to the garden border. Zoom in, then use the scrollbars to inspect seams and marker centres.</p>
</div>
</div>
</section>
<script>{_CALIBRATION_JS}</script>"""
    return layout(request, body, "Calibration · FarmBot Vision")


def _latest_images_by_location(images: list, tolerance_mm: float = 25.0) -> list:
    """Keep the newest capture at each FarmBot grid coordinate."""
    selected = []
    for image in sorted(images, key=lambda item: item.created_at, reverse=True):
        if any(
            math.hypot(image.meta.x - other.meta.x, image.meta.y - other.meta.y) <= tolerance_mm
            for other in selected
        ):
            continue
        selected.append(image)
    return selected


@app.get("/api/vision/images")
async def vision_images(
    entry_id: str,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    if date_from is None or date_to is None:
        default_from, default_to = _calibration_grid_range(entry_id)
        date_from = date_from or default_from
        date_to = date_to or default_to
    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=UTC)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=UTC)
    if date_from > date_to:
        raise HTTPException(422, "From must be before to")
    if date_from < now - timedelta(hours=720):
        raise HTTPException(422, "Photo history is limited to the most recent 30 days")
    lookback_hours = max(1, min(720, math.ceil((now - date_from).total_seconds() / 3600)))
    try:
        inventory = await client.inventory(
            InventoryRequest(config_entry_id=entry_id, image_lookback_hours=lookback_hours)
        )
    except HomeAssistantError as exc:
        LOGGER.warning(
            "GET /api/vision/images failed: entry_id=%s (%s): %s",
            entry_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(502, "could not load images") from exc
    filtered_images = [
        image
        for image in inventory.images
        if image.processed and date_from <= image.created_at <= date_to
    ]
    images = [
        {"id": i.id, "created_at": i.created_at.isoformat(), "x": i.meta.x, "y": i.meta.y}
        for i in _latest_images_by_location(filtered_images)
    ]
    plants = [
        {
            "id": p.id,
            "name": p.name,
            "slug": p.openfarm_slug,
            "x": p.x,
            "y": p.y,
            "radius": p.radius,
        }
        for p in inventory.plants
    ]
    weeds = [
        {"id": w.id, "name": w.name, "x": w.x, "y": w.y, "radius": w.radius}
        for w in inventory.weeds
    ]
    # Farm Designer sizes the map from the bot's current axis lengths. Prefer
    # those live limits over a saved photo-grid record, which may predate a
    # firmware axis-length correction and would stretch the rendered mosaic.
    bed_bounds = None
    bed_bounds_source = None
    try:
        soil = await client.soil_points(entry_id)
        x_bounds = soil.motion.axis_bounds.get("x")
        y_bounds = soil.motion.axis_bounds.get("y")
        if x_bounds is not None and y_bounds is not None:
            bed_bounds = {"x": x_bounds, "y": y_bounds}
            bed_bounds_source = "live FarmBot axes"
    except HomeAssistantError:
        pass
    if bed_bounds is None:
        record = photo_grid_store.load()
        if record is not None and record.config_entry_id == entry_id:
            bed_bounds = record.bed_bounds
            bed_bounds_source = "saved grid fallback"
    return JSONResponse(
        {
            "images": images,
            "plants": plants,
            "weeds": weeds,
            "bed_bounds": bed_bounds,
            "bed_bounds_source": bed_bounds_source,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        }
    )


@app.get("/api/vision/image/{image_id}.jpg")
async def vision_image(entry_id: str, image_id: int) -> Response:
    try:
        response = await client.image(
            VisionImageRequest(
                config_entry_id=entry_id,
                image_id=image_id,
                max_width=settings.analysis_width,
                max_height=settings.analysis_height,
            ),
            settings.max_image_payload_bytes,
        )
    except HomeAssistantError as exc:
        LOGGER.warning(
            "GET /api/vision/image/%s.jpg failed: entry_id=%s (%s): %s",
            image_id,
            entry_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(502, "could not load image") from exc
    headers = {
        "X-FarmBot-Processed-Width": str(response.width),
        "X-FarmBot-Processed-Height": str(response.height),
    }
    if response.oriented_width is not None and response.oriented_height is not None:
        headers["X-FarmBot-Oriented-Width"] = str(response.oriented_width)
        headers["X-FarmBot-Oriented-Height"] = str(response.oriented_height)
    return Response(
        base64.b64decode(response.image_base64),
        media_type="image/jpeg",
        headers=headers,
    )


@app.post("/calibration")
async def save_calibration(
    entry_id: str = Form(...),
    coordinate_scale: float = Form(...),
    reference_width: int = Form(...),
    reference_height: int = Form(...),
    rotation: float = Form(0),
    offset_x: float = Form(0),
    offset_y: float = Form(0),
    origin_location: str = Form("top_left"),
    map_origin: str = Form("top_left"),
    rotate_map: bool = Form(False),
) -> RedirectResponse:
    """Persist the FarmBot camera calibration for a bot.

    The entered values are written to the durable /data store (the master record
    that survives restarts) and the derived processed-resolution calibration is
    made the active one in the database (the runtime source the analysis
    pipeline reads).
    """
    try:
        origin = OriginLocation(origin_location)
    except ValueError as exc:
        raise HTTPException(400, "invalid origin location") from exc
    try:
        # Direct unit-level calls do not pass through FastAPI's form decoder,
        # so an omitted Form default arrives as field metadata rather than text.
        display_origin = OriginLocation(map_origin if isinstance(map_origin, str) else "top_left")
    except ValueError as exc:
        raise HTTPException(400, "invalid map origin") from exc
    display_rotate = rotate_map if isinstance(rotate_map, bool) else False
    try:
        values = FarmbotCalibrationInput(
            coordinate_scale=coordinate_scale,
            reference_width=reference_width,
            reference_height=reference_height,
            rotation_degrees=rotation,
            origin_location=origin,
            map_origin=display_origin,
            rotate_map=display_rotate,
            offset_x_mm=offset_x,
            offset_y_mm=offset_y,
        )
        calibration = _calibration_from_input(entry_id, values)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    calibration_store.save(entry_id, values)
    database.save_calibration(entry_id, calibration)
    return RedirectResponse("settings", status_code=303)


@app.get("/artifact/{name}")
async def artifact(name: str) -> FileResponse:
    safe_name = Path(name).name
    path = settings.data_dir / "artifacts" / safe_name
    if safe_name != name or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


def _measurement_from_row(row: dict) -> Measurement:
    payload = {name: row[name] for name in Measurement.model_fields if name in row}
    try:
        payload["artifact_paths"] = json.loads(row.get("artifact_paths_json") or "[]")
    except (TypeError, json.JSONDecodeError):
        payload["artifact_paths"] = []
    return Measurement.model_validate(payload)


def _center_move_blocked(row: dict) -> str | None:
    """Refusal message when zones forbid a plant centre at the suggested point."""
    verdict = zone_verdict(
        ZoneAspect.CENTERS, row["recommended_center_x"], row["recommended_center_y"]
    )
    if verdict.allowed:
        return None
    return f"A plant centre cannot be moved there: {verdict.reason}"


def _radius_growth_blocked(row: dict) -> str | None:
    """Refusal message when zones forbid the recommended protection radius.

    Measurements recorded before the plant position was stored cannot be
    checked; those keep their previous behaviour rather than being blocked.
    """
    center_x, center_y = row.get("recorded_center_x"), row.get("recorded_center_y")
    if center_x is None or center_y is None:
        return None
    verdict = zone_verdict(
        ZoneAspect.RADIUS,
        float(center_x),
        float(center_y),
        float(row["recommended_protection_radius_mm"]),
    )
    if verdict.allowed:
        return None
    return f"The recommended radius is not allowed to extend there: {verdict.reason}"


def _action_response(
    request: Request, status: str, message: str, *, error_status: int | None = None
) -> Response:
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(
            {"status": status, "message": message},
            status_code=error_status or 200,
        )
    if error_status is not None:
        raise HTTPException(error_status, message)
    destination = ingress_base(request)
    if destination == "./":
        destination = "../../../"
    return RedirectResponse(destination, status_code=303)


@app.post("/recommendations/{measurement_id}/{action}")
async def recommendation(request: Request, measurement_id: str, action: str) -> Response:
    if action not in {"approve", "reject", "move-center"}:
        raise HTTPException(400)
    row = database.measurement(measurement_id)
    if row is None:
        raise HTTPException(404)
    if database.has_terminal_decision(measurement_id):
        return _action_response(
            request, "conflict", "This recommendation was already reviewed", error_status=409
        )
    if action == "move-center":
        if not row.get("center_misaligned") or row.get("recommended_center_x") is None:
            return _action_response(
                request, "conflict", "No centre correction is available", error_status=409
            )
        blocked = _center_move_blocked(row)
        if blocked is not None:
            return _action_response(request, "conflict", blocked, error_status=409)
        entry_id = row.get("config_entry_id") or settings.selected_config_entry_id
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=entry_id,
                image_lookback_hours=settings.image_lookback_hours,
            )
        )
        plant = next((p for p in inventory.plants if p.id == row["plant_id"]), None)
        if plant is None:
            return _action_response(
                request, "conflict", "Plant is no longer active", error_status=409
            )
        result = await client.apply_plant_center(
            ApplyPlantCenterRequest(
                config_entry_id=entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_x=plant.x,
                expected_y=plant.y,
                recommended_x=row["recommended_center_x"],
                recommended_y=row["recommended_center_y"],
                apply=True,
                human_approved=True,
            )
        )
        status = str(result.get("status", "error"))
        if status == "applied":
            database.record_decision(measurement_id, "center_moved", result)
            return _action_response(
                request,
                "updated",
                "Plant centre moved; you can still apply or reject the radius",
            )
        return _action_response(
            request,
            status,
            str(result.get("message") or status),
            error_status=409 if status == "conflict" else None,
        )
    if action == "approve":
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if not row.get("calibrated", 1):
            return _action_response(
                request, "conflict", "Calibration is required", error_status=409
            )
        if row["recommended_protection_radius_mm"] == row["current_radius_mm"]:
            database.record_group_decision(
                measurement_id,
                "approved_no_change",
                {
                    "current_radius_mm": row["current_radius_mm"],
                    "observed_radius_mm": row["recommended_protection_radius_mm"],
                },
            )
            return _action_response(
                request, "applied", "Observation approved; no radius change was needed"
            )
        blocked = _radius_growth_blocked(row)
        if blocked is not None:
            return _action_response(request, "conflict", blocked, error_status=409)
        entry_id = row.get("config_entry_id") or settings.selected_config_entry_id
        try:
            result = await client.apply_radius(
                ApplyRadiusRequest(
                    config_entry_id=entry_id,
                    plant_id=row["plant_id"],
                    measurement_id=measurement_id,
                    expected_current_radius_mm=row["current_radius_mm"],
                    recommended_radius_mm=row["recommended_protection_radius_mm"],
                    confidence=row["confidence"],
                    apply=True,
                    human_approved=True,
                )
            )
        except StaleRadiusError:
            await client.inventory(
                InventoryRequest(
                    config_entry_id=entry_id,
                    image_lookback_hours=settings.image_lookback_hours,
                )
            )
            return _action_response(
                request,
                "conflict",
                "The plant radius changed; inventory refreshed",
                error_status=409,
            )
        status = str(result.get("status", "error"))
        message = str(result.get("message") or status)
        if status != "applied":
            if status == "conflict":
                await client.inventory(
                    InventoryRequest(
                        config_entry_id=entry_id,
                        image_lookback_hours=settings.image_lookback_hours,
                    )
                )
            return _action_response(
                request,
                status,
                message,
                error_status=409 if status == "conflict" else None,
            )
        database.update_measurement_outcome(measurement_id, decision="applied", applied=True)
        database.record_group_decision(measurement_id, "applied", result)
        approved_measurement = _measurement_from_row(row)
        if approved_measurement.plant_age_days is None:
            curve_message = "skipped because plant age is unavailable"
        else:
            try:
                inventory = await client.inventory(
                    InventoryRequest(
                        config_entry_id=entry_id,
                        image_lookback_hours=settings.image_lookback_hours,
                    )
                )
                curve_result = await jobs._update_curve_after_radius(
                    entry_id, inventory, approved_measurement, human_approved=True
                )
                curve_message = str(curve_result.get("message") or curve_result.get("status", ""))
            except HomeAssistantError as exc:
                LOGGER.warning("Radius applied but curve inventory/update failed: %s", exc)
                curve_message = "deferred because inventory was unavailable"
        return _action_response(
            request,
            "applied",
            f"Radius applied. Curve update: {curve_message}",
        )
    database.record_group_decision(measurement_id, "reject", {})
    return _action_response(request, "rejected", "Recommendation rejected")


@app.post("/removals/{measurement_id}/{action}")
async def removal(request: Request, measurement_id: str, action: str) -> Response:
    if action not in {"approve", "keep", "move-center"}:
        raise HTTPException(400)
    row = database.measurement(measurement_id)
    if row is None:
        raise HTTPException(404)
    if database.has_terminal_decision(measurement_id):
        return _action_response(
            request, "conflict", "This removal was already reviewed", error_status=409
        )
    entry_id = row.get("config_entry_id") or settings.selected_config_entry_id
    if not database.is_latest_plant_measurement(entry_id, row["plant_id"], measurement_id):
        return _action_response(
            request,
            "conflict",
            "A newer canopy observation exists; removal was not applied",
            error_status=409,
        )
    if action == "keep":
        database.record_group_decision(measurement_id, "keep", {})
        return _action_response(request, "rejected", "Plant kept")
    if action == "move-center":
        if not row.get("center_misaligned") or row.get("recommended_center_x") is None:
            return _action_response(
                request, "conflict", "No centre correction is available", error_status=409
            )
        blocked = _center_move_blocked(row)
        if blocked is not None:
            return _action_response(request, "conflict", blocked, error_status=409)
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=entry_id,
                image_lookback_hours=settings.image_lookback_hours,
            )
        )
        plant = next((p for p in inventory.plants if p.id == row["plant_id"]), None)
        if plant is None:
            return _action_response(
                request, "conflict", "Plant is no longer active", error_status=409
            )
        result = await client.apply_plant_center(
            ApplyPlantCenterRequest(
                config_entry_id=entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_x=plant.x,
                expected_y=plant.y,
                recommended_x=row["recommended_center_x"],
                recommended_y=row["recommended_center_y"],
                apply=True,
                human_approved=True,
            )
        )
        status = str(result.get("status", "error"))
        if status == "applied":
            database.record_group_decision(measurement_id, "keep", result)
        return _action_response(
            request,
            status,
            str(result.get("message") or status),
            error_status=409 if status != "applied" else None,
        )
    # Radius is not part of the removal itself, but the companion service uses
    # it as an optimistic-concurrency token. Refresh it immediately before the
    # request so an older vision measurement does not make every otherwise
    # valid, explicitly approved removal fail as stale.
    inventory = await client.inventory(
        InventoryRequest(
            config_entry_id=entry_id,
            image_lookback_hours=settings.image_lookback_hours,
        )
    )
    plant = next((p for p in inventory.plants if p.id == row["plant_id"]), None)
    if plant is None:
        return _action_response(request, "conflict", "Plant is no longer active", error_status=409)
    try:
        result = await client.apply_removal(
            ApplyRemovalRequest(
                config_entry_id=entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_current_radius_mm=plant.radius,
                confidence=row["confidence"],
                apply=True,
                human_approved=True,
            )
        )
    except StaleRadiusError:
        return _action_response(
            request, "conflict", "The plant changed; removal was not applied", error_status=409
        )
    status = str(result.get("status", "error"))
    message = str(result.get("message") or status)
    if status != "applied":
        return _action_response(request, status, message, error_status=409)
    database.update_measurement_outcome(measurement_id, decision="removed", applied=True)
    database.record_group_decision(measurement_id, "removed", result)
    return _action_response(request, "applied", message)


@app.post("/curve-proposals/{proposal_id}/{action}")
async def curve_proposal_action(
    request: Request,
    proposal_id: int,
    action: str,
    value: float | None = Form(None),
    confirm_override: bool = Form(False),
) -> Response:
    if action not in {"apply", "discard-new", "discard-old"}:
        raise HTTPException(400)
    proposal = database.curve_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(404)
    if proposal["status"] != "flagged":
        return _action_response(
            request, "conflict", "This proposal was already reviewed", error_status=409
        )
    if action == "discard-new":
        database.update_curve_proposal(proposal_id, "rejected")
        return _action_response(request, "rejected", "New curve value discarded")
    previous = json.loads(proposal["previous_data_json"] or "{}")
    proposed = json.loads(proposal["data_json"] or "{}")
    day = int(proposal["plant_age_days"])
    new_value = float(value if value is not None else proposed[str(day)])
    base = dict(previous)
    if action == "discard-old" and proposal["conflict_day"] is not None:
        base.pop(str(proposal["conflict_day"]), None)
    edit = propose_curve_point(
        base,
        day,
        new_value,
        max_daily_growth_mm=settings.maximum_daily_radius_growth_mm,
        maximum_plant_radius_mm=settings.maximum_plant_radius_mm,
    )
    if edit.verdict == "flagged" and not confirm_override:
        return _action_response(
            request,
            "conflict",
            f"Edited value is still flagged: {edit.reason}; confirm the override to apply",
            error_status=409,
        )
    entry_id = proposal["config_entry_id"] or settings.selected_config_entry_id
    inventory = await client.inventory(
        InventoryRequest(
            config_entry_id=entry_id,
            image_lookback_hours=settings.image_lookback_hours,
        )
    )
    plant = next((item for item in inventory.plants if item.id == proposal["plant_id"]), None)
    assigned = (
        None
        if plant is None
        else next((item for item in inventory.curves if item.id == plant.spread_curve_id), None)
    )
    if plant is None or assigned is None or assigned.data != previous:
        return _action_response(
            request,
            "conflict",
            "The plant's assigned curve changed after this proposal was created",
            error_status=409,
        )
    curve_data = {
        control_day: float(round(diameter)) for control_day, diameter in edit.data.items()
    }
    result = await client.upsert_curve(
        UpsertCurveRequest(
            config_entry_id=entry_id,
            crop_slug=proposal["crop_slug"],
            curve_id=proposal["farmbot_curve_id"],
            name=proposal["curve_name"],
            data=curve_data,
            assign_to_plant_ids=[proposal["plant_id"]],
            apply=True,
            human_approved=True,
        )
    )
    status = str(result.get("status", "error"))
    message = str(result.get("message") or status)
    if status != "applied":
        return _action_response(request, status, message, error_status=409)
    database.update_curve_proposal(proposal_id, "applied", curve_data)
    database.record_decision(proposal["measurement_id"], "curve_applied", result)
    return _action_response(request, "applied", message)
