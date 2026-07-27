from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

import pytest
import yaml

import farmbot_vision.web as web
from farmbot_vision.models import (
    BotList,
    Decision,
    Inventory,
    Measurement,
    SoilMeasurement,
    SoilMotionState,
    SoilPoint,
    SoilPointInventory,
    SoilSite,
    VisionRequestEvent,
)


@pytest.mark.asyncio
async def test_grid_repair_requires_advertised_v2_capability(monkeypatch):
    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                    }
                ]
            }
        )

    monkeypatch.setattr(web.client, "list_bots", list_bots)
    try:
        with pytest.raises(
            web.HomeAssistantError,
            match="requires FarmBot integration V2.0.0",
        ):
            await web._require_grid_repair_capability()
    finally:
        web.settings.selected_config_entry_id = previous


@pytest.mark.asyncio
async def test_grid_repair_accepts_advertised_v2_capability(monkeypatch):
    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                        "integration_version": "2.0.0",
                        "capabilities": ["photo_grid_repair"],
                    }
                ]
            }
        )

    monkeypatch.setattr(web.client, "list_bots", list_bots)
    try:
        await web._require_grid_repair_capability()
    finally:
        web.settings.selected_config_entry_id = previous


@pytest.mark.asyncio
async def test_grid_repair_recheck_forces_a_fresh_inventory_lookup(monkeypatch):
    """The Recheck button must not be blocked by the 5-minute inspection cache
    or by the photo_grid_repair capability check -- it only reads the grid
    state, so it must work even when a repair itself can't be started.
    """
    import base64 as b64
    from types import SimpleNamespace

    import cv2
    import numpy as np

    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"
    web.grid_repair_state.update(run=None, checked_at=None, error="", message="")

    def _image(image_id: int, minute: int, x: float, y: float) -> dict:
        return {
            "id": image_id,
            "created_at": f"2026-07-27T01:{minute:02d}:00+00:00",
            "x": x,
            "y": y,
            "z": 0,
        }

    async def inventory(_request):
        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": datetime.now(UTC),
                "plants": [],
                "images": [
                    _image(1, 0, 0, 0),
                    _image(2, 1, 0, 100),
                    _image(3, 2, 0, 200),
                    _image(4, 3, 100, 0),
                    _image(5, 4, 100, 100),
                    # (100, 200) is missing.
                ],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    plain_frame = np.full((120, 160, 3), (35, 90, 35), np.uint8)
    _, encoded = cv2.imencode(".jpg", plain_frame)
    plain_jpeg_base64 = b64.b64encode(encoded.tobytes()).decode()

    async def image(_request, _max_bytes):
        return SimpleNamespace(image_base64=plain_jpeg_base64)

    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "image", image)
    try:
        status, headers, _ = await asgi_request("/grid-repair/recheck", method="POST")
    finally:
        web.settings.selected_config_entry_id = previous

    assert status == 303
    location = unquote(headers[b"location"].decode())
    assert "Found 1 missing and 0 gantry" in location


@pytest.mark.asyncio
async def test_grid_repair_caps_targets_at_the_service_limit(monkeypatch):
    """start_vision_grid_repair accepts at most MAX_REPAIR_TARGETS_PER_CALL
    targets per call (docs/integration-contract.md: "one to twelve"). A grid
    large enough to have more missing cells than that must be sent in a
    capped first batch, not all at once -- an uncapped request is rejected
    by the integration's schema with a bare HTTP 400.
    """
    import base64 as b64
    from types import SimpleNamespace

    import cv2
    import numpy as np

    previous = web.settings.selected_config_entry_id
    web.settings.selected_config_entry_id = "entry-1"
    web.grid_repair_state.update(run=None, checked_at=None, error="", message="")

    # A 6x6 grid (36 expected cells) with a 3x4 block plus one extra cell
    # missing (13 missing, exceeding the 12-per-call cap) while every row and
    # column keeps at least one present image, so all 6 x/y axis positions are
    # still detected -- an entirely absent row/column wouldn't register as
    # "missing" at all, since the grid axes are derived from present images.
    xs = [i * 100 for i in range(6)]
    ys = [i * 100 for i in range(6)]
    missing_cells = {(x, y) for x in xs[3:] for y in ys[2:]} | {(xs[2], ys[5])}
    present_cells = [(x, y) for x in xs for y in ys if (x, y) not in missing_cells]

    async def list_bots():
        return BotList.model_validate(
            {
                "bots": [
                    {
                        "config_entry_id": "entry-1",
                        "device_id": "42",
                        "name": "FarmBot",
                        "integration_version": "2.0.0",
                        "capabilities": ["photo_grid_repair"],
                    }
                ]
            }
        )

    async def inventory(_request):
        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": datetime.now(UTC),
                "plants": [],
                "images": [
                    {
                        "id": index,
                        "created_at": f"2026-07-27T01:{index % 60:02d}:00+00:00",
                        "x": x,
                        "y": y,
                        "z": 0,
                    }
                    for index, (x, y) in enumerate(present_cells)
                ],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    plain_frame = np.full((120, 160, 3), (35, 90, 35), np.uint8)
    _, encoded = cv2.imencode(".jpg", plain_frame)
    plain_jpeg_base64 = b64.b64encode(encoded.tobytes()).decode()

    async def image(_request, _max_bytes):
        return SimpleNamespace(image_base64=plain_jpeg_base64)

    captured_targets = []

    async def start_grid_repair(_entry_id, targets):
        captured_targets.append(targets)
        return {"status": "queued", "repair_id": "r1", "message": "Photo-grid repair queued"}

    monkeypatch.setattr(web.client, "list_bots", list_bots)
    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "image", image)
    monkeypatch.setattr(web.client, "start_grid_repair", start_grid_repair)
    try:
        result = await web.start_photo_grid_repair()
    finally:
        web.settings.selected_config_entry_id = previous

    assert len(captured_targets[0]) == web.MAX_REPAIR_TARGETS_PER_CALL
    assert result["status"] == "queued"
    assert "1 more cell(s) need a follow-up repair" in result["message"]


def _review_measurement(**updates) -> Measurement:
    values = {
        "measurement_id": uuid4(),
        "config_entry_id": "review-bot",
        "plant_id": 812,
        "crop_slug": "lettuce",
        "image_id": 19,
        "image_timestamp": datetime.now(UTC),
        "current_radius_mm": 40,
        "typical_canopy_radius_mm": 45,
        "maximum_accepted_canopy_radius_mm": 50,
        "recommended_protection_radius_mm": 70,
        "confidence": 0.42,
        "decision": Decision.RECOMMENDED,
        "reason": "safe radius increase",
        "algorithm_version": "test",
    }
    values.update(updates)
    return Measurement(**values)


async def asgi_request(
    path: str,
    *,
    method: str = "GET",
    query_string: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict[bytes, bytes], bytes]:
    messages: list[dict] = []
    body = bytearray()

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)
        if message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    encoded_path = path.encode("ascii")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": encoded_path,
        "query_string": query_string,
        "headers": headers or [],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    await web.app(scope, receive, send)
    response = next(message for message in messages if message["type"] == "http.response.start")
    return response["status"], dict(response["headers"]), bytes(body)


@pytest.mark.asyncio
async def test_soil_height_page_lists_points_and_warns_below_three(monkeypatch):
    async def soil_points(_entry_id):
        return SoilPointInventory(
            device_id="42",
            generated_at=datetime.now(UTC),
            points=[
                SoilPoint(
                    id=70,
                    name="Clear soil",
                    x=100,
                    y=200,
                    z=-400,
                    updated_at=datetime(2026, 7, 1, tzinfo=UTC),
                ),
            ],
            motion=SoilMotionState(
                connected=True,
                busy=False,
                locked=False,
                position={"x": 0, "y": 0, "z": 0},
                z_direction=-1,
                axis_bounds={"x": (0, 1000), "y": (0, 1000), "z": (-500, 0)},
            ),
        )

    monkeypatch.setattr(web.settings, "selected_config_entry_id", "soil-entry")

    async def safe_sites(_entry_id, _baseline):
        inventory = await soil_points(_entry_id)
        return inventory, [
            SoilSite(
                point_id=70,
                point_name="Clear soil",
                expected_x=100,
                expected_y=200,
                expected_z=-400,
                point_updated_at=datetime(2026, 7, 1, tzinfo=UTC),
                capture_x=175,
                capture_y=200,
                relocation_distance_mm=75,
                clearance_mm=20,
            )
        ]

    monkeypatch.setattr(web.soil_jobs, "safe_sites", safe_sites)
    status, _, body = await asgi_request("/soil-height")
    assert status == 200
    assert b"Clear soil" in body
    assert b"Measure selected" in body
    assert b"Fewer than three stale soil points" in body
    assert b"replace the assigned stale point" in body


@pytest.mark.asyncio
async def test_soil_apply_is_human_approved_and_audited(monkeypatch):
    measurement = SoilMeasurement(
        measurement_id=uuid4(),
        config_entry_id="soil-entry",
        point_id=72,
        point_name="Soil 72",
        expected_x=100,
        expected_y=200,
        old_z_mm=-400,
        point_updated_at=datetime(2026, 7, 1, tzinfo=UTC),
        capture_x=125,
        capture_y=225,
        relocation_distance_mm=35.36,
        proposed_z_mm=-395,
        confidence=0.9,
        uncertainty_mm=3,
        status="valid",
        reason="passed",
    )
    web.database.save_soil_measurement(measurement)
    seen = {}

    async def apply_soil_height(request):
        seen.update(request.model_dump(mode="json"))
        return {"status": "applied", "message": "updated"}

    monkeypatch.setattr(web.client, "apply_soil_height", apply_soil_height)
    result = await web._apply_soil_measurement(str(measurement.measurement_id))
    assert result["status"] == "applied"
    assert seen["apply"] is True
    assert seen["human_approved"] is True
    assert web.database.soil_measurement(str(measurement.measurement_id))["status"] == "applied"


@pytest.mark.asyncio
async def test_root_and_duplicate_leading_slash_routes():
    status, _, body = await asgi_request("/")
    assert status == 200
    assert b"FarmBot Vision" in body

    for path in ("//", "///"):
        status, _, body = await asgi_request(path)
        assert status == 200
        assert b"FarmBot Vision" in body


@pytest.mark.asyncio
async def test_duplicate_slashes_reach_health_and_settings():
    status, _, body = await asgi_request("//api/health", query_string=b"check=//")
    assert status == 200
    assert json.loads(body)["status"] == "ok"

    status, _, body = await asgi_request("///settings")
    assert status == 200
    assert b"FarmBot calibration" in body

    status, _, body = await asgi_request("/health")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


@pytest.mark.asyncio
async def test_normal_path_is_unchanged_and_query_string_survives():
    status, _, body = await asgi_request("/api/health", query_string=b"check=//")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


@pytest.mark.asyncio
async def test_post_duplicate_path_works(monkeypatch: pytest.MonkeyPatch):
    async def fake_run(*args, **kwargs):
        return {"accepted": True}

    monkeypatch.setattr(web.jobs, "run", fake_run)
    status, headers, _ = await asgi_request("//analyse", method="POST")
    assert status == 303
    assert headers[b"location"] == b"./"


@pytest.mark.asyncio
async def test_approval_json_reports_rejection_without_recording_a_false_success(monkeypatch):
    measurement = _review_measurement()
    web.database.save_measurements([measurement])
    calls = []

    async def rejected(request):
        calls.append(request)
        return {"status": "rejected", "message": "FarmBot declined this change"}

    monkeypatch.setattr(web.client, "apply_radius", rejected)
    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body) == {
        "status": "rejected",
        "message": "FarmBot declined this change",
    }
    assert calls[0].human_approved is True
    decisions = [
        row
        for row in web.database.recent_decisions()
        if row["measurement_id"] == str(measurement.measurement_id)
    ]
    assert decisions == []


@pytest.mark.asyncio
async def test_approval_json_records_applied_and_html_post_still_redirects(monkeypatch):
    json_measurement = _review_measurement()
    html_measurement = _review_measurement()
    web.database.save_measurements([json_measurement, html_measurement])

    async def applied(_request):
        return {"status": "applied", "message": "radius updated"}

    monkeypatch.setattr(web.client, "apply_radius", applied)
    status, _, body = await asgi_request(
        f"/recommendations/{json_measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 200
    assert json.loads(body)["status"] == "applied"
    decisions = [
        row
        for row in web.database.recent_decisions()
        if row["measurement_id"] == str(json_measurement.measurement_id)
    ]
    assert [row["action"] for row in decisions] == ["applied"]

    status, headers, _ = await asgi_request(
        f"/recommendations/{html_measurement.measurement_id}/approve", method="POST"
    )
    assert status == 303
    assert b"location" in headers


@pytest.mark.asyncio
async def test_uncertain_measurement_is_manually_reviewable_and_applicable(monkeypatch):
    measurement = _review_measurement(
        decision=Decision.UNCERTAIN,
        reason="overlap lowers confidence for automation",
    )
    web.database.save_measurements([measurement])
    calls = []

    async def applied(request):
        calls.append(request)
        return {"status": "applied", "message": "radius verified"}

    monkeypatch.setattr(web.client, "apply_radius", applied)
    status, _, dashboard = await asgi_request("/")
    assert status == 200
    html = dashboard.decode()
    assert str(measurement.measurement_id) in html
    assert "Not reviewable" not in html
    assert "Apply radius" in html

    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 200
    assert json.loads(body)["status"] == "applied"
    assert calls[0].human_approved is True


@pytest.mark.asyncio
async def test_manual_approval_can_apply_a_radius_reduction(monkeypatch):
    measurement = _review_measurement(
        current_radius_mm=100,
        recommended_protection_radius_mm=90,
        confidence=0.5,
        decision=Decision.UNCERTAIN,
        reason="large radius reduction requires human review",
    )
    web.database.save_measurements([measurement])
    calls = []

    async def applied(request):
        calls.append(request)
        return {"status": "applied", "message": "radius updated"}

    monkeypatch.setattr(web.client, "apply_radius", applied)
    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body)["status"] == "applied"
    assert calls[0].recommended_radius_mm == 90
    assert calls[0].human_approved is True


@pytest.mark.asyncio
async def test_dashboard_modal_uses_artifact_manifest_and_pending_rows(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    overlay = artifact_dir / "review-overlay.jpg"
    mask = artifact_dir / "review-mask.png"
    overlay.write_bytes(b"overlay")
    mask.write_bytes(b"mask")
    measurement = _review_measurement(
        overlay_path=str(overlay),
        mask_path=str(mask),
        artifact_paths=[str(overlay), str(mask)],
    )
    web.database.save_measurements([measurement])
    monkeypatch.setattr(web.settings, "data_dir", tmp_path)

    status, _, body = await asgi_request("/")
    html = body.decode()
    assert status == 200
    assert "id=overlay-modal" in html
    assert "data-artifacts=" in html
    assert "artifact/review-overlay.jpg" in html
    assert "artifact/review-mask.png" in html

    # The modal still relies on the deliberately restricted artifact route.
    status, _, served = await asgi_request("/artifact/review-overlay.jpg")
    assert status == 200
    assert served == b"overlay"

    web.database.record_decision(str(measurement.measurement_id), "applied", {})
    assert str(measurement.measurement_id) not in {
        row["measurement_id"] for row in web.database.pending_measurements()
    }


@pytest.mark.asyncio
async def test_weed_settings_page_exposes_pipeline_training_and_automation_controls():
    status, _, body = await asgi_request("/weed-settings")
    html = body.decode()

    assert status == 200
    assert "Candidate size, colour and shape" in html
    assert "Known crop protection" in html
    assert "Multi-image confirmation" in html
    assert "Learned visual verifier" in html
    assert "name=automatic_creation_confidence" in html
    assert 'action="weed-model/train"' in html


@pytest.mark.asyncio
async def test_canopy_settings_page_exposes_fusion_and_automation_controls():
    status, _, body = await asgi_request("/canopy-settings")
    html = body.decode()

    assert status == 200
    assert "Multi-image canopy fusion" in html
    assert "name=always_fuse_when_available" in html
    assert "name=minimum_angular_coverage" in html
    assert "name=automatic_requires_reliable_fusion" in html
    assert "name=save_diagnostics" in html


@pytest.mark.asyncio
async def test_dashboard_plant_view_uses_only_clean_and_mask_composites(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    clean = artifact_dir / "plant-composite.jpg"
    composite_overlay = artifact_dir / "plant-composite-overlay.jpg"
    diagnostic_overlay = artifact_dir / "frame-overlay.jpg"
    raw_mask = artifact_dir / "frame-mask.png"
    for path in (clean, composite_overlay, diagnostic_overlay, raw_mask):
        path.write_bytes(b"image")
    measurement = _review_measurement(
        composite_path=str(clean),
        composite_overlay_path=str(composite_overlay),
        overlay_path=str(diagnostic_overlay),
        mask_path=str(raw_mask),
        artifact_paths=[str(diagnostic_overlay), str(raw_mask)],
    )
    web.database.save_measurements([measurement])
    monkeypatch.setattr(web.settings, "data_dir", tmp_path)

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert 'data-composite-clean="artifact/plant-composite.jpg"' in html
    assert 'data-composite-overlay="artifact/plant-composite-overlay.jpg"' in html
    assert "Original images" in html
    assert "Show mask overlay" in html
    assert "artifact/frame-overlay.jpg" not in html
    assert "artifact/frame-mask.png" not in html


@pytest.mark.asyncio
async def test_missing_plant_table_shows_crop_and_center_coordinates():
    measurement = _review_measurement(
        vegetation_absent=True,
        center_misaligned=True,
        recorded_center_x=123.4,
        recorded_center_y=567.8,
        recommended_center_px=(130.5, 575.25),
        absent_observations=2,
    )
    web.database.save_measurements([measurement])

    status, _, body = await asgi_request("/")
    html = body.decode()

    assert status == 200
    assert "<th>Crop</th>" in html
    assert "Recorded center (X, Y mm)" in html
    assert "Move center to (X, Y mm)" in html
    assert "<td>lettuce</td>" in html
    assert "<td>X 123.4, Y 567.8</td>" in html
    assert "<td>X 130.5, Y 575.2</td>" in html
    assert "<th>Plant</th><th>Absent looks</th>" not in html


@pytest.mark.asyncio
async def test_removal_approval_uses_fresh_inventory_radius(monkeypatch):
    measurement = _review_measurement(
        vegetation_absent=True,
        absent_observations=2,
        current_radius_mm=40,
        recommended_protection_radius_mm=0,
    )
    web.database.save_measurements([measurement])
    calls = []

    async def inventory(_request):
        return Inventory.model_validate(
            {
                "device_id": "42",
                "generated_at": datetime.now(UTC),
                "plants": [
                    {
                        "id": measurement.plant_id,
                        "name": "Lettuce",
                        "openfarm_slug": "lettuce",
                        "x": 100,
                        "y": 200,
                        "radius": 55,
                        "plant_stage": "planted",
                    }
                ],
                "images": [],
                "curves": [],
                "camera_calibration": {"available": False},
            }
        )

    async def apply_removal(request):
        calls.append(request)
        return {"status": "applied", "message": "Plant removed"}

    monkeypatch.setattr(web.client, "inventory", inventory)
    monkeypatch.setattr(web.client, "apply_removal", apply_removal)

    status, _, body = await asgi_request(
        f"/removals/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )

    assert status == 200
    assert json.loads(body)["status"] == "applied"
    assert calls[0].expected_current_radius_mm == 55
    assert calls[0].human_approved is True


@pytest.mark.asyncio
async def test_event_listener_targets_new_image_and_uses_configured_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    async def events():
        yield VisionRequestEvent(config_entry_id="entry-1", device_id="device_42", image_id=99)

    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        return {"accepted": True}

    monkeypatch.setattr(web.client, "vision_events", events)
    monkeypatch.setattr(web.jobs, "run", fake_run)
    monkeypatch.setattr(web.settings, "mode", web.OperatingMode.RECOMMEND)
    await web.event_listener()
    assert calls == [
        {
            "entry_id": "entry-1",
            "mode": web.OperatingMode.RECOMMEND,
            "plant_ids": [],
            "image_ids": [99],
            "trigger": "new_image",
            "queue_if_busy": True,
        }
    ]


@pytest.mark.asyncio
async def test_startup_auto_selects_only_loaded_farmbot(monkeypatch: pytest.MonkeyPatch):
    async def list_bots():
        return BotList.model_validate(
            {"bots": [{"config_entry_id": "entry-1", "device_id": "42", "name": "FarmBot"}]}
        )

    monkeypatch.setattr(web.settings, "selected_config_entry_id", "")
    monkeypatch.setattr(web.client, "list_bots", list_bots)
    await web.resolve_config_entry()
    assert web.settings.selected_config_entry_id == "entry-1"


@pytest.mark.asyncio
async def test_ingress_html_uses_relative_links_without_logging_session(
    caplog: pytest.LogCaptureFixture,
):
    ingress_path = "/api/hassio_ingress/temporary-session-id/"
    status, _, body = await asgi_request(
        "/",
        headers=[(b"x-ingress-path", ingress_path.encode("ascii"))],
    )
    html = body.decode()
    assert status == 200
    assert f'<base href="{ingress_path}">' in html
    assert 'href="/settings"' not in html
    assert 'href="/api/health"' not in html
    assert 'action="/analyse"' not in html
    assert "//" not in html.replace("http://", "")
    assert ingress_path not in caplog.text

    _, _, settings_body = await asgi_request(
        "/settings",
        headers=[(b"x-ingress-path", ingress_path.encode("ascii"))],
    )
    settings_html = settings_body.decode()
    assert "fetch('api/vision/images" in settings_html
    assert "image.src='api/vision/image/" in settings_html
    assert "f.action='calibration'" in settings_html
    assert 'href="/settings"' not in settings_html


def test_direct_asgi_middleware_normalizes_scope_without_touching_query():
    captured: dict = {}

    async def downstream(scope, receive, send):
        captured.update(scope)

    middleware = web.NormalizeIngressPathMiddleware(downstream)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "path": "//api/health",
        "raw_path": b"//api/health",
        "query_string": b"token=//preserve",
        "headers": [(b"x-ingress-path", b"/api/hassio_ingress/session")],
    }
    import asyncio

    asyncio.run(middleware(scope, receive, send))
    assert captured["path"] == "/api/health"
    assert captured["raw_path"] == b"/api/health"
    assert captured["query_string"] == b"token=//preserve"
    assert captured["headers"] == scope["headers"]


@pytest.mark.asyncio
async def test_save_calibration_farmbot_values_branch():
    from farmbot_vision.models import OriginLocation

    response = await web.save_calibration(
        entry_id="botFB",
        rotation=-31.9,
        offset_x=0,
        offset_y=0,
        origin_location="top_right",
        coordinate_scale=0.242,
        reference_width=2592,
        reference_height=1944,
    )
    assert response.status_code == 303
    calibration = web.database.active_calibration("botFB")
    assert calibration is not None
    assert calibration.source == "manual"
    assert calibration.origin_location == OriginLocation.TOP_RIGHT
    assert calibration.rotation_degrees == -31.9
    width = web.settings.resolution.width
    assert calibration.pixels_per_mm_x == pytest.approx((1 / 0.242) * width / 2592)


@pytest.mark.asyncio
async def test_save_calibration_persists_to_data_store():
    """Saved FarmBot inputs are written verbatim to the durable /data store."""
    await web.save_calibration(
        entry_id="botStore",
        rotation=12.0,
        offset_x=3,
        offset_y=-4,
        origin_location="bottom_left",
        coordinate_scale=0.3,
        reference_width=2592,
        reference_height=1944,
    )
    stored = web.calibration_store.get("botStore")
    assert stored is not None
    assert stored.coordinate_scale == 0.3
    assert stored.rotation_degrees == 12.0
    assert stored.offset_x_mm == 3
    assert str(stored.origin_location) == "bottom_left"


@pytest.mark.asyncio
async def test_save_calibration_rejects_nonpositive_scale():
    with pytest.raises((web.HTTPException, ValueError)):
        await web.save_calibration(
            entry_id="botFB",
            coordinate_scale=0,
            reference_width=2592,
            reference_height=1944,
        )


@pytest.mark.asyncio
async def test_save_calibration_rejects_unknown_origin():
    with pytest.raises(web.HTTPException) as exc:
        await web.save_calibration(
            entry_id="botFB",
            origin_location="middle",
            coordinate_scale=0.242,
            reference_width=2592,
            reference_height=1944,
        )
    assert exc.value.status_code == 400


def test_app_config_uses_default_ingress_entry():
    config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    assert config["ingress"] is True
    assert config["ingress_port"] == 8099
    assert config["panel_icon"] == "mdi:sprout"
    assert config["panel_title"] == "FarmBot Vision"
    assert config["homeassistant_api"] is True
    assert "ingress_entry" not in config
