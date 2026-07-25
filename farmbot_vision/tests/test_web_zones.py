from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.parse import urlencode
from uuid import uuid4

import pytest
from test_web import _review_measurement, asgi_request

import farmbot_vision.web as web
from farmbot_vision.zones import Zone, ZoneKind, ZoneSet, ZoneShape


async def post_form(path: str, fields: dict[str, str]) -> tuple[int, dict[bytes, bytes], bytes]:
    """POST an HTML form exactly as the zones page does."""
    body = urlencode(fields).encode()
    messages: list[dict] = []
    response_body = bytearray()
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)
        if message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/x-www-form-urlencoded"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    await web.app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    return start["status"], dict(start["headers"]), bytes(response_body)


@pytest.fixture(autouse=True)
def empty_zone_store():
    """Every test starts from no zones and leaves none behind."""
    web.zone_store.save(ZoneSet())
    yield
    web.zone_store.save(ZoneSet())


def _exclusion(**updates) -> Zone:
    values = {
        "name": "Greenhouse path",
        "kind": ZoneKind.EXCLUSION,
        "shape": ZoneShape.RECTANGLE,
        "min_x": 0,
        "min_y": 0,
        "max_x": 1_000,
        "max_y": 1_000,
    }
    values.update(updates)
    return Zone(**values)


def _weed_detection(x: float, y: float) -> str:
    detection_id = str(uuid4())
    web.database.save_weed_detection(
        detection_id=detection_id,
        config_entry_id="zone-bot",
        image_id=7,
        image_timestamp=datetime.now(UTC),
        x=x,
        y=y,
        z=0,
        area_mm2=120,
        radius_mm=15,
        confidence=0.9,
        overlay_path=None,
        status="recommended",
    )
    return detection_id


@pytest.mark.asyncio
async def test_zones_page_lists_zones_and_is_linked_from_every_page():
    web.zone_store.add(_exclusion(name="Water tank"))

    status, _, body = await asgi_request("/zones")
    html = body.decode()
    assert status == 200
    assert "Boundaries and exclusion zones" in html
    assert "Water tank" in html
    assert "rectangle X 0…1000, Y 0…1000 mm" in html
    # The map receives the zone geometry without needing another request.
    assert "data-zones=" in html
    assert "&quot;name&quot;:&quot;Water tank&quot;" in html

    _, _, dashboard = await asgi_request("/")
    assert 'href="zones"' in dashboard.decode()


@pytest.mark.asyncio
async def test_zones_api_reports_the_stored_configuration():
    web.zone_store.add(_exclusion())
    status, _, body = await asgi_request("/api/zones")
    assert status == 200
    payload = json.loads(body)
    assert [zone["name"] for zone in payload["zones"]] == ["Greenhouse path"]
    assert payload["zones"][0]["kind"] == "exclusion"


@pytest.mark.asyncio
async def test_create_update_and_delete_a_zone_through_the_form_routes():
    status, headers, _ = await post_form(
        "/zones",
        {
            "name": "Bed 1",
            "kind": "boundary",
            "shape": "rectangle",
            "allow_weeds": "true",
            "allow_plant_centers": "true",
            "allow_plant_radius": "true",
            "min_x": "1000",
            "min_y": "0",
            "max_x": "0",
            "max_y": "800",
        },
    )
    assert status == 303
    assert headers[b"location"] == b"zones"
    zones = web.zone_store.zones()
    assert len(zones) == 1
    zone = zones[0]
    # Corners are normalized, so either drag direction is accepted.
    assert (zone.min_x, zone.max_x, zone.min_y, zone.max_y) == (0, 1_000, 0, 800)
    assert zone.allow_weeds and zone.allow_plant_centers and zone.allow_plant_radius

    # Only the ticked boxes are submitted, so an unticked one clears the flag.
    status, _, _ = await post_form(
        f"/zones/{zone.zone_id}/update",
        {"allow_plant_radius": "true", "enabled": "true"},
    )
    assert status == 303
    updated = web.zone_store.zones()[0]
    assert updated.allow_weeds is False
    assert updated.allow_plant_centers is False
    assert updated.allow_plant_radius is True
    assert updated.enabled is True

    status, _, _ = await post_form(f"/zones/{zone.zone_id}/delete", {})
    assert status == 303
    assert web.zone_store.zones() == []

    status, _, _ = await post_form(f"/zones/{zone.zone_id}/delete", {})
    assert status == 404
    status, _, _ = await post_form(f"/zones/{zone.zone_id}/update", {})
    assert status == 404


@pytest.mark.asyncio
async def test_polygon_points_are_parsed_and_bad_input_is_refused():
    status, _, _ = await post_form(
        "/zones",
        {
            "name": "Triangle",
            "kind": "exclusion",
            "shape": "polygon",
            "points": "0, 0\n1200, 0\n1200, 800",
        },
    )
    assert status == 303
    assert web.zone_store.zones()[0].points == [(0, 0), (1200, 0), (1200, 800)]

    base = {"name": "Bad", "kind": "exclusion", "shape": "polygon"}
    status, _, _ = await post_form("/zones", {**base, "points": "0, 0, 5"})
    assert status == 422
    status, _, _ = await post_form("/zones", {**base, "points": "0, 0"})
    assert status == 422
    status, _, _ = await post_form("/zones", {**base, "points": "left, right"})
    assert status == 422
    status, _, _ = await post_form(
        "/zones", {"name": "Bad kind", "kind": "fence", "shape": "rectangle"}
    )
    assert status == 400
    # A zero-area rectangle is refused instead of silently matching nothing.
    status, _, _ = await post_form(
        "/zones",
        {"name": "Flat", "kind": "boundary", "shape": "rectangle", "max_x": "0", "max_y": "0"},
    )
    assert status == 422
    assert len(web.zone_store.zones()) == 1


@pytest.mark.asyncio
async def test_weed_approval_is_refused_inside_an_exclusion_zone(monkeypatch):
    web.zone_store.add(_exclusion())
    created = []

    async def create_weed(request):
        created.append(request)
        return {"status": "applied", "message": "weed created"}

    monkeypatch.setattr(web.client, "create_weed", create_weed)

    blocked = _weed_detection(500, 500)
    response = await web.approve_weed(blocked)
    assert response.status_code == 409
    assert b"Greenhouse path" in response.body
    assert created == []
    # The recommendation is kept so the zones can be corrected instead.
    assert web.database.weed_detection(str(blocked))["status"] == "recommended"

    allowed = _weed_detection(2_000, 2_000)
    response = await web.approve_weed(allowed)
    assert json.loads(response.body)["status"] == "applied"
    assert len(created) == 1


@pytest.mark.asyncio
async def test_radius_approval_is_refused_when_the_disc_reaches_an_exclusion_zone(monkeypatch):
    web.zone_store.add(_exclusion(name="Fence line", min_x=600, max_x=2_000))
    applied = []

    async def apply_radius(request):
        applied.append(request)
        return {"status": "applied", "message": "radius updated"}

    monkeypatch.setattr(web.client, "apply_radius", apply_radius)
    measurement = _review_measurement(
        recorded_center_x=500,
        recorded_center_y=500,
        recommended_protection_radius_mm=200,
    )
    web.database.save_measurements([measurement])

    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 409
    result = json.loads(body)
    assert result["status"] == "conflict"
    assert "Fence line" in result["message"]
    assert applied == []

    # The same radius applies once the plant is clear of the zone.
    far = _review_measurement(
        recorded_center_x=100,
        recorded_center_y=100,
        recommended_protection_radius_mm=200,
    )
    web.database.save_measurements([far])
    status, _, body = await asgi_request(
        f"/recommendations/{far.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 200
    assert json.loads(body)["status"] == "applied"


@pytest.mark.asyncio
async def test_center_move_is_refused_outside_a_boundary(monkeypatch):
    web.zone_store.add(
        Zone(
            name="Raised bed",
            kind=ZoneKind.BOUNDARY,
            shape=ZoneShape.CIRCLE,
            center_x=0,
            center_y=0,
            radius_mm=300,
            allow_plant_centers=True,
        )
    )
    moved = []

    async def apply_plant_center(request):
        moved.append(request)
        return {"status": "applied", "message": "centre moved"}

    monkeypatch.setattr(web.client, "apply_plant_center", apply_plant_center)
    measurement = _review_measurement(
        center_misaligned=True,
        recommended_center_px=(900.0, 900.0),
    )
    web.database.save_measurements([measurement])

    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/move-center",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 409
    assert "boundary" in json.loads(body)["message"]
    assert moved == []


@pytest.mark.asyncio
async def test_measurements_without_a_recorded_position_are_not_zone_blocked(monkeypatch):
    """Older rows predate stored plant positions and keep working as before."""
    web.zone_store.add(_exclusion())

    async def apply_radius(_request):
        return {"status": "applied", "message": "radius updated"}

    monkeypatch.setattr(web.client, "apply_radius", apply_radius)
    measurement = _review_measurement()
    web.database.save_measurements([measurement])

    status, _, body = await asgi_request(
        f"/recommendations/{measurement.measurement_id}/approve",
        method="POST",
        headers=[(b"accept", b"application/json")],
    )
    assert status == 200
    assert json.loads(body)["status"] == "applied"
