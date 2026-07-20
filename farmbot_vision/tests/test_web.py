from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import farmbot_vision.web as web
from farmbot_vision.models import BotList, VisionRequestEvent


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
async def test_event_listener_targets_new_image_and_uses_configured_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    async def events():
        yield VisionRequestEvent(
            config_entry_id="entry-1", device_id="device_42", image_id=99
        )

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
