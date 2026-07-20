from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from farmbot_vision.home_assistant import (
    HomeAssistantClient,
    HomeAssistantError,
    StaleRadiusError,
)
from farmbot_vision.models import ApplyRadiusRequest, InventoryRequest, VisionRequestEvent


@pytest.mark.asyncio
async def test_unexpected_server_error_is_retried_then_raised(monkeypatch: pytest.MonkeyPatch):
    """A bare 500 from Home Assistant must not crash the caller uncaught.

    ``response.raise_for_status()`` raises ``httpx.HTTPStatusError`` for any
    status outside the special-cased sets; it must be retried like a network
    error and surfaced as ``HomeAssistantError`` so route handlers can map it
    to a clean response instead of an unhandled ASGI exception.
    """

    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"error": "boom"})

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = HomeAssistantClient(token="test", base_url="http://test")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(HomeAssistantError):
        await client.inventory(InventoryRequest(config_entry_id="entry"))
    assert calls == 3
    assert sleeps == [1, 2]
    await client.close()


@pytest.mark.asyncio
async def test_inventory_accepts_flat_image_coordinates():
    """A companion integration build observed in production sends image
    coordinates flat on the image object (``x``/``y``/``z``) with no
    ``processed`` flag, instead of the documented nested ``meta`` object.
    The inventory call must still succeed rather than rejecting every image.
    """

    async def handler(_request):
        return httpx.Response(
            200,
            json={
                "changed_states": [],
                "service_response": {
                    "device_id": "device_28660",
                    "generated_at": "2026-07-18T14:37:40.984259+00:00",
                    "plants": [],
                    "images": [
                        {
                            "id": 1,
                            "created_at": "2026-07-18T14:00:00+00:00",
                            "x": 2600,
                            "y": 230,
                            "z": 0,
                        }
                    ],
                    "curves": [],
                    "camera_calibration": {"available": False},
                },
            },
        )

    client = HomeAssistantClient(token="test", base_url="http://test")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    inventory = await client.inventory(InventoryRequest(config_entry_id="entry"))
    assert inventory.images[0].processed is True
    assert inventory.images[0].meta.x == 2600
    assert inventory.images[0].meta.y == 230
    await client.close()


@pytest.mark.asyncio
async def test_inventory_accepts_unrecognized_camera_calibration_basis():
    """A companion integration build observed in production sends a
    ``camera_calibration.basis`` value outside the two documented literals.

    ``basis`` here is informational only -- ``calibration.py`` never reads
    ``CameraCalibration.basis`` (unlike ``ProcessedCalibration.basis``, which
    is load-bearing) -- so an unrecognized value must degrade to ``None``
    rather than failing the whole inventory response.
    """

    async def handler(_request):
        return httpx.Response(
            200,
            json={
                "changed_states": [],
                "service_response": {
                    "device_id": "device_28660",
                    "generated_at": "2026-07-18T15:10:43.264781+00:00",
                    "plants": [],
                    "images": [],
                    "curves": [],
                    "camera_calibration": {
                        "available": True,
                        "pixels_per_mm_x": 3.1,
                        "pixels_per_mm_y": 3.1,
                        "basis": "native",
                    },
                },
            },
        )

    client = HomeAssistantClient(token="test", base_url="http://test")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    inventory = await client.inventory(InventoryRequest(config_entry_id="entry"))
    assert inventory.camera_calibration.basis is None
    assert inventory.camera_calibration.pixels_per_mm_x == 3.1
    await client.close()


@pytest.mark.asyncio
async def test_stale_current_radius_conflict():
    async def handler(_request):
        return httpx.Response(409, json={"error": "stale radius"})

    client = HomeAssistantClient(token="test", base_url="http://test")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(StaleRadiusError):
        await client.apply_radius(
            ApplyRadiusRequest(
                config_entry_id="entry",
                plant_id=1,
                measurement_id="b2fbfe4f-2a7b-48b7-8e53-f2cd3fd92aba",
                expected_current_radius_mm=100,
                recommended_radius_mm=120,
                confidence=0.95,
                apply=True,
            )
        )
    await client.close()


@pytest.mark.parametrize("mode", ["observe", "recommend", "auto_radius"])
def test_vision_request_event_contract(mode):
    event = VisionRequestEvent.model_validate(
        {
            "config_entry_id": "entry",
            "device_id": "device",
            "plant_ids": [],
            "mode": mode,
        }
    )
    assert event.device_id == "device"
    assert event.plant_ids == []

    without_device = VisionRequestEvent.model_validate(
        {"config_entry_id": "entry", "plant_ids": [1, 2], "mode": mode}
    )
    assert without_device.device_id is None


def test_automatic_photo_event_uses_app_mode_and_targets_one_image():
    event = VisionRequestEvent.model_validate(
        {
            "config_entry_id": "entry",
            "device_id": "device_42",
            "plant_ids": [],
            "image_id": 3043473,
        }
    )
    assert event.mode is None
    assert event.image_id == 3043473


@pytest.mark.parametrize(
    "payload",
    [
        {"config_entry_id": "entry", "plant_ids": [0], "mode": "recommend"},
        {"config_entry_id": "entry", "plant_ids": [-1], "mode": "recommend"},
        {"config_entry_id": "entry", "plant_ids": [True], "mode": "recommend"},
        {"config_entry_id": "entry", "plant_ids": [], "mode": "invalid"},
        {"config_entry_id": "entry", "image_id": 0},
        {"config_entry_id": "entry", "image_id": True},
        {"config_entry_id": "entry", "plant_ids": [], "mode": "recommend", "unexpected": 1},
    ],
)
def test_vision_request_event_rejects_invalid_payload(payload):
    with pytest.raises(ValidationError):
        VisionRequestEvent.model_validate(payload)


class FakeSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def recv(self):
        return next(self.messages)

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def websocket_messages(*events):
    return [
        json.dumps({"type": "auth_required"}),
        json.dumps({"type": "auth_ok"}),
        *events,
    ]


@pytest.mark.asyncio
async def test_malformed_and_invalid_events_are_skipped_then_valid_event_is_yielded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    sensitive = "supervisor-token image-data /api/hassio_ingress/private-session"
    socket = FakeSocket(
        websocket_messages(
            "not-json",
            "still-not-json",
            "also-not-json",
            json.dumps(
                {
                    "type": "event",
                    "event": {
                        "data": {
                            "config_entry_id": "entry",
                            "device_id": sensitive,
                            "plant_ids": [0],
                            "mode": "recommend",
                            "secret": sensitive,
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "type": "event",
                    "event": {
                        "data": {
                            "config_entry_id": "entry",
                            "device_id": "device",
                            "plant_ids": [],
                            "mode": "recommend",
                        }
                    },
                }
            ),
        )
    )
    monkeypatch.setattr(
        "farmbot_vision.home_assistant.websockets.connect", lambda *args, **kwargs: socket
    )
    client = HomeAssistantClient(token="supervisor-token")
    event = await anext(client.vision_events())
    assert event.config_entry_id == "entry"
    assert event.device_id == "device"
    assert "supervisor-token" not in caplog.text
    assert "image-data" not in caplog.text
    assert "/api/hassio_ingress/private-session" not in caplog.text
    await client.close()


@pytest.mark.asyncio
async def test_authentication_failure_reconnects(monkeypatch: pytest.MonkeyPatch):
    failed = FakeSocket(
        [json.dumps({"type": "auth_required"}), json.dumps({"type": "auth_invalid"})]
    )
    succeeding = FakeSocket(
        websocket_messages(
            json.dumps(
                {
                    "type": "event",
                    "event": {"data": {"config_entry_id": "entry", "mode": "observe"}},
                }
            )
        )
    )
    sockets = iter([failed, succeeding])
    sleeps = []
    monkeypatch.setattr(
        "farmbot_vision.home_assistant.websockets.connect", lambda *args, **kwargs: next(sockets)
    )

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = HomeAssistantClient(token="test")
    event = await anext(client.vision_events())
    assert event.mode == "observe"
    assert sleeps == [15]
    await client.close()


@pytest.mark.asyncio
async def test_socket_closure_reconnects(monkeypatch: pytest.MonkeyPatch):
    closed = FakeSocket(
        websocket_messages(json.dumps({"type": "result", "id": 1, "success": True}))
    )
    succeeding = FakeSocket(
        websocket_messages(
            json.dumps(
                {
                    "type": "event",
                    "event": {"data": {"config_entry_id": "entry", "mode": "auto_radius"}},
                }
            )
        )
    )
    sockets = iter([closed, succeeding])
    sleeps = []
    monkeypatch.setattr(
        "farmbot_vision.home_assistant.websockets.connect", lambda *args, **kwargs: next(sockets)
    )

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = HomeAssistantClient(token="test")
    event = await anext(client.vision_events())
    assert event.mode == "auto_radius"
    assert sleeps == [15]
    await client.close()


@pytest.mark.asyncio
async def test_subscription_failure_reconnects(monkeypatch: pytest.MonkeyPatch):
    failed = FakeSocket(
        websocket_messages(json.dumps({"type": "result", "id": 1, "success": False}))
    )
    succeeding = FakeSocket(
        websocket_messages(
            json.dumps(
                {
                    "type": "event",
                    "event": {"data": {"config_entry_id": "entry", "mode": "recommend"}},
                }
            )
        )
    )
    sockets = iter([failed, succeeding])
    sleeps = []
    monkeypatch.setattr(
        "farmbot_vision.home_assistant.websockets.connect", lambda *args, **kwargs: next(sockets)
    )

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = HomeAssistantClient(token="test")
    event = await anext(client.vision_events())
    assert event.mode == "recommend"
    assert sleeps == [15]
    await client.close()


@pytest.mark.asyncio
async def test_websocket_connect_sends_supervisor_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
):
    """The Supervisor /core/websocket proxy authorizes the add-on from the
    supervisor token on the HTTP upgrade, like the REST proxy. Without the
    Bearer header the upgrade is rejected (websockets InvalidStatus) before the
    in-band auth exchange, so the listener never connects. Guard that the token
    is presented on the handshake."""
    socket = FakeSocket(
        websocket_messages(
            json.dumps(
                {
                    "type": "event",
                    "event": {"data": {"config_entry_id": "entry", "mode": "recommend"}},
                }
            )
        )
    )
    captured: dict = {}

    def fake_connect(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return socket

    monkeypatch.setattr("farmbot_vision.home_assistant.websockets.connect", fake_connect)
    client = HomeAssistantClient(token="supervisor-token")
    event = await anext(client.vision_events())
    assert event.config_entry_id == "entry"
    assert captured["kwargs"]["additional_headers"] == {"Authorization": "Bearer supervisor-token"}
    await client.close()
