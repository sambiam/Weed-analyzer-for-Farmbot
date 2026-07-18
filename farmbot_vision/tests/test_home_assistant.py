from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from farmbot_vision.home_assistant import HomeAssistantClient, StaleRadiusError
from farmbot_vision.models import ApplyRadiusRequest, VisionRequestEvent
from pydantic import ValidationError


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


@pytest.mark.parametrize(
    "payload",
    [
        {"config_entry_id": "entry", "plant_ids": [0], "mode": "recommend"},
        {"config_entry_id": "entry", "plant_ids": [-1], "mode": "recommend"},
        {"config_entry_id": "entry", "plant_ids": [True], "mode": "recommend"},
        {"config_entry_id": "entry", "plant_ids": [], "mode": "invalid"},
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
