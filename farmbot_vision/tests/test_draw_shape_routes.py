"""Routes behind the experimental Draw shape tab.

This is the only path in the app that moves the bot outside FarmBot OS's
motion planning, so what these pin is mostly refusal: no bot selected, an
integration too old to have the capability, an empty program. The one positive
behaviour that matters as much is that the text in the editor is what gets
sent -- the point of showing an editable program is lost if the app quietly
regenerates it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from test_web import asgi_request

import farmbot_vision.web as web
from farmbot_vision.models import (
    Bot,
    BotList,
    GcodeRunStatus,
    SoilMotionState,
    SoilPointInventory,
)

ENTRY_ID = "entry-1"


def _bot(capabilities=("experimental_raw_gcode",)):
    return BotList(
        bots=[
            Bot(
                config_entry_id=ENTRY_ID,
                device_id="42",
                name="FarmBot",
                integration_version="2.6.0",
                capabilities=list(capabilities),
            )
        ]
    )


def _inventory():
    return SoilPointInventory(
        device_id="42",
        generated_at=datetime.now(UTC),
        points=[],
        motion=SoilMotionState(
            connected=True,
            busy=False,
            locked=False,
            position={"x": 100.0, "y": 100.0, "z": 0.0},
            z_direction=-1,
            axis_bounds={"x": (0.0, 2000.0), "y": (0.0, 1000.0), "z": (-500.0, 0.0)},
        ),
    )


class FakeClient:
    """Stands in for HomeAssistantClient, recording what the app sent."""

    def __init__(self, *, capabilities=("experimental_raw_gcode",), status=None):
        self._capabilities = capabilities
        self._status = status or GcodeRunStatus(
            status="queued",
            message="Raw G-code run queued",
            run_id="11111111-2222-3333-4444-555555555555",
        )
        self.started: list = []
        self.status_requests: list = []

    async def list_bots(self):
        return _bot(self._capabilities)

    async def soil_points(self, _entry_id):
        return _inventory()

    async def start_gcode_run(self, request):
        self.started.append(request)
        return self._status

    async def gcode_run_status(self, entry_id, run_id):
        self.status_requests.append((entry_id, run_id))
        return GcodeRunStatus(
            status="running", message="Executed chunk 1 of 2", chunks_sent=1, chunks_total=2
        )


@pytest.fixture(autouse=True)
def selected_bot(monkeypatch):
    monkeypatch.setattr(web.settings, "selected_config_entry_id", ENTRY_ID)
    web._last_gcode_run.clear()
    yield
    web._last_gcode_run.clear()


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(web, "client", client)
    return client


async def _post_json(path: str, payload: dict):
    status, _headers, body = await asgi_request(
        path,
        method="POST",
        raw_body=json.dumps(payload).encode(),
        content_type="application/json",
    )
    return status, json.loads(body or b"{}")


PLAN = {
    "shape": "circle",
    "center_x": 500,
    "center_y": 400,
    "circumradius_mm": 100,
    "feed_mm_per_min": 400,
}


@pytest.mark.anyio
async def test_the_plan_route_returns_a_program_and_its_path(fake_client):
    status, data = await _post_json("/api/draw-shape/plan", PLAN)

    assert status == 200
    assert data["lines"][0].startswith(";")
    assert any(line.startswith("G00 X") for line in data["lines"])
    assert data["points"][0] == data["points"][-1]  # a closed path
    assert data["center"] == [500.0, 400.0]
    assert data["segments"] >= 8
    assert "mm of path" in data["summary"]


@pytest.mark.anyio
async def test_planning_never_touches_the_bot(fake_client):
    """Previewing a shape must not queue anything."""
    await _post_json("/api/draw-shape/plan", PLAN)

    assert fake_client.started == []


@pytest.mark.anyio
async def test_an_auto_segment_field_posts_blank_and_is_not_an_error(fake_client):
    status, data = await _post_json("/api/draw-shape/plan", {**PLAN, "segments": "", "sides": ""})

    assert status == 200
    assert data["segments"] >= 8


@pytest.mark.anyio
async def test_an_impossible_shape_is_a_clean_400(fake_client):
    status, data = await _post_json("/api/draw-shape/plan", {**PLAN, "circumradius_mm": 1})

    assert status == 400
    assert "Circumradius" in data["detail"]


@pytest.mark.anyio
async def test_the_editors_text_is_what_gets_sent(fake_client):
    """Hand edits are the point of the editor; the app must not regenerate."""
    edited = "G21\nG90\nG00 X111 Y222\nG00 X333 Y444\n"

    status, data = await _post_json(
        "/api/draw-shape/run",
        {"lines": edited, "feed_mm_per_min": 250, "return_to_start": False},
    )

    assert status == 200
    assert data["status"] == "queued"
    request = fake_client.started[0]
    assert request.lines == ["G21", "G90", "G00 X111 Y222", "G00 X333 Y444"]
    assert request.feed_mm_per_min == 250
    assert request.return_to_start is False
    assert request.dry_run is False
    # The integration requires this and it is never inferred from a default.
    assert request.acknowledge_experimental is True


@pytest.mark.anyio
async def test_a_dry_run_is_forwarded_as_a_dry_run(fake_client):
    await _post_json("/api/draw-shape/run", {"lines": "G90\nG00 X111 Y222", "dry_run": True})

    assert fake_client.started[0].dry_run is True


@pytest.mark.anyio
async def test_an_empty_program_is_refused_before_reaching_the_bot(fake_client):
    status, data = await _post_json("/api/draw-shape/run", {"lines": "\n  \n\n"})

    assert status == 400
    assert "empty" in data["detail"]
    assert fake_client.started == []


@pytest.mark.anyio
async def test_an_integration_without_the_capability_is_refused(monkeypatch):
    client = FakeClient(capabilities=("photo_grid_repair",))
    monkeypatch.setattr(web, "client", client)

    status, data = await _post_json("/api/draw-shape/run", {"lines": "G90\nG00 X10 Y10"})

    assert status == 409
    assert "V2.6.0" in data["detail"]
    assert client.started == []


@pytest.mark.anyio
async def test_no_selected_bot_is_refused(monkeypatch, fake_client):
    monkeypatch.setattr(web.settings, "selected_config_entry_id", None)

    status, _data = await _post_json("/api/draw-shape/run", {"lines": "G90\nG00 X10 Y10"})

    assert status == 409
    assert fake_client.started == []


@pytest.mark.anyio
async def test_status_reports_idle_until_a_run_has_been_started(fake_client):
    status, _headers, body = await asgi_request("/api/draw-shape/status")

    assert status == 200
    assert json.loads(body)["status"] == "idle"
    assert fake_client.status_requests == []


@pytest.mark.anyio
async def test_status_follows_the_run_that_was_started(fake_client):
    await _post_json("/api/draw-shape/run", {"lines": "G90\nG00 X10 Y10"})

    _status, _headers, body = await asgi_request("/api/draw-shape/status")
    data = json.loads(body)

    assert data["status"] == "running"
    assert data["chunks_sent"] == 1
    assert fake_client.status_requests == [(ENTRY_ID, "11111111-2222-3333-4444-555555555555")]


@pytest.mark.anyio
async def test_a_dry_run_does_not_become_the_tracked_run(fake_client):
    """A validation has no run to poll; it must not overwrite a real one."""
    fake_client._status = GcodeRunStatus(status="validated", message="Program is valid")

    await _post_json("/api/draw-shape/run", {"lines": "G90\nG00 X10 Y10", "dry_run": True})
    _status, _headers, body = await asgi_request("/api/draw-shape/status")

    assert json.loads(body)["status"] == "idle"


@pytest.mark.anyio
async def test_the_page_warns_that_farmbot_os_safety_is_bypassed(fake_client):
    status, _headers, body = await asgi_request("/draw-shape")
    page = body.decode()

    assert status == 200
    assert "experimental" in page.lower()
    assert "outside FarmBot OS" in page
    assert "emergency stop" in page
    # The firmware caveat belongs on the page, not just in the source.
    assert "G01" in page and "straight line" in page
