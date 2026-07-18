from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, TypeVar

import httpx
import websockets
from pydantic import BaseModel, ValidationError

from .models import (
    ApplyRadiusRequest,
    BotList,
    Inventory,
    InventoryRequest,
    UpsertCurveRequest,
    VisionImage,
    VisionImageRequest,
    VisionRequestEvent,
    VisionStatus,
)

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# JPEG Start-Of-Frame markers carry the image dimensions. Everything except
# these and the standalone markers below has a two-byte length we can skip.
_SOF_MARKERS = frozenset(
    range(0xC0, 0xD0)  # SOF0..SOF15
) - {0xC4, 0xC8, 0xCC}  # DHT, JPG, DAC are not frame headers
_STANDALONE = frozenset({0xD8, 0xD9, *range(0xD0, 0xD8), 0x01})


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) from a JPEG's SOF marker without decoding pixels.

    Deliberately dependency-free so image validation never pulls OpenCV into
    the Home Assistant client. Returns ``None`` for malformed data.
    """
    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    index = 2
    length = len(data)
    while index + 1 < length:
        if data[index] != 0xFF:
            return None
        # Skip any fill bytes (0xFF padding) before the marker code.
        while index < length and data[index] == 0xFF:
            index += 1
        if index >= length:
            return None
        marker = data[index]
        index += 1
        if marker in _STANDALONE:
            continue
        if index + 1 >= length:
            return None
        segment_length = (data[index] << 8) | data[index + 1]
        if segment_length < 2:
            return None
        if marker in _SOF_MARKERS:
            if index + 6 >= length:
                return None
            height = (data[index + 3] << 8) | data[index + 4]
            width = (data[index + 5] << 8) | data[index + 6]
            if width <= 0 or height <= 0:
                return None
            return width, height
        index += segment_length
    return None


class HomeAssistantError(RuntimeError):
    pass


class HomeAssistantConnectionError(HomeAssistantError):
    """The event WebSocket could not be established or remained open."""


class HomeAssistantAuthenticationError(HomeAssistantError):
    """Home Assistant rejected WebSocket authentication."""


class HomeAssistantSubscriptionError(HomeAssistantError):
    """Home Assistant rejected the vision event subscription."""


class StaleRadiusError(HomeAssistantError):
    pass


def _snippet(text: str, limit: int = 300) -> str:
    """Truncate response text for logging so a large payload (e.g. a base64
    image on an unexpected-shape response) never floods the log."""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[:limit]}… ({len(text)} bytes total)"


class HomeAssistantClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = "http://supervisor/core/api",
        ws_url: str = "ws://supervisor/core/websocket",
        timeout: float = 30,
    ):
        self._token = token if token is not None else os.getenv("SUPERVISOR_TOKEN", "")
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self.timeout = timeout
        if not self._token:
            LOGGER.warning(
                "No Home Assistant token configured (SUPERVISOR_TOKEN is unset/empty); "
                "every request to %s will be rejected with 401",
                self.base_url,
            )
        else:
            LOGGER.info(
                "Home Assistant client configured: base_url=%s ws_url=%s token_length=%d",
                self.base_url,
                self.ws_url,
                len(self._token),
            )
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def _service(
        self,
        action: str,
        payload: BaseModel | dict[str, Any],
        model: type[T] | None = None,
        *,
        return_response: bool = True,
    ) -> T | dict[str, Any]:
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        url = f"{self.base_url}/services/farmbot/{action}"
        if return_response:
            url += "?return_response"
        last_error: Exception | None = None
        for attempt in range(3):
            LOGGER.debug("Calling Home Assistant service %s (attempt %d/3)", action, attempt + 1)
            try:
                response = await self._http.post(url, json=body)
                if response.status_code in {409, 412}:
                    LOGGER.info(
                        "Home Assistant service %s reported a stale radius (HTTP %d)",
                        action,
                        response.status_code,
                    )
                    raise StaleRadiusError("FarmBot radius changed; inventory refresh required")
                if response.status_code in {400, 401, 403, 422}:
                    LOGGER.error(
                        "Home Assistant service %s rejected the request with a non-retryable "
                        "HTTP %d: %s",
                        action,
                        response.status_code,
                        _snippet(response.text),
                    )
                    raise HomeAssistantError(
                        f"non-retryable Home Assistant response {response.status_code}"
                    )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and data:
                    data = data[0].get("service_response", data[0])
                elif isinstance(data, dict) and "service_response" in data:
                    data = data["service_response"]
                return model.model_validate(data) if model else data
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                LOGGER.warning(
                    "Home Assistant service %s failed on attempt %d/3 (%s%s); %s",
                    action,
                    attempt + 1,
                    type(exc).__name__,
                    f" HTTP {status}" if status is not None else "",
                    "retrying" if attempt < 2 else "giving up",
                )
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
            except (json.JSONDecodeError, ValidationError) as exc:
                LOGGER.error(
                    "Home Assistant service %s returned a response that does not match the "
                    "expected contract (%s): %s; raw response: %s",
                    action,
                    type(exc).__name__,
                    exc if isinstance(exc, json.JSONDecodeError) else _validation_summary(exc),
                    _snippet(response.text),
                )
                raise HomeAssistantError("malformed FarmBot integration response") from exc
        LOGGER.error(
            "Home Assistant service %s failed after 3 attempts; last error: %s: %s",
            action,
            type(last_error).__name__ if last_error else "unknown",
            last_error,
        )
        raise HomeAssistantError("Home Assistant temporarily unavailable") from last_error

    async def list_bots(self) -> BotList:
        return await self._service("list_vision_bots", {}, BotList)  # type: ignore[return-value]

    async def inventory(self, request: InventoryRequest) -> Inventory:
        return await self._service("get_vision_inventory", request, Inventory)  # type: ignore[return-value]

    async def image(self, request: VisionImageRequest, max_payload_bytes: int) -> VisionImage:
        result = await self._service("get_vision_image", request, VisionImage)
        if not isinstance(result, VisionImage):
            raise HomeAssistantError("malformed image response")
        # Do not accept a frame larger than what was requested or the ceiling.
        if result.width > request.max_width or result.height > request.max_height:
            raise HomeAssistantError("image response exceeds requested dimensions")
        if len(result.image_base64) > (max_payload_bytes * 4 // 3 + 8):
            raise HomeAssistantError("image response exceeds configured limit")
        try:
            decoded = base64.b64decode(result.image_base64, validate=True)
        except ValueError as exc:
            raise HomeAssistantError("image response contains malformed base64") from exc
        if len(decoded) > max_payload_bytes:
            raise HomeAssistantError("decoded image exceeds configured limit")
        if not decoded.startswith(b"\xff\xd8"):
            raise HomeAssistantError("image response is not JPEG data")
        if hashlib.sha256(decoded).hexdigest().lower() != result.sha256.lower():
            raise HomeAssistantError("image checksum mismatch")
        dimensions = _jpeg_dimensions(decoded)
        if dimensions is None:
            raise HomeAssistantError("image response contains malformed JPEG data")
        if dimensions != (result.width, result.height):
            raise HomeAssistantError("decoded image dimensions do not match the response")
        return result

    async def apply_radius(self, request: ApplyRadiusRequest) -> dict[str, Any]:
        return await self._service("apply_vision_radius", request)  # type: ignore[return-value]

    async def upsert_curve(self, request: UpsertCurveRequest) -> dict[str, Any]:
        return await self._service("upsert_vision_spread_curve", request)  # type: ignore[return-value]

    async def report_status(self, status: VisionStatus) -> None:
        await self._service("report_vision_status", status, return_response=False)

    async def vision_events(self) -> AsyncIterator[VisionRequestEvent]:
        while True:
            LOGGER.info("Vision event listener: connecting to %s", self.ws_url)
            try:
                async with websockets.connect(self.ws_url, open_timeout=10) as socket:
                    LOGGER.debug(
                        "Vision event listener: WebSocket transport open, awaiting handshake"
                    )
                    try:
                        auth_required = json.loads(await socket.recv())
                    except json.JSONDecodeError as exc:
                        raise HomeAssistantConnectionError("malformed WebSocket handshake") from exc
                    if (
                        not isinstance(auth_required, dict)
                        or auth_required.get("type") != "auth_required"
                    ):
                        raise HomeAssistantConnectionError(
                            f"unexpected WebSocket handshake: {_snippet(str(auth_required))}"
                        )
                    await socket.send(json.dumps({"type": "auth", "access_token": self._token}))
                    try:
                        auth = json.loads(await socket.recv())
                    except json.JSONDecodeError as exc:
                        raise HomeAssistantConnectionError(
                            "malformed WebSocket authentication response"
                        ) from exc
                    if not isinstance(auth, dict) or auth.get("type") != "auth_ok":
                        raise HomeAssistantAuthenticationError(
                            f"Home Assistant WebSocket authentication failed: {_snippet(str(auth))}"
                        )
                    LOGGER.info("Vision event listener: WebSocket authenticated")
                    await socket.send(
                        json.dumps(
                            {
                                "id": 1,
                                "type": "subscribe_events",
                                "event_type": "farmbot_vision_request",
                            }
                        )
                    )
                    async for raw in socket:
                        try:
                            message = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            LOGGER.warning(
                                "Vision event skipped: malformed JSON (%s)", _snippet(str(raw))
                            )
                            continue
                        if not isinstance(message, dict):
                            LOGGER.warning(
                                "Vision event skipped: malformed message (%s)",
                                _snippet(str(message)),
                            )
                            continue
                        if message.get("type") == "result" and message.get("id") == 1:
                            if not message.get("success"):
                                raise HomeAssistantSubscriptionError(
                                    "vision event subscription rejected: "
                                    f"{_snippet(str(message.get('error')))}"
                                )
                            LOGGER.info(
                                "Vision event listener: subscribed to farmbot_vision_request "
                                "and connected"
                            )
                            continue
                        if message.get("type") != "event":
                            continue
                        event_data = message.get("event", {})
                        data = event_data.get("data") if isinstance(event_data, dict) else None
                        try:
                            event = VisionRequestEvent.model_validate(data)
                        except ValidationError as exc:
                            LOGGER.warning(
                                "Vision event skipped: validation failed (%s)",
                                _validation_summary(exc),
                            )
                            continue
                        LOGGER.debug(
                            "FarmBot Vision request accepted: config_entry_id=%s mode=%s "
                            "plant_count=%d device_id_supplied=%s",
                            event.config_entry_id,
                            event.mode,
                            len(event.plant_ids),
                            event.device_id is not None,
                        )
                        yield event
                    raise HomeAssistantConnectionError("WebSocket connection closed by peer")
            except HomeAssistantAuthenticationError as exc:
                LOGGER.warning("Vision event listener: authentication failure: %s", exc)
                await asyncio.sleep(15)
            except HomeAssistantSubscriptionError as exc:
                LOGGER.warning("Vision event listener: subscription failure: %s", exc)
                await asyncio.sleep(15)
            except (OSError, websockets.WebSocketException) as exc:
                LOGGER.warning(
                    "Vision event listener: WebSocket connection failure (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(15)
            except HomeAssistantConnectionError as exc:
                LOGGER.warning("Vision event listener: connection failure: %s", exc)
                await asyncio.sleep(15)
            except HomeAssistantError as exc:
                LOGGER.warning(
                    "Vision event listener: unexpected failure (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(15)


def _validation_summary(exc: ValidationError) -> str:
    """Return field/type-only details suitable for a safe warning log."""

    details = []
    for error in exc.errors()[:4]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "event"
        details.append(f"{location}:{error.get('type', 'validation_error')}")
    return ", ".join(details) or "validation_error"
