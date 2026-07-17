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


class HomeAssistantError(RuntimeError):
    pass


class StaleRadiusError(HomeAssistantError):
    pass


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
            try:
                response = await self._http.post(url, json=body)
                if response.status_code in {409, 412}:
                    raise StaleRadiusError("FarmBot radius changed; inventory refresh required")
                if response.status_code in {400, 401, 403, 422}:
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
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise HomeAssistantError("malformed FarmBot integration response") from exc
        raise HomeAssistantError("Home Assistant temporarily unavailable") from last_error

    async def list_bots(self) -> BotList:
        return await self._service("list_vision_bots", {}, BotList)  # type: ignore[return-value]

    async def inventory(self, request: InventoryRequest) -> Inventory:
        return await self._service("get_vision_inventory", request, Inventory)  # type: ignore[return-value]

    async def image(self, request: VisionImageRequest, max_payload_bytes: int) -> VisionImage:
        result = await self._service("get_vision_image", request, VisionImage)
        if not isinstance(result, VisionImage):
            raise HomeAssistantError("malformed image response")
        if len(result.image_base64) > (max_payload_bytes * 4 // 3 + 8):
            raise HomeAssistantError("image response exceeds configured limit")
        try:
            decoded = base64.b64decode(result.image_base64, validate=True)
        except ValueError as exc:
            raise HomeAssistantError("image response contains malformed base64") from exc
        if len(decoded) > max_payload_bytes:
            raise HomeAssistantError("decoded image exceeds configured limit")
        if hashlib.sha256(decoded).hexdigest().lower() != result.sha256.lower():
            raise HomeAssistantError("image checksum mismatch")
        return result

    async def apply_radius(self, request: ApplyRadiusRequest) -> dict[str, Any]:
        return await self._service("apply_vision_radius", request)  # type: ignore[return-value]

    async def upsert_curve(self, request: UpsertCurveRequest) -> dict[str, Any]:
        return await self._service("upsert_vision_spread_curve", request)  # type: ignore[return-value]

    async def report_status(self, status: VisionStatus) -> None:
        await self._service("report_vision_status", status, return_response=False)

    async def vision_events(self) -> AsyncIterator[VisionRequestEvent]:
        while True:
            try:
                async with websockets.connect(self.ws_url, open_timeout=10) as socket:
                    auth_required = json.loads(await socket.recv())
                    if auth_required.get("type") != "auth_required":
                        raise HomeAssistantError("unexpected WebSocket handshake")
                    await socket.send(json.dumps({"type": "auth", "access_token": self._token}))
                    auth = json.loads(await socket.recv())
                    if auth.get("type") != "auth_ok":
                        raise HomeAssistantError("Home Assistant WebSocket authentication failed")
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
                        message = json.loads(raw)
                        if message.get("type") == "event":
                            yield VisionRequestEvent.model_validate(message["event"]["data"])
            except (
                OSError,
                websockets.WebSocketException,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                LOGGER.warning("Vision event connection interrupted: %s", type(exc).__name__)
                await asyncio.sleep(15)
