import httpx
import pytest
from farmbot_vision.home_assistant import HomeAssistantClient, StaleRadiusError
from farmbot_vision.models import ApplyRadiusRequest


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
