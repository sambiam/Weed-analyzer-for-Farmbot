from __future__ import annotations

import base64

import httpx
import numpy as np
import pytest
from conftest import encode_jpeg, vision_image_dict
from farmbot_vision.home_assistant import HomeAssistantClient
from farmbot_vision.models import VisionImage, VisionImageRequest


def _image(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :, 1] = 40
    return image


@pytest.mark.parametrize(
    ("preset_w", "preset_h"),
    [(640, 480), (960, 720), (1280, 960)],
)
def test_native_2592x1944_maps_to_each_preset(preset_w, preset_h):
    # A 2592x1944 native frame downscaled to the preset must validate with
    # consistent resize scales.
    payload = vision_image_dict(
        _image(preset_w, preset_h),
        source_wh=(2592, 1944),
        oriented_wh=(2592, 1944),
    )
    model = VisionImage.model_validate(payload)
    assert model.full_metadata
    assert model.resize_scale_x == pytest.approx(preset_w / 2592, rel=1e-6)
    assert model.width == preset_w and model.height == preset_h


def test_non_4_3_aspect_scaling_is_accepted():
    # 1920x1080 (16:9) oriented frame scaled to 640x360 preserving aspect.
    payload = vision_image_dict(
        _image(640, 360),
        source_wh=(1920, 1080),
        oriented_wh=(1920, 1080),
    )
    model = VisionImage.model_validate(payload)
    assert model.full_metadata


def test_inconsistent_resize_scales_are_rejected():
    payload = vision_image_dict(
        _image(640, 480),
        source_wh=(2592, 1944),
        oriented_wh=(2592, 1944),
        resize_override=(0.9, 0.9),  # wildly wrong versus 640/2592
    )
    with pytest.raises(ValueError, match="resize_scale"):
        VisionImage.model_validate(payload)


def test_distorted_aspect_ratio_is_rejected():
    payload = vision_image_dict(
        _image(640, 480),
        source_wh=(1280, 960),
        oriented_wh=(1280, 960),
        resize_override=(0.5, 0.25),  # anisotropic
    )
    with pytest.raises(ValueError):
        VisionImage.model_validate(payload)


def test_unexpected_upscaling_is_rejected():
    payload = vision_image_dict(
        _image(1280, 960),
        source_wh=(640, 480),
        oriented_wh=(640, 480),
        resize_override=(2.0, 2.0),
    )
    with pytest.raises(ValueError, match="upscaling"):
        VisionImage.model_validate(payload)


def test_oversized_processed_dimensions_are_rejected():
    payload = vision_image_dict(_image(1280, 960))
    payload["width"] = 1281
    with pytest.raises(ValueError):
        VisionImage.model_validate(payload)


def test_partial_metadata_is_a_contract_error():
    payload = vision_image_dict(_image(640, 480))
    del payload["oriented_width"]  # break the completeness of the v2 set
    with pytest.raises(ValueError, match="incomplete image contract"):
        VisionImage.model_validate(payload)


def test_legacy_v1_image_is_accepted_as_legacy():
    payload = vision_image_dict(_image(640, 480), with_v2=False)
    model = VisionImage.model_validate(payload)
    assert not model.full_metadata


def test_source_sha256_format_is_validated_but_not_verified():
    payload = vision_image_dict(_image(640, 480))
    payload["source_sha256"] = "00" * 32  # valid 64-hex format
    assert VisionImage.model_validate(payload).source_sha256 == "00" * 32
    payload["source_sha256"] = "nothex"
    with pytest.raises(ValueError):
        VisionImage.model_validate(payload)


# -------------------- client-side byte validation --------------------


def _client(payload: dict) -> HomeAssistantClient:
    def handler(_request):
        return httpx.Response(200, json=payload)

    client = HomeAssistantClient(token="test", base_url="http://test")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _request() -> VisionImageRequest:
    return VisionImageRequest(config_entry_id="e", image_id=1, max_width=1280, max_height=960)


@pytest.mark.asyncio
async def test_correct_checksum_and_dimensions_are_accepted():
    client = _client(vision_image_dict(_image(960, 720)))
    result = await client.image(_request(), 5 * 1024 * 1024)
    assert result.width == 960 and result.height == 720
    await client.close()


@pytest.mark.asyncio
async def test_incorrect_checksum_is_rejected():
    payload = vision_image_dict(_image(640, 480), sha_override="ab" * 32)
    client = _client(payload)
    with pytest.raises(Exception, match="checksum"):
        await client.image(_request(), 5 * 1024 * 1024)
    await client.close()


@pytest.mark.asyncio
async def test_invalid_base64_is_rejected():
    payload = vision_image_dict(_image(640, 480), base64_override="!!!not base64!!!")
    client = _client(payload)
    with pytest.raises(Exception, match="base64"):
        await client.image(_request(), 5 * 1024 * 1024)
    await client.close()


@pytest.mark.asyncio
async def test_decoded_dimension_mismatch_is_rejected():
    # Report 800x600 but ship a 640x480 JPEG.
    real = encode_jpeg(_image(640, 480))
    payload = vision_image_dict(_image(640, 480))
    payload["width"], payload["height"] = 800, 600
    payload["source_width"] = payload["oriented_width"] = 800
    payload["source_height"] = payload["oriented_height"] = 600
    payload["resize_scale_x"] = payload["resize_scale_y"] = 1.0
    import hashlib

    payload["sha256"] = hashlib.sha256(real).hexdigest()
    payload["image_base64"] = base64.b64encode(real).decode("ascii")
    client = _client(payload)
    with pytest.raises(Exception, match="dimensions do not match"):
        await client.image(_request(), 5 * 1024 * 1024)
    await client.close()


@pytest.mark.asyncio
async def test_response_larger_than_requested_is_rejected():
    payload = vision_image_dict(_image(1280, 960))
    client = _client(payload)
    request = VisionImageRequest(config_entry_id="e", image_id=1, max_width=640, max_height=480)
    with pytest.raises(Exception, match="exceeds requested"):
        await client.image(request, 5 * 1024 * 1024)
    await client.close()
