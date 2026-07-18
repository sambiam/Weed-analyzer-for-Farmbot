# Companion FarmBot integration contract

Contract version: **farmbot-vision-v2**. Minimum compatible companion FarmBot
integration release: **1.2.0** (the release that implements the returned-JPEG
checksum, source/oriented/processed dimensions, resize scales and processed
calibration). Version 1.2.0 of the companion integration in the sibling
`Farmbot-for-Home-Assistant` repository implements this contract.

All actions are in the `farmbot` domain. Response actions must support Home Assistant service response data. Unknown, invalid, or unauthorised fields must fail rather than be coerced. Timestamps are ISO-8601. The integration remains the only component that talks to FarmBot APIs.

## `farmbot.list_vision_bots`

No input. Response:

```json
{"bots":[{"config_entry_id":"string","device_id":"string","name":"string"}]}
```

## `farmbot.get_vision_inventory`

Input: `{"config_entry_id":"string","image_lookback_hours":72}`.

Response fields:

- `device_id`, `generated_at`
- `plants[]`: `id`, `name`, `openfarm_slug`, `x`, `y`, `z`, `radius`, `plant_stage`, nullable `planted_at`, nullable `spread_curve_id`
- `images[]`: `id`, `created_at`, `processed`, and `meta` containing `x`, `y`, `z`, optional `name`.
  The app also tolerates a non-conforming shape observed from at least one companion
  integration build in production: `x`/`y`/`z`/`name` sent flat on the image object
  instead of nested under `meta`, and `processed` omitted entirely (treated as `true`).
  This is a compatibility shim in `InventoryImage._normalize`, not the target contract —
  new integration work should still emit the nested/complete shape above.
- `curves[]`: `id`, `name`, `type` (must be `spread`), and day-string to diameter mapping `data`
- `camera_calibration`: `available`, nullable positive `pixels_per_mm_x/y`, nullable `rotation_degrees`, nullable `offset_x_mm/y`, and (v2) `reference_width`, `reference_height`, `basis` (`reference_image` or `native_frame`)

When `available` is true both pixel scales are required. This is the **reference** (normalized) calibration: `pixels_per_mm_*` are stated relative to `reference_width` x `reference_height`. The app scales them to the processed resolution — a native scale is never applied directly to a resized frame. Image metadata coordinates are defined as the ground coordinate at image centre.

## `farmbot.get_vision_image`

Input: `{"config_entry_id":"string","image_id":456,"max_width":960,"max_height":720}`. `max_width`/`max_height` are the app's configured analysis resolution and are at most 1280 x 960.

Response (contract v2):

- `image_id`, `content_type` (only `image/jpeg`)
- lowercase hex `sha256` **over the returned JPEG bytes** (the app verifies the bytes it receives)
- optional `source_sha256` over the original download (format-checked only; never verified because the original is not shipped)
- `source_width`, `source_height` (before EXIF orientation)
- `oriented_width`, `oriented_height` (after EXIF orientation)
- `width`, `height` (processed, ≤ requested and ≤ 1280 x 960)
- `resize_scale_x` = `width / oriented_width`, `resize_scale_y` = `height / oriented_height`
- `image_base64`, and `meta` containing `x`, `y`, `z`, `created_at`
- optional `processed_calibration`: `{available, pixels_per_mm_x, pixels_per_mm_y, rotation_degrees, offset_x_mm, offset_y_mm, basis:"processed_image", width, height}` where `width`/`height` equal the returned image

The integration must resize before base64 encoding and must not return signed URLs. The app fetches sequentially and independently validates the checksum, base64, JPEG format, decoded dimensions, resize-scale consistency, aspect ratio, absence of upscaling, and payload/dimension limits. Older v1 responses (no `source_*`/`oriented_*`/`resize_scale_*`) are accepted as a legacy path but yield pixel-only diagnostics with no metric writes.

## `farmbot.apply_vision_radius`

Input:

```json
{"config_entry_id":"string","plant_id":123,"measurement_id":"UUID","expected_current_radius_mm":120.0,"recommended_radius_mm":185.0,"confidence":0.94,"apply":false}
```

The integration must validate bounds, authorization, and optimistic concurrency. A stale current radius must return HTTP-equivalent conflict semantics (409 or 412 through the service response path) or a structured failure that can be mapped to that condition. `apply:false` validates without mutation. It may return a response.

## `farmbot.upsert_vision_spread_curve`

Input:

```json
{"config_entry_id":"string","crop_slug":"lettuce","curve_id":null,"name":"[FarmBot Vision] Lettuce protection spread","data":{"1":30,"14":100,"30":300},"assign_to_plant_ids":[123,124],"apply":false}
```

Values are diameters in millimetres. A new curve name must use `[FarmBot Vision]`. The integration must reject modification of non-adopted user curves and validate app-owned curve IDs. `apply:false` validates only. FarmBot Vision 0.1.0 does not call this advanced action automatically.

## `farmbot.report_vision_status`

Input fields: `config_entry_id`, `available`, `status` (`idle|running|warning|error`), nullable `job_id`, nullable `last_completed_at`, integer `plants_analysed`, `recommendations`, `automatically_applied`, `uncertain`, and a non-sensitive `message` no longer than 240 characters. The integration should avoid creating recorder churn when the payload is unchanged.

## `farmbot_vision_request` event

Event data:

```json
{
  "config_entry_id": "string",
  "device_id": "string",
  "plant_ids": [],
  "mode": "recommend"
}
```

`device_id` is optional for backward compatibility. `mode` must be one of
`observe`, `recommend`, or `auto_radius`; every `plant_ids` value must be a
positive integer. An empty plant list means all eligible plants. The app
rejects unknown event fields. A malformed event is logged with sanitized field
and error-type details and skipped without reconnecting the active subscription,
so a later valid event is still processed. The integration should not emit
overlapping requests repeatedly.

## Exact companion-integration work still required

1. Register the five actions above with response support where specified.
2. Resolve `config_entry_id` only against loaded FarmBot entries and authorise every resource/write.
3. Add bounded image download/resize/JPEG encoding and SHA-256 response generation.
4. Expose plant, image, spread-curve, and calibration serializers with the exact field semantics above.
5. Implement optimistic radius concurrency and map stale writes distinctly.
6. Track adopted/FarmBot-Vision curve ownership and reject arbitrary user-curve modification.
7. Add status entities or diagnostics with update de-duplication.
8. Emit the request event from integration services/UI controls.
9. Add contract, malformed-response, authentication, reauthentication, stale-write, and permission tests.
10. Declare the minimum companion version once that integration release exists. **Done: companion 1.2.0 implements contract farmbot-vision-v2.**

## Contract v2 summary of required integration capabilities

1. `sha256` computed over the returned (re-encoded) JPEG bytes.
2. `source_width/height`, `oriented_width/height`, processed `width/height`.
3. `resize_scale_x/y` equal to processed÷oriented in each axis.
4. `processed_calibration` (basis `processed_image`) when calibration is known, plus reference dimensions on `camera_calibration`.

The minimum compatible companion integration version implementing all four is **1.2.0**.
